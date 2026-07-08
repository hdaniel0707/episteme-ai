"""Training loop for the text-transformer workflows (W5/W6, plan section 3.3).

Fine-tunes the chosen backbone (+head, or LoRA adapters) directly on raw title/abstract text:
on-the-fly tokenization (never persisted to disk), fp16 autocast + GradScaler, gradient
accumulation to a target effective batch of 64, linear-warmup + cosine-decay LR, and early
stopping on val Spearman (reg) / val macro-F1 (clf). CUDA OOM is caught once and retried with a
halved micro-batch / doubled accumulation (plan section 9.6).

``TextTransformerTrainer.run`` writes ``seed_<i>/`` with an HF-format ``checkpoint/``,
``result.json`` and ``training_curve.png`` and returns the test metrics dict.
"""

from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from workflow.src.iwf.data.dataset import split_masks
from workflow.src.iwf.evaluation.metrics import classification_metrics, regression_metrics
from workflow.src.iwf.evaluation.plots import plot_tfm_loss_lr, plot_tfm_training_curve


def reg_metrics(y_true, y_pred) -> dict:
    return regression_metrics(y_true, y_pred)


def clf_metrics(y_true, y_pred, n_classes: int) -> dict:
    # class ids are enough here; the shared function only needs the class count
    return classification_metrics(y_true, y_pred, [str(i) for i in range(n_classes)])


def _val_score(task: str, metrics: dict) -> float:
    return metrics["macro_f1"] if task == "clf" else metrics["spearman"]


class TextTransformerTrainer:
    """Fine-tune one text-transformer seed end-to-end.

    ``progress`` is an optional callable receiving small dicts (seed/epoch/best metric) so a live
    display can show progress. ``model`` lets a caller inject a pre-built (mock) model so the
    offline smoke test never downloads weights.
    """

    EFFECTIVE_BATCH = 64
    PATIENCE = 3

    def __init__(self, cfg, task: str, *, progress=None, model=None):
        self.cfg = cfg
        self.task = task                       # "clf" | "reg"
        self.progress = progress
        self._model_override = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------------------------------- data
    def _prepare_arrays(self, df: pd.DataFrame, class_names):
        titles = df["title"].fillna("").astype(str).to_numpy()
        abstracts = df["abstract"].fillna("").astype(str).to_numpy()
        masks = split_masks(df)
        if self.task == "clf":
            mapping = {name: i for i, name in enumerate(class_names)}
            y = df["class_label"].map(mapping)
            keep = y.notna().to_numpy()
            y = y.fillna(0).astype(int).to_numpy()
            out_dim = len(class_names)
        else:
            y = pd.to_numeric(df["target"], errors="coerce").to_numpy(dtype=np.float32)
            keep = np.isfinite(y)
            out_dim = 1
        # restrict every split mask to rows with a usable label
        masks = {sp: (m & keep) for sp, m in masks.items()}
        return titles, abstracts, y, masks, out_dim

    def _build_model(self, out_dim: int):
        if self._model_override is not None:
            return self._model_override
        from workflow.src.iwf.registry import get_module
        cls = get_module("text_transformer", self.cfg.text_transformer_arch)
        return cls(self.cfg, out_dim, self.task)

    # ----------------------------------------------------------------- public entry
    def run(self, df: pd.DataFrame, *, seed: int, output_dir, class_names=None) -> dict:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        class_names = list(class_names or getattr(self.cfg, "class_names", []) or [])
        titles, abstracts, y, masks, out_dim = self._prepare_arrays(df, class_names)

        micro = max(1, int(self.cfg.text_transformer_batch_size))
        grad_accum = max(1, round(self.EFFECTIVE_BATCH / micro))

        try:
            return self._fit(titles, abstracts, y, masks, out_dim, class_names,
                             seed, out_dir, micro, grad_accum)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            micro2 = max(1, micro // 2)
            grad_accum2 = grad_accum * 2
            print(f"[train_tfm] CUDA OOM at micro-batch {micro}; retrying once with micro-batch "
                  f"{micro2} x grad_accum {grad_accum2} (effective batch unchanged).")
            try:
                return self._fit(titles, abstracts, y, masks, out_dim, class_names,
                                 seed, out_dir, micro2, grad_accum2)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("[train_tfm] CUDA OOM again after the retry. Lower TEXT_TRANSFORMER_BATCH_SIZE "
                      "or TEXT_TRANSFORMER_MAX_LEN, or use a LoRA arch.", file=sys.stderr)
                raise SystemExit(1)

    # ----------------------------------------------------------------- pre-tokenization (CPU+GPU)
    def _pretokenize(self, model, titles, abstracts):
        """Tokenize ALL texts once into fixed-length CPU tensors (pinned for async GPU transfer).

        This eliminates re-tokenizing every batch every epoch — the biggest CPU bottleneck.
        Tokenization runs on CPU; the tensors are pinned so ``.to(device, non_blocking=True)``
        overlaps the CPU→GPU transfer with GPU compute.
        """
        max_len = int(getattr(self.cfg, "text_transformer_max_len", 512) or 512)
        chunk = 256
        all_ids, all_mask, all_types = [], [], []
        has_types = None

        for start in range(0, len(titles), chunk):
            end = min(start + chunk, len(titles))
            batch = model.tokenize(titles[start:end], abstracts[start:end])
            ids = batch["input_ids"]
            mask = batch["attention_mask"]
            types = batch.get("token_type_ids")

            cur_len = ids.shape[1]
            if cur_len < max_len:
                pad_len = max_len - cur_len
                ids = torch.nn.functional.pad(ids, (0, pad_len), value=0)
                mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
                if types is not None:
                    types = torch.nn.functional.pad(types, (0, pad_len), value=0)
            elif cur_len > max_len:
                ids = ids[:, :max_len]
                mask = mask[:, :max_len]
                if types is not None:
                    types = types[:, :max_len]

            all_ids.append(ids)
            all_mask.append(mask)
            if has_types is None:
                has_types = types is not None
            if types is not None:
                all_types.append(types)

        result = {
            "input_ids": torch.cat(all_ids),
            "attention_mask": torch.cat(all_mask),
        }
        if has_types and all_types:
            result["token_type_ids"] = torch.cat(all_types)

        if self.device == "cuda":
            result = {k: v.pin_memory() for k, v in result.items()}
        return result

    def _batch_from_tokens(self, tokens, sel) -> dict:
        """Select a sub-batch from pre-tokenized tensors and transfer to GPU asynchronously."""
        nb = self.device == "cuda"
        return {k: v[sel].to(self.device, non_blocking=nb) for k, v in tokens.items()}

    # ----------------------------------------------------------------- training
    def _fit(self, titles, abstracts, y, masks, out_dim, class_names,
             seed, out_dir, micro, grad_accum) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.empty_cache()

        model = self._build_model(out_dim).to(self.device)

        # Pre-tokenize everything once on CPU before training starts
        print(f"[train_tfm] pre-tokenizing {len(titles)} texts on CPU...", end=" ", flush=True)
        tokens = self._pretokenize(model, titles, abstracts)
        print("done.")

        use_amp = self.device == "cuda"
        scaler = torch.amp.GradScaler(self.device, enabled=use_amp)

        if self.task == "clf":
            from workflow.src.iwf.models.classifier import class_weights
            cw = class_weights(y[masks["train"]], out_dim, self.cfg.class_weight,
                               self.cfg.effective_beta)
            weight_t = (torch.tensor(cw, dtype=torch.float32, device=self.device)
                        if cw is not None else None)
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=weight_t, label_smoothing=float(self.cfg.label_smoothing))
        else:
            loss_fn = torch.nn.MSELoss()

        opt = torch.optim.AdamW(model.trainable_parameters(),
                                lr=float(self.cfg.text_transformer_lr),
                                weight_decay=float(self.cfg.weight_decay))

        train_idx = np.where(masks["train"])[0]
        n_micro = max(1, math.ceil(len(train_idx) / micro))
        steps_per_epoch = max(1, math.ceil(n_micro / grad_accum))
        epochs = max(1, int(self.cfg.text_transformer_epochs))
        total_steps = epochs * steps_per_epoch
        warmup = max(1, int(round(float(self.cfg.text_transformer_warmup_ratio) * total_steps)))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            prog = (step - warmup) / max(1, total_steps - warmup)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        history = {"step_loss": [], "step_lr": [], "val_metric": [], "best_epoch": 0}
        best_score, best_metrics, no_improve = -math.inf, None, 0
        rng = np.random.default_rng(seed)
        eff_batch = micro * grad_accum

        for epoch in range(epochs):
            model.train()
            order = train_idx.copy()
            rng.shuffle(order)
            opt.zero_grad(set_to_none=True)
            for b in range(n_micro):
                sel = order[b * micro:(b + 1) * micro]
                if len(sel) == 0:
                    continue
                batch = self._batch_from_tokens(tokens, sel)
                yb = self._targets(y[sel])
                with torch.autocast(device_type=self.device, enabled=use_amp):
                    out = model(batch)
                    loss = loss_fn(out, yb) / grad_accum
                scaler.scale(loss).backward()
                history["step_loss"].append(float(loss.item()) * grad_accum)
                if (b + 1) % grad_accum == 0 or (b + 1) == n_micro:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    scheduler.step()
                    history["step_lr"].append(float(scheduler.get_last_lr()[0]))

            val_metrics = self._evaluate_from_tokens(model, tokens, y, masks["val"],
                                                     out_dim, micro)
            score = _val_score(self.task, val_metrics)
            history["val_metric"].append(score)
            if self.progress is not None:
                self.progress({"epoch": epoch + 1, "epochs": epochs, "val_metric": score,
                               "eff_batch": eff_batch})

            if score > best_score + 1e-6:
                best_score, no_improve = score, 0
                history["best_epoch"] = epoch + 1
                model.save(out_dir)
                # Evaluate all three splits on the current best model
                test_m, y_test_arr, y_pred_arr = self._evaluate_from_tokens(
                    model, tokens, y, masks["test"], out_dim, micro, return_preds=True)
                train_m, y_train_arr, y_pred_train_arr = self._evaluate_from_tokens(
                    model, tokens, y, masks["train"], out_dim, micro, return_preds=True)
                val_m_full, y_val_arr, y_pred_val_arr = self._evaluate_from_tokens(
                    model, tokens, y, masks["val"], out_dim, micro, return_preds=True)
                n_val = int(masks["val"].sum())
                n_train = int(masks["train"].sum())
                val_extra = {"n": n_val}
                if self.task == "clf":
                    val_extra["correct"] = int(round(float(val_m_full.get("accuracy", 0)) * n_val))
                best_metrics = {
                    **test_m,
                    "best_val_metric": float(best_score),
                    "best_epoch": epoch + 1,
                    "effective_batch": eff_batch,
                    "train": {**train_m, "n": n_train,
                              "y_true": y_train_arr.tolist(), "y_pred": y_pred_train_arr.tolist()},
                    "val": {**val_m_full, **val_extra,
                            "y_true": y_val_arr.tolist(), "y_pred": y_pred_val_arr.tolist()},
                    "test": {**test_m,
                             "y_true": y_test_arr.tolist(), "y_pred": y_pred_arr.tolist()},
                    "y_test": y_test_arr.tolist(),   # backward compat root-level keys
                    "y_pred": y_pred_arr.tolist(),
                }
            else:
                no_improve += 1
                if no_improve >= self.PATIENCE:
                    break

        if best_metrics is None:
            test_m, y_test_arr, y_pred_arr = self._evaluate_from_tokens(
                model, tokens, y, masks["test"], out_dim, micro, return_preds=True)
            train_m, y_train_arr, y_pred_train_arr = self._evaluate_from_tokens(
                model, tokens, y, masks["train"], out_dim, micro, return_preds=True)
            val_m, y_val_arr, y_pred_val_arr = self._evaluate_from_tokens(
                model, tokens, y, masks["val"], out_dim, micro, return_preds=True)
            best_metrics = {
                **test_m,
                "train": {**train_m, "y_true": y_train_arr.tolist(), "y_pred": y_pred_train_arr.tolist()},
                "val": {**val_m, "y_true": y_val_arr.tolist(), "y_pred": y_pred_val_arr.tolist()},
                "test": {**test_m, "y_true": y_test_arr.tolist(), "y_pred": y_pred_arr.tolist()},
                "y_test": y_test_arr.tolist(),
                "y_pred": y_pred_arr.tolist(),
            }

        best_metrics["history"] = history
        best_metrics["arch"] = self.cfg.text_transformer_arch
        best_metrics["seed"] = int(seed)
        (out_dir / "result.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
        try:
            plot_tfm_training_curve(history, out_dir / "training_curve.png")
            plot_tfm_loss_lr(history, out_dir / "loss_lr_curve.png")
        except Exception as e:
            print(f"[train_tfm] curve plot skipped ({e})")

        del model, tokens
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return best_metrics

    # ----------------------------------------------------------------- helpers
    def _targets(self, ys) -> torch.Tensor:
        if self.task == "clf":
            return torch.tensor(ys, dtype=torch.long, device=self.device)
        return torch.tensor(ys, dtype=torch.float32, device=self.device).view(-1, 1)

    @torch.no_grad()
    def _evaluate_from_tokens(self, model, tokens, y, mask, out_dim, micro,
                               *, return_preds: bool = False):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            empty = np.array([], dtype=(np.int64 if self.task == "clf" else np.float32))
            metrics = (clf_metrics([0], [0], out_dim) if self.task == "clf"
                       else reg_metrics([], []))
            return (metrics, empty, empty) if return_preds else metrics
        model.eval()
        preds = []
        for b in range(math.ceil(len(idx) / micro)):
            sel = idx[b * micro:(b + 1) * micro]
            batch = self._batch_from_tokens(tokens, sel)
            out = model(batch)
            if self.task == "clf":
                preds.append(out.argmax(dim=1).cpu().numpy())
            else:
                preds.append(out.view(-1).float().cpu().numpy())
        pred = np.concatenate(preds)
        y_true = y[idx]
        metrics = clf_metrics(y_true, pred, out_dim) if self.task == "clf" else reg_metrics(y_true, pred)
        return (metrics, y_true, pred) if return_preds else metrics
