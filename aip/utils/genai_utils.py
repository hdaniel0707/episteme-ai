import anthropic
import pandas as pd
from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


class RephrasedPaper(BaseModel):
    title_rephrased: str
    abstract_rephrased: str


SYSTEM_PROMPT_NORMAL = """You are an expert scientific editor.

Your task is to rewrite research paper titles and abstracts while strictly preserving their original meaning.

Requirements:
- Preserve all scientific claims, findings, conclusions, and technical meaning.
- Do not add new information.
- Do not remove important information.
- Do not change numerical values, metrics, dataset names, model names, or scientific terminology unless only grammatical reformulation is needed.
- Rewrite aggressively at the sentence and phrase level to maximize lexical and syntactic variation while preserving semantics.
- Avoid copying long phrases from the original text.
- The rewritten text must remain fluent, natural, and publication-quality English.
- The rewritten abstract should have approximately the same length as the original.
- The rewritten title should remain concise and scientifically accurate."""

SYSTEM_PROMPT_AGRESSIVE = """You are an expert scientific editor specializing in creative reformulation.

Your task is to rewrite research paper titles and abstracts in a way that reads as if written by a different author — same findings, different voice and structure.

Requirements:
- Preserve all scientific claims, findings, conclusions, and technical meaning without exception.
- Do not introduce new information or omit important content.
- Do not alter numerical values, metrics, dataset names, model names, or domain-specific terminology unless a purely grammatical fix is needed.
- Restructure sentences aggressively: change clause order, split or merge sentences, shift from active to passive voice (or vice versa), and reframe how ideas are introduced.
- Replace vocabulary at every opportunity — synonyms, paraphrases, alternative phrasings — so that the rewritten text shares minimal surface overlap with the original.
- Reorganize the flow of the abstract where possible: if the original presents A then B then C, consider leading with a different element or reordering supporting points, as long as logical coherence and scientific accuracy are fully maintained.
- The final text must be fluent, natural, and of publication quality.
- The rewritten abstract should match the original in length (±10%).
- The rewritten title must remain concise, informative, and scientifically precise."""

SYSTEM_PROMPTS: dict[str, str] = {
    "normal": SYSTEM_PROMPT_NORMAL,
    "aggressive": SYSTEM_PROMPT_AGRESSIVE,
}


def _rephrase_openai(
    client: OpenAI, model: str, system_prompt: str, title: str, abstract: str
) -> RephrasedPaper | None:
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"},
            ],
            response_format=RephrasedPaper,
        )
        return response.choices[0].message.parsed
    except Exception as e:
        print(f"[openai] {e}")
        return None


def _rephrase_anthropic(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    title: str,
    abstract: str,
) -> RephrasedPaper | None:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            tools=[
                {
                    "name": "rephrase_paper",
                    "description": "Return the rephrased title and abstract.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "title_rephrased": {"type": "string"},
                            "abstract_rephrased": {"type": "string"},
                        },
                        "required": ["title_rephrased", "abstract_rephrased"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "rephrase_paper"},
            messages=[
                {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"}
            ],
        )
        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_block is None:
            return None
        inp: dict = tool_block.input  # type: ignore[assignment]
        if "title_rephrased" not in inp or "abstract_rephrased" not in inp:
            return None
        return RephrasedPaper(
            title_rephrased=inp["title_rephrased"],
            abstract_rephrased=inp["abstract_rephrased"],
        )
    except Exception as e:
        print(f"[anthropic] {e}")
        return None


def _rephrase_gemini(
    client: genai.Client,
    model: str,
    system_prompt: str,
    title: str,
    abstract: str,
) -> RephrasedPaper | None:
    try:
        response = client.models.generate_content(
            model=model,
            contents=f"Title: {title}\n\nAbstract: {abstract}",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=RephrasedPaper,
            ),
        )
        return response.parsed
    except Exception as e:
        print(f"[gemini] {e}")
        return None


def add_ai_generated_data(
    df: pd.DataFrame,
    model_prompt_pairs: list[tuple[str, str]] = [
        ("gemini-2.5-flash-lite", "normal"),
        ("claude-haiku-4-5-20251001", "normal"),
        ("gpt-4o-mini", "normal"),
    ],
) -> tuple[pd.DataFrame, int]:
    openai_client = OpenAI()
    anthropic_client = anthropic.Anthropic()
    gemini_client = genai.Client()

    # results[idx][(model, prompt_name)] = RephrasedPaper — only populated if all models succeed
    results: dict = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating rephrases"):
        title = row.get("title") or ""
        abstract = row.get("abstract") or ""
        row_results = {}

        for model, prompt_name in model_prompt_pairs:
            system_prompt = SYSTEM_PROMPTS[prompt_name]

            if model.startswith("claude"):
                result = _rephrase_anthropic(
                    anthropic_client, model, system_prompt, title, abstract
                )
            elif model.startswith("gemini"):
                result = _rephrase_gemini(
                    gemini_client, model, system_prompt, title, abstract
                )
            elif model.startswith("gpt") or model.startswith("o"):
                result = _rephrase_openai(
                    openai_client, model, system_prompt, title, abstract
                )
            else:
                raise ValueError(f"Unsupported model: {model}")

            if result and result.title_rephrased and result.abstract_rephrased:
                row_results[(model, prompt_name)] = result
            else:
                break  # any failure → discard this row entirely

        if len(row_results) == len(model_prompt_pairs):
            results[idx] = row_results

    valid_indices = list(results.keys())
    total_dropped = len(df) - len(valid_indices)

    new_rows = []
    for idx in valid_indices:
        row = df.loc[idx]
        for (model, prompt_name), result in results[idx].items():
            new_row = row.copy()
            new_row["title"] = result.title_rephrased
            new_row["abstract"] = result.abstract_rephrased
            new_row["generated_by"] = f"{model}:{prompt_name}"
            new_rows.append(new_row)

    return (
        pd.concat([df.loc[valid_indices], pd.DataFrame(new_rows)], ignore_index=True),
        total_dropped,
    )
