import anthropic
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


class RephrasedPaper(BaseModel):
    title_rephrased: str
    abstract_rephrased: str


SYSTEM_PROMPT = """You are an expert scientific editor.

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


def _rephrase_openai(
    client: OpenAI, model: str, title: str, abstract: str
) -> RephrasedPaper | None:
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {title}\n\nAbstract: {abstract}"},
        ],
        response_format=RephrasedPaper,
    )
    return response.choices[0].message.parsed


def _rephrase_anthropic(
    client: anthropic.Anthropic, model: str, title: str, abstract: str
) -> RephrasedPaper | None:
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
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
    return RephrasedPaper(
        title_rephrased=inp["title_rephrased"],
        abstract_rephrased=inp["abstract_rephrased"],
    )


def add_ai_generated_data(
    df: pd.DataFrame,
    models: list[str] = [
        "claude-haiku-4-5-20251001",
        "gpt-4o-mini",
    ],
) -> tuple[pd.DataFrame, int]:
    openai_client = OpenAI()
    anthropic_client = anthropic.Anthropic()

    new_rows = []
    total_dropped = 0

    for model in models:
        valid_indices = []

        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Generating [{model}]"
        ):
            title = row.get("title") or ""
            abstract = row.get("abstract") or ""

            if model.startswith("claude"):
                result = _rephrase_anthropic(anthropic_client, model, title, abstract)
            else:
                result = _rephrase_openai(openai_client, model, title, abstract)

            if (
                not result
                or not result.title_rephrased
                or not result.abstract_rephrased
            ):
                continue

            new_row = row.copy()
            new_row["title"] = result.title_rephrased
            new_row["abstract"] = result.abstract_rephrased
            new_row["generated_by"] = model
            new_rows.append(new_row)
            valid_indices.append(idx)

        total_dropped += len(df) - len(valid_indices)

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True), total_dropped
