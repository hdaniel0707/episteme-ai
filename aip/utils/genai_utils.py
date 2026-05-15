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


def add_ai_generated_data(
    df: pd.DataFrame, model: str = "gpt-4o-mini", debug_n: int | None = None
) -> tuple[pd.DataFrame, int]:
    client = OpenAI()

    rows_df = df.head(debug_n) if debug_n is not None else df
    new_rows = []
    valid_indices = []

    for idx, row in tqdm(rows_df.iterrows(), total=len(rows_df), desc="Generating"):
        title = row.get("title") or ""
        abstract = row.get("abstract") or ""

        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Title: {title}\n\nAbstract: {abstract}",
                },
            ],
            response_format=RephrasedPaper,
        )

        result = response.choices[0].message.parsed
        if not result or not result.title_rephrased or not result.abstract_rephrased:
            continue

        new_row = row.copy()
        new_row["title"] = result.title_rephrased
        new_row["abstract"] = result.abstract_rephrased
        new_row["generated_by"] = model
        new_rows.append(new_row)
        valid_indices.append(idx)

    dropped_indices = [i for i in rows_df.index if i not in valid_indices]
    df_filtered = df.drop(index=dropped_indices)
    return pd.concat([df_filtered, pd.DataFrame(new_rows)], ignore_index=True), len(dropped_indices)
