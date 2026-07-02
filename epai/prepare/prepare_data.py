from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from epai.utils.sqlite_utils import load_db
from epai.utils.genai_utils import add_ai_generated_data
from epai.utils.parquet_utils import save_parquet
from epai.utils.embedding_utils import add_embedding

BASE_DIR = Path(__file__).parent.parent.parent
DB_DIR = BASE_DIR / "data"

######################################
ADD_GEN_AI = True
ADD_EMB_SPECTER2 = True
DB_ID = "0523_A_10"
DB_PATH = DB_DIR / "sql" / (DB_ID + ".db")
PARQUET_PATH = DB_DIR / "parquet" / (DB_ID + ".parquet")
AI_PARQUET_PATH = DB_DIR / "parquet" / (DB_ID + "_AI.parquet")

REQUIRED_FIEDLS = ["embedding", "title", "abstract", "year", "best_author_h_index"]

if __name__ == "__main__":

    df = load_db(DB_PATH, require_fields=REQUIRED_FIEDLS, year_from=1900, year_to=2026)
    # DEBUG
    # df = df.head(5)

    if ADD_GEN_AI:
        df, removed_article_n = add_ai_generated_data(
            df,
            model_prompt_pairs=[
                ("gemini-2.5-flash-lite", "normal"),
                ("claude-haiku-4-5-20251001", "normal"),
                ("gpt-4o-mini", "normal"),
                ("gpt-4o-mini", "aggressive"),
            ],
        )
        print(f"Removed article number: {removed_article_n}")

        if ADD_EMB_SPECTER2:
            df = add_embedding(df, emb_type="specter2")

        save_parquet(df, AI_PARQUET_PATH)
    else:
        if ADD_EMB_SPECTER2:
            df = add_embedding(df, emb_type="specter2")
        save_parquet(df, PARQUET_PATH)
