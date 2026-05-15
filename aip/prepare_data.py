from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from aip.utils.sqlite_utils import load_db
from aip.utils.genai_utils import add_ai_generated_data
from aip.utils.parquet_utils import save_parquet

BASE_DIR = Path(__file__).parent.parent
DB_DIR = BASE_DIR / "data"

######################################
ADD_GEN_AI = False
DB_ID = "0515_A_500"
DB_PATH = DB_DIR / "sql" / (DB_ID + ".db")
PAQUET_PATH = DB_DIR / "paquet" / (DB_ID + ".parquet")
AI_PAQUET_PATH = DB_DIR / "paquet" / (DB_ID + "_AI.parquet")

REQUIRED_FIEDLS = ["embedding", "title", "abstract", "year", "best_author_h_index"]

if __name__ == "__main__":

    df = load_db(DB_PATH, require_fields=REQUIRED_FIEDLS, year_from=1900, year_to=2026)

    if ADD_GEN_AI:
        df, removed_article_n = add_ai_generated_data(df, debug_n=5)
        print(f"Removed article number: {removed_article_n}")

        save_parquet(df, AI_PAQUET_PATH)
    else:
        save_parquet(df, PAQUET_PATH)
