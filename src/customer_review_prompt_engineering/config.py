from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "Reviews.csv"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "prompt_versions.yaml"


def load_settings() -> dict:
    load_dotenv()
    return {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "data_path": Path(os.getenv("DATA_PATH", str(DEFAULT_DATA_PATH))),
    }
