import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Curated list shown in the UI dropdown. Keep in sync with Groq's available models.
GROQ_AVAILABLE_MODELS: list[str] = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

HDBSCAN_MIN_CLUSTER_SIZE: int = 2
HDBSCAN_MIN_SAMPLES: int = 1

MAX_CLAIMS_PER_PAPER: int = 15
MAX_PAPER_CHARS: int = 24_000

OUTPUT_DIR: Path = Path(os.getenv("CPC_OUTPUT_DIR", str(BASE_DIR / "output")))
PROMPTS_DIR: Path = BASE_DIR / "cpc_agent" / "prompts"
