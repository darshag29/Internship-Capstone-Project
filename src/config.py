"""
Central configuration for the Stock Market Intelligence System.

All API keys are loaded from environment variables (via a local .env file).
Never hardcode keys in source files.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- Paths -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# ---- Assets ------------------------------------------------------------
# ^BSESN = Sensex, add/replace your 5th chosen stock below (default HDFC Bank)
ASSETS = ["^BSESN", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

ASSET_DISPLAY_NAMES = {
    "^BSESN": "Sensex",
    "RELIANCE.NS": "Reliance",
    "TCS.NS": "TCS",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
}

# Company search terms for news (index name doesn't work well on NewsAPI)
NEWS_QUERY_MAP = {
    "^BSESN": "Sensex OR BSE India",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
}

# ---- API Keys (set these in a .env file, see .env.example) ------------
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ---- Business cost assumptions (Task 9) --------------------------------
COST_FALSE_POSITIVE = 100  # predicted UP, actually DOWN -> bad trade cost
COST_FALSE_NEGATIVE = 50   # predicted DOWN, actually UP -> missed opportunity cost

# ---- Modeling ------------------------------------------------------------
TEST_SIZE_FRACTION = 0.2   # last 20% of time-ordered data held out for testing
RANDOM_STATE = 42
