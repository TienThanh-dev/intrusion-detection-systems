import os
from dotenv import load_dotenv

load_dotenv()

MODEL_MAP = {
    "binary_lgbm": os.getenv("MODEL_BINARY_LGBM"),
    "binary_rf": os.getenv("MODEL_BINARY_RF"),
    "multi_lgbm": os.getenv("MODEL_MULTI_LGBM"),
    "multi_rf": os.getenv("MODEL_MULTI_RF"),
}