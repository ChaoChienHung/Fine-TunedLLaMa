import os
from datetime import datetime


# ---------
# Cache
# ---------
CACHE_DIR = "./cache/"
DATA_DIR = os.path.join(CACHE_DIR, "data")
MODEL_DIR = os.path.join(CACHE_DIR, "model")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")


# --------
# Model
# --------
MODEL = "meta-llama/Llama-2-7b-hf"
LORA_PATH = "./lora_out_adapter"


# ----------------
# Hyperparameters
# ----------------
MAX_LENGTH = 2048

# --------------------------------------
# Get today's date in YYYY-MM-DD format
# --------------------------------------
DATETIME = datetime.today().strftime('%Y-%m-%d %H-%M-%S')
