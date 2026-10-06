"""Central configuration for the question answering application."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Streamlit reloads this module when source changes, so local .env values are
# applied on the next source refresh without embedding secrets in Python code.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.getenv("QA_MODEL_NAME", "Jacaranda/YorubaLlama")
# Used only when the requested model is gated, inaccessible, or cannot be loaded.
FALLBACK_MODEL_NAME = os.getenv("QA_FALLBACK_MODEL_NAME", "deepset/roberta-base-squad2")
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "2048"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "64"))
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
SAMPLE_CONTEXT_PATH = PROJECT_ROOT / "sample_context.txt"
LOG_FILE = OUTPUT_DIR / "qa_application.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
HF_TOKEN = os.getenv("HF_TOKEN")

# Computed at runtime by utils.get_device_info(); kept here for discoverability.
DEVICE_INFORMATION = "Automatic: CUDA GPU when available, otherwise CPU"

for directory in (OUTPUT_DIR, MODEL_DIR, ASSETS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
