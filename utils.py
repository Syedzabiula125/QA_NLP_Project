"""Shared utilities for I/O, logging, results, and visualizations."""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

import config


def configure_logging() -> logging.Logger:
    """Configure a reusable file and console logger exactly once."""
    logger = logging.getLogger("qa_application")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


LOGGER = configure_logging()


def get_device_info() -> dict[str, Any]:
    """Report Torch and TensorFlow accelerator availability without failing imports."""
    info: dict[str, Any] = {"torch_cuda": False, "tensorflow_gpus": [], "device": "cpu"}
    try:
        import torch

        info["torch_cuda"] = torch.cuda.is_available()
        if info["torch_cuda"]:
            info["device"] = "cuda"
            info["torch_device_name"] = torch.cuda.get_device_name(0)
    except (ImportError, RuntimeError) as exc:
        info["torch_error"] = str(exc)
    try:
        import tensorflow as tf

        info["tensorflow_gpus"] = [gpu.name for gpu in tf.config.list_physical_devices("GPU")]
    except (ImportError, RuntimeError) as exc:
        info["tensorflow_error"] = str(exc)
    return info


def read_context_file(path: str | Path = config.SAMPLE_CONTEXT_PATH) -> str:
    """Return UTF-8 context from *path*, with a helpful missing-file error."""
    context_path = Path(path)
    if not context_path.is_file():
        raise FileNotFoundError(f"Context file not found: {context_path}")
    content = context_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Context file is empty: {context_path}")
    return content


def parse_batch_questions(raw_questions: str) -> list[str]:
    """Split newline-separated questions and discard blank entries."""
    return [line.strip() for line in raw_questions.splitlines() if line.strip()]


def safe_filename(value: str) -> str:
    """Convert arbitrary text into a portable filename component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or "results"


def results_to_frame(results: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Create a stable tabular representation of QA results."""
    rows = list(results)
    columns = [
        "question", "answer", "confidence", "processing_time_seconds",
        "start", "end", "mode", "error",
    ]
    return pd.DataFrame(rows, columns=columns)


def save_results_csv(results: Iterable[dict[str, Any]], prefix: str = "qa_results") -> Path:
    """Persist QA results in outputs and return the created path."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = config.OUTPUT_DIR / f"{safe_filename(prefix)}_{stamp}.csv"
    results_to_frame(results).to_csv(destination, index=False, encoding="utf-8")
    LOGGER.info("Saved results CSV: %s", destination)
    return destination


def create_visualizations(results: Iterable[dict[str, Any]]) -> dict[str, plt.Figure]:
    """Return bar, histogram, line, pie, and box figures for usable results."""
    frame = results_to_frame(results).copy()
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
    frame["processing_time_seconds"] = pd.to_numeric(
        frame["processing_time_seconds"], errors="coerce"
    ).fillna(0.0)
    labels = [f"Q{i + 1}" for i in range(len(frame))]
    figures: dict[str, plt.Figure] = {}

    def finish(fig: plt.Figure, title: str) -> plt.Figure:
        fig.suptitle(title, fontweight="bold")
        fig.tight_layout()
        return fig

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, frame["confidence"], color="#4C78A8")
    axis.set_ylabel("Confidence")
    axis.set_ylim(0, 1)
    figures["Confidence bar chart"] = finish(fig, "Confidence by Question")

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.hist(frame["confidence"], bins=min(10, max(1, len(frame))), color="#59A14F", edgecolor="white")
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Count")
    figures["Confidence histogram"] = finish(fig, "Confidence Distribution")

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(labels, frame["processing_time_seconds"], marker="o", color="#E15759")
    axis.set_ylabel("Seconds")
    figures["Inference time line chart"] = finish(fig, "Inference Time by Question")

    fig, axis = plt.subplots(figsize=(6, 5))
    values = frame["processing_time_seconds"].clip(lower=0)
    if values.sum() == 0:
        values = pd.Series([1] * len(frame))
    axis.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    figures["Inference time pie chart"] = finish(fig, "Relative Inference Time")

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.boxplot([frame["confidence"], frame["processing_time_seconds"]], tick_labels=["Confidence", "Time (s)"])
    figures["Confidence/time box plot"] = finish(fig, "Result Distribution")
    return figures
