from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import load_settings
from .data import build_review_block, dataset_insights, load_reviews
from .evaluation import classification_metrics
from .llm import LLMClient, extract_json_object, normalize_label
from .prompts import get_prompt


def variables_for_task(task: str, reviews: pd.DataFrame | str, insights: dict | None = None) -> dict:
    if isinstance(reviews, pd.DataFrame):
        review_text = build_review_block(reviews)
    else:
        review_text = reviews
    return {
        "reviews": review_text,
        "review": review_text,
        "dataset_insights": json.dumps(insights or {}, indent=2),
    }


def run_prompt(task: str, version: str, reviews: pd.DataFrame | str, client: LLMClient, insights: dict | None = None) -> str:
    variables = variables_for_task(task, reviews, insights)
    prompt = get_prompt(task, version, variables)
    temperature = 0.1 if task == "classification" else 0.3
    return client.run(prompt, temperature=temperature)


def evaluate_classification_sample(data_path: Path, version: str = "v3", sample_size: int = 25) -> dict:
    settings = load_settings()
    df = load_reviews(data_path, sample_size=sample_size)
    client = LLMClient(settings["api_key"], settings["model"])
    predictions = []

    for _, row in df.iterrows():
        response = run_prompt("classification", version, str(row["review"]), client)
        parsed = extract_json_object(response)
        label = parsed.get("label", response) if parsed else response
        predictions.append(normalize_label(str(label)))

    metrics = classification_metrics(df["reference_label"].tolist(), predictions)
    metrics["sample_size"] = len(df)
    metrics["prompt_version"] = version
    return metrics


def load_default_dataset(sample_size: int | None = 1000) -> pd.DataFrame:
    settings = load_settings()
    return load_reviews(settings["data_path"], sample_size=sample_size)

