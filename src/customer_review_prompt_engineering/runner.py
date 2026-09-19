from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import load_settings
from .data import build_review_block, dataset_insights, load_reviews
from .evaluation import classification_metrics
from .llm import extract_json_object, normalize_label
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


def run_prompt(task: str, version: str, reviews: pd.DataFrame | str, client, insights: dict | None = None) -> str:
    variables = variables_for_task(task, reviews, insights)
    prompt = get_prompt(task, version, variables)
    temperature = 0.1 if task == "classification" else 0.3
    return client.run(prompt, temperature=temperature)


def local_prompt_response(task: str, version: str, reviews: pd.DataFrame | str, insights: dict | None = None) -> str:
    if isinstance(reviews, pd.DataFrame):
        df = reviews.copy()
        review_text = build_review_block(df)
    else:
        df = pd.DataFrame()
        review_text = reviews.strip() or "No review text provided."

    if task == "classification":
        if not df.empty and "Score" in df.columns:
            score = int(df.iloc[0]["Score"])
            label = "Negative" if score <= 2 else "Neutral" if score == 3 else "Positive"
            if version == "v1":
                return label
            if version == "v2":
                return f"Label: {label}\nReason: Local mode classified this from the review rating ({score})."
            return json.dumps(
                {
                    "label": label,
                    "confidence": "High",
                    "reason": f"Local mode used the review rating ({score}) as the sentiment signal.",
                },
                indent=2,
            )
        return json.dumps(
            {
                "label": "Unknown",
                "confidence": "Low",
                "reason": "Local mode needs a dataset rating to classify reliably.",
            },
            indent=2,
        )

    if task == "data_analysis":
        facts = insights or {}
        if version == "v1":
            return "\n".join(
                [
                    "### Local Data Analysis",
                    f"The dataset contains {facts.get('review_count', 'unknown')} reviews with an average rating of {facts.get('average_score', 'unknown')}.",
                    "This is a simple baseline response, similar to what a basic prompt might produce.",
                ]
            )
        if version == "v2":
            return "\n".join(
                [
                    "### Local Data Analysis",
                    f"- Rating pattern: average rating is {facts.get('average_score', 'unknown')}.",
                    f"- Positive reviews: {facts.get('label_distribution', {}).get('Positive', 0)}",
                    f"- Negative reviews: {facts.get('label_distribution', {}).get('Negative', 0)}",
                    "- Business insight: compare high-rating themes against low-rating complaints before deciding improvements.",
                ]
            )
        return "\n".join(
            [
                "### Local Data Analysis",
                f"- Reviews loaded: {facts.get('review_count', 'Unknown')}",
                f"- Average rating: {facts.get('average_score', 'Unknown')}",
                f"- Score distribution: {facts.get('score_distribution', {})}",
                f"- Label distribution: {facts.get('label_distribution', {})}",
                "- This local mode summarizes available dataset facts without calling an API.",
            ]
        )

    if task == "summarization":
        if version == "v1":
            return "\n".join(
                [
                    "### Local Summary",
                    "Basic summary mode:",
                    "",
                    review_text[:700],
                ]
            )
        if version == "v2":
            return "\n".join(
                [
                    "### Local Summary",
                    "**Key positive points:** Review the selected text for praise and high ratings.",
                    "**Key negative points:** Look for complaints, low ratings, delivery, taste, or packaging issues.",
                    "**Customer concerns:** Check repeated problems across selected reviews.",
                    "**Overall opinion:** Local mode prepares the structure without using API credits.",
                ]
            )
        return "\n".join(
            [
                "### Local Structured Review Synthesis",
                "## Key Positive Points",
                "- Local mode highlights where a V3 prompt would place evidence-grounded positives.",
                "## Key Negative Points",
                "- Local mode highlights where a V3 prompt would separate complaints or risks.",
                "## Customer Concerns",
                "- Use OpenAI API mode later for full natural-language synthesis.",
                "## Overall Opinion",
                "The selected review text is prepared in the stricter V3 output format.",
                "",
                "```text",
                review_text[:1200],
                "```",
            ]
        )

    if task == "content_generation":
        if version == "v1":
            return "\n".join(
                [
                    "### Local Recommendations",
                    "- Improve the product based on customer reviews.",
                    "- Improve customer experience where complaints appear.",
                ]
            )
        if version == "v2":
            return "\n".join(
                [
                    "### Local Recommendations",
                    "- Product improvement ideas: inspect repeated negative review themes.",
                    "- Packaging or delivery improvements: check low-rated complaints for logistics issues.",
                    "- Customer experience recommendations: prioritize fixes that affect many reviews.",
                    "- Risks to monitor: track repeated complaints over time.",
                ]
            )
        return "\n".join(
            [
                "### Local Prioritized Action Plan",
                "| Recommendation | Review Evidence | Expected Impact | Effort | Priority | Confidence |",
                "| --- | --- | --- | --- | --- | --- |",
                "| Review lowest-rated comments first | Rating and label distribution | Finds urgent issues | Low | High | Medium |",
                "| Compare repeated complaints with positive themes | Selected review sample | Avoids overreacting to one review | Medium | Medium | Medium |",
                "",
                "**Top next step:** Use the selected reviews to identify the most repeated complaint.",
                "**What to measure:** Rating changes, complaint frequency, and positive theme frequency.",
            ]
        )

    return "Unsupported task."


def evaluate_classification_sample(data_path: Path, version: str = "v3", sample_size: int = 25, mode: str = "local") -> dict:
    df = load_reviews(data_path, sample_size=sample_size)
    client = None
    if mode == "openai":
        from .llm import LLMClient

        settings = load_settings()
        client = LLMClient(settings["api_key"], settings["model"])
    predictions = []

    for _, row in df.iterrows():
        if mode == "openai":
            if client is None:
                raise RuntimeError("OpenAI client was not initialized.")
            response = run_prompt("classification", version, str(row["review"]), client)
        else:
            response = local_prompt_response("classification", version, pd.DataFrame([row]))
        parsed = extract_json_object(response)
        label = parsed.get("label", response) if parsed else response
        predictions.append(normalize_label(str(label)))

    metrics = classification_metrics(df["reference_label"].tolist(), predictions)
    metrics["sample_size"] = len(df)
    metrics["prompt_version"] = version
    metrics["mode"] = mode
    return metrics


def load_default_dataset(sample_size: int | None = 1000) -> pd.DataFrame:
    settings = load_settings()
    return load_reviews(settings["data_path"], sample_size=sample_size)
