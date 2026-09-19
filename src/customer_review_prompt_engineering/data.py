from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"Score", "Summary", "Text"}


def rating_to_label(score: int) -> str:
    if score <= 2:
        return "Negative"
    if score == 3:
        return "Neutral"
    return "Positive"


def load_reviews(path: Path, sample_size: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["Score", "Text"]).copy()
    df["Score"] = df["Score"].astype(int)
    df["Summary"] = df["Summary"].fillna("")
    df["Text"] = df["Text"].astype(str)
    df["review"] = (df["Summary"].str.strip() + ". " + df["Text"].str.strip()).str.strip(". ")
    df["reference_label"] = df["Score"].apply(rating_to_label)

    if "Time" in df.columns:
        df["review_date"] = pd.to_datetime(df["Time"], unit="s", errors="coerce")

    cols = [c for c in ["Id", "ProductId", "Score", "reference_label", "Summary", "Text", "review", "review_date"] if c in df.columns]
    df = df[cols]
    if sample_size:
        df = df.head(sample_size)
    return df


def save_processed_reviews(raw_path: Path, processed_path: Path, sample_size: int | None = None) -> pd.DataFrame:
    df = load_reviews(raw_path, sample_size=sample_size)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    return df


def build_review_block(rows: pd.DataFrame, max_reviews: int = 8, max_chars: int = 700) -> str:
    lines = []
    for i, row in rows.head(max_reviews).iterrows():
        text = str(row["review"])[:max_chars]
        lines.append(f"Review {len(lines) + 1} | Rating: {row['Score']} | Text: {text}")
    return "\n".join(lines)


def dataset_insights(df: pd.DataFrame) -> dict:
    score_counts = df["Score"].value_counts().sort_index().to_dict()
    label_counts = df["reference_label"].value_counts().to_dict()
    avg_score = float(df["Score"].mean())
    return {
        "review_count": int(len(df)),
        "average_score": round(avg_score, 3),
        "score_distribution": score_counts,
        "label_distribution": label_counts,
    }
