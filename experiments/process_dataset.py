from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_review_prompt_engineering.config import DEFAULT_DATA_PATH, ROOT_DIR
from customer_review_prompt_engineering.data import save_processed_reviews


DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "reviews_cleaned.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean the raw reviews dataset and save a processed CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Path to raw Reviews.csv.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path for the processed CSV.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional number of rows to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = save_processed_reviews(args.input, args.output, sample_size=args.sample_size)
    print(f"Saved {len(df)} cleaned reviews to {args.output}")


if __name__ == "__main__":
    main()
