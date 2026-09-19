from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_review_prompt_engineering.config import DEFAULT_DATA_PATH
from customer_review_prompt_engineering.runner import evaluate_classification_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate review sentiment prompt versions with rating-derived labels.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--version", choices=["v1", "v2", "v3"], default="v3")
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--mode", choices=["local", "openai"], default="local")
    parser.add_argument("--output", type=Path, default=Path("reports/classification_metrics.json"))
    args = parser.parse_args()

    metrics = evaluate_classification_sample(args.data_path, args.version, args.sample_size, mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
