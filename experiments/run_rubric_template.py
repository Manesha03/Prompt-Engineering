from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_review_prompt_engineering.evaluation import empty_rubric_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a blank rubric sheet for manual prompt evaluation.")
    parser.add_argument("--task", choices=["data_analysis", "summarization", "content_generation"], required=True)
    parser.add_argument("--version", choices=["v1", "v2", "v3"], required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or Path(f"reports/{args.task}_{args.version}_rubric.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df = empty_rubric_results(args.task, args.version)
    df.to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
