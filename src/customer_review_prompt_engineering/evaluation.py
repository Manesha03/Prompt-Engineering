from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


RUBRIC = {
    "relevance": "Does the response directly address the selected customer-review task?",
    "completeness": "Does it cover the requested dimensions without omitting important information?",
    "consistency": "Is the reasoning internally consistent and supported by the review text?",
    "instruction_following": "Does it follow format, label, length, and constraint instructions?",
    "output_quality": "Is the response clear, useful, concise, and presentation-ready?",
}


@dataclass
class RubricScore:
    relevance: int
    completeness: int
    consistency: int
    instruction_following: int
    output_quality: int
    notes: str = ""

    @property
    def total(self) -> int:
        return self.relevance + self.completeness + self.consistency + self.instruction_following + self.output_quality


def classification_metrics(reference: list[str], predicted: list[str]) -> dict:
    labels = ["Negative", "Neutral", "Positive"]
    cm = confusion_matrix(reference, predicted, labels=labels)
    return {
        "accuracy": round(float(accuracy_score(reference, predicted)), 4),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(reference, predicted, labels=labels, zero_division=0, output_dict=True),
    }


def empty_rubric_results(task: str, prompt_version: str) -> pd.DataFrame:
    rows = []
    for criterion, description in RUBRIC.items():
        rows.append(
            {
                "task": task,
                "prompt_version": prompt_version,
                "criterion": criterion,
                "description": description,
                "score_1_to_5": None,
                "evidence_or_notes": "",
            }
        )
    return pd.DataFrame(rows)

