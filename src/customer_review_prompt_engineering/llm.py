from __future__ import annotations

import json
import re

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None


class LLMClient:
    def __init__(self, api_key: str, model: str):
        if OpenAI is None:
            raise ModuleNotFoundError("The openai package is not installed. Run `pip install -r requirements.txt` or use Local free mode.")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Create a .env file from .env.example.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def run(self, prompt: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


def extract_json_object(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_label(label: str) -> str:
    lowered = label.strip().lower()
    if "negative" in lowered:
        return "Negative"
    if "neutral" in lowered or "mixed" in lowered:
        return "Neutral"
    if "positive" in lowered:
        return "Positive"
    return "Unknown"
