from __future__ import annotations

from pathlib import Path

import yaml

from .config import PROMPT_PATH


def load_prompts(path: Path = PROMPT_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_prompt(task: str, version: str, variables: dict, path: Path = PROMPT_PATH) -> str:
    prompts = load_prompts(path)
    template = prompts[task][version]["template"]
    return template.format(**variables)


def prompt_metadata(task: str, version: str, path: Path = PROMPT_PATH) -> dict:
    prompts = load_prompts(path)
    return prompts[task][version]

