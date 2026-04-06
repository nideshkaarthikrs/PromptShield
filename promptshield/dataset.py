"""Dataset loader for PromptShield tasks."""

import json
import os

_TASK_MAP = {
    "task_easy": "easy.json",
    "task_medium": "medium.json",
    "task_hard": "hard.json",
}

_TASKS_DIR = os.path.join(os.path.dirname(__file__), "tasks")


def load_dataset(task_name: str) -> list:
    if task_name not in _TASK_MAP:
        raise ValueError(
            f"Unknown task '{task_name}'. Valid options: {list(_TASK_MAP.keys())}"
        )
    path = os.path.join(_TASKS_DIR, _TASK_MAP[task_name])
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
