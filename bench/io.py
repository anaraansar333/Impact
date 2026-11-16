from __future__ import annotations

from pathlib import Path

from .schema import Task

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "data" / "tasks"


def load_tasks(path: Path | None = None) -> list[Task]:
    """Читает один .jsonl или все .jsonl из data/tasks. Ошибки валидации — с номером строки."""
    files = [path] if path else sorted(TASK_DIR.glob("*.jsonl"))
    tasks, errors = [], []
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                tasks.append(Task.model_validate_json(line))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{f.name}:{i}: {e}")
    if errors:
        raise ValueError("ошибки в задачах:\n" + "\n".join(errors))
    return tasks
