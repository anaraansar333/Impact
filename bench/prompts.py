"""Сборка промптов. Системный промпт — из prompts/<variant>.md,
пользовательское сообщение — условие + список искомых величин + общий блок формата.
Блок формата одинаков для всех вариантов, чтобы сравнение было честным."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .schema import Task

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
VARIANTS = ("zero_shot", "cot", "guarded")


def _read(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def build_prompt(task: Task, variant: str) -> tuple[str, str, str]:
    """Возвращает (system, user, prompt_hash)."""
    if variant not in VARIANTS:
        raise ValueError(f"неизвестный вариант промпта {variant!r}, допустимы {VARIANTS}")
    system = _read(variant)
    wanted = "\n".join(f"- {a.name} — {a.description}" for a in task.answers)
    user = f"{task.statement}\n\nИскомые величины:\n{wanted}\n\n{_read('output_format')}"
    prompt_hash = hashlib.sha256((system + "\n---\n" + user).encode("utf-8")).hexdigest()[:12]
    return system, user, prompt_hash
