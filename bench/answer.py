"""Разбор FINAL_ANSWER, сравнение с эталоном, проверка формул валидации."""
from __future__ import annotations

import json
import math
import re

from .schema import AnswerCheck, ModelAnswer, Task

_MARKER = re.compile(r"FINAL_ANSWER\s*:\s*", re.IGNORECASE)
_MATH_NS = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}


def _to_float(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("−", "-").replace(" ", "").replace(",", ".")
    s = re.sub(r"[·×x\*]10\^?\(?([+-]?\d+)\)?", r"e\1", s)  # 1.5·10^3 -> 1.5e3
    return float(s)


def parse_final_answer(text: str) -> list[ModelAnswer]:
    """Берёт последний FINAL_ANSWER в тексте. Бросает ValueError, если не удалось."""
    matches = list(_MARKER.finditer(text or ""))
    if not matches:
        raise ValueError("нет строки FINAL_ANSWER")
    start = text.find("{", matches[-1].end())
    if start < 0:
        raise ValueError("после FINAL_ANSWER нет JSON")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    items = obj.get("answers", []) if isinstance(obj, dict) else obj
    out = [ModelAnswer(name=str(i["name"]), value=_to_float(i["value"]), unit=str(i.get("unit", ""))) for i in items]
    if not out:
        raise ValueError("пустой список answers")
    return out


def _close(got: float, exp: float, tol: float) -> bool:
    return abs(got - exp) <= (tol * abs(exp) if exp != 0 else tol)


def _hint(got: float, exp: float, tol: float) -> str | None:
    if exp != 0 and _close(-got, exp, tol):
        return "SIGN"
    if got != 0 and exp != 0:
        ratio = math.log10(abs(got / exp))
        n = round(ratio)
        if n != 0 and abs(ratio - n) < 0.01:
            return "DIM"
    return None


def compare(task: Task, parsed: list[ModelAnswer]) -> tuple[bool, list[AnswerCheck]]:
    by_name = {p.name.strip().lower(): p for p in parsed}
    checks = []
    for a in task.answers:
        p = by_name.get(a.name.lower())
        if p is None:
            checks.append(AnswerCheck(name=a.name, expected=a.value, ok=False, hint="MISSING"))
            continue
        ok = _close(p.value, a.value, a.tol)
        checks.append(AnswerCheck(
            name=a.name, expected=a.value, got=p.value, unit_got=p.unit, ok=ok,
            hint=None if ok else _hint(p.value, a.value, a.tol),
        ))
    return all(c.ok for c in checks), checks


def run_checks(task: Task) -> list[str]:
    """Пересчитывает эталон по формулам checks. Возвращает список расхождений (пустой — всё ок).
    eval используется только на данных, которые пишет команда, не на ответах моделей."""
    env = dict(_MATH_NS)
    expected = {a.name: a for a in task.answers}
    problems = []
    for name, expr in task.checks.items():
        try:
            val = float(eval(expr, {"__builtins__": {}}, env))  # noqa: S307
        except Exception as e:  # noqa: BLE001
            problems.append(f"{task.id}/{name}: формула не вычисляется ({e})")
            continue
        a = expected[name]
        if not _close(val, a.value, a.tol / 10):
            problems.append(f"{task.id}/{name}: по формуле {val:.6g}, в answers {a.value:.6g}")
    return problems
