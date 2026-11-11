"""Единая схема данных бенчмарка (Pydantic v2).

Task  — одна задача (строка в data/tasks/*.jsonl)
RunRecord — результат одного вызова модели (строка в runs/*.jsonl)

JSON Schema генерируется: python scripts/export_schema.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

DOMAIN_PREFIX = {"mechanics": "MECH", "electrical": "ELEC", "thermofluids": "THERM"}


class Answer(BaseModel):
    """Одна искомая величина. Значение — в основных единицах СИ без приставок."""
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", examples=["M_max"])
    description: str = Field(examples=["максимальный изгибающий момент"])
    value: float
    unit: str = Field(examples=["N*m"])
    tol: float = Field(0.01, gt=0, description="Допуск: 0.01 = 1 %. Если эталон равен 0 — абсолютный допуск")


class Task(BaseModel):
    id: str = Field(pattern=r"^(MECH|ELEC|THERM)-[A-Z]{2,6}-\d{4}$", examples=["MECH-BEAM-0001"])
    domain: Literal["mechanics", "electrical", "thermofluids"]
    subtopic: str
    difficulty: int = Field(ge=1, le=3, description="1 — одна формула, 2 — несколько шагов, 3 — составная модель")
    statement: str = Field(min_length=20, description="Полное условие, все данные в тексте")
    answers: list[Answer] = Field(min_length=1)
    solution: str = Field(description="Эталонное решение")
    checks: dict[str, str] = Field(
        default_factory=dict,
        description="Необязательно. Имя ответа -> формула с числами, напр. \"10000*4**2/8\" (доступны sqrt, log, sin...)",
    )
    source: str = Field("", description="Откуда задача, напр. \"Тарг, №12.3, числа изменены\"")
    author: str
    verified: bool = False

    @model_validator(mode="after")
    def _consistency(self) -> "Task":
        if not self.id.startswith(DOMAIN_PREFIX[self.domain] + "-"):
            raise ValueError(f"id {self.id} не соответствует домену {self.domain}")
        names = [a.name for a in self.answers]
        if len(names) != len(set(names)):
            raise ValueError("имена в answers должны быть уникальны")
        if set(self.checks) - set(names):
            raise ValueError(f"checks ссылаются на несуществующие ответы: {set(self.checks) - set(names)}")
        return self


# ---------- Результаты прогонов (заполняет код, руками не пишется) ----------

class ModelAnswer(BaseModel):
    name: str
    value: float
    unit: str = ""


class AnswerCheck(BaseModel):
    name: str
    expected: float
    got: Optional[float] = None
    unit_got: str = ""
    ok: bool
    hint: Optional[str] = Field(None, description="Авто-подсказка разметчику: SIGN / DIM / MISSING")


class RunRecord(BaseModel):
    run_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task_id: str
    provider: str
    model: str
    prompt_variant: str
    prompt_hash: str
    settings: dict = Field(default_factory=dict)
    response_text: str = ""
    stop_reason: Optional[str] = None
    parsed: Optional[list[ModelAnswer]] = None
    parse_error: Optional[str] = None
    checks: list[AnswerCheck] = Field(default_factory=list)
    correct: Optional[bool] = None
    usage: dict = Field(default_factory=dict)
    latency_s: Optional[float] = None
    attempts: int = 1
    error: Optional[str] = None
    # заполняется вручную при разметке (docs/error_taxonomy.md)
    error_code: Optional[str] = None
    phys_flag: Optional[bool] = None
    annotator: Optional[str] = None
