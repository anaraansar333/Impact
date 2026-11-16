"""Единый клиент для OpenAI, Anthropic и open-source моделей через Together / Groq.

Обработка лимитов:
  1) клиентский троттлинг — не чаще RPM запросов в минуту на провайдера;
  2) повтор при 429 / 408 / 409 / 5xx / сетевых ошибках с экспоненциальной паузой + jitter,
     с учётом заголовка retry-after, если сервер его прислал.
Встроенные ретраи SDK отключены (max_retries=0), чтобы каждый повтор логировался нами.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger("bench.llm")

OPENAI_COMPATIBLE = {
    "openai": (None, "OPENAI_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}
PROVIDERS = (*OPENAI_COMPATIBLE, "anthropic")
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass
class LLMResult:
    text: str
    stop_reason: str | None
    usage: dict
    latency_s: float
    attempts: int
    response_id: str | None = None
    settings: dict = field(default_factory=dict)


class RateLimiter:
    """Простой интервальный троттлинг: не чаще rpm запросов в минуту (потокобезопасно)."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


class LLMClient:
    def __init__(self, provider: str, model: str, *, rpm: float | None = None,
                 max_retries: int | None = None, max_backoff_s: float | None = None,
                 timeout_s: float = 600.0):
        if provider not in PROVIDERS:
            raise ValueError(f"провайдер {provider!r} не поддерживается: {PROVIDERS}")
        self.provider, self.model = provider, model
        self.limiter = RateLimiter(rpm if rpm is not None else float(os.getenv("DEFAULT_RPM", 30)))
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("MAX_RETRIES", 6))
        self.max_backoff_s = max_backoff_s if max_backoff_s is not None else float(os.getenv("MAX_BACKOFF_S", 90))

        if provider == "anthropic":
            import anthropic
            self._sdk = anthropic
            self._client = anthropic.Anthropic(max_retries=0, timeout=timeout_s)
        else:
            import openai
            base_url, key_env = OPENAI_COMPATIBLE[provider]
            key = os.getenv(key_env)
            if not key:
                raise RuntimeError(f"не задан {key_env} (см. .env.example)")
            self._sdk = openai
            self._client = openai.OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=timeout_s)

    # ---------- публичный метод ----------
    def complete(self, system: str, user: str, *, max_tokens: int = 16000,
                 temperature: float | None = None, effort: str | None = None) -> LLMResult:
        """effort — только для Anthropic (low/medium/high/xhigh/max).
        temperature — только для OpenAI-совместимых; у новых моделей Claude параметры сэмплинга не принимаются."""
        attempt = 0
        while True:
            attempt += 1
            self.limiter.wait()
            t0 = time.monotonic()
            try:
                if self.provider == "anthropic":
                    res = self._call_anthropic(system, user, max_tokens, effort)
                else:
                    res = self._call_openai(system, user, max_tokens, temperature)
                res.latency_s = round(time.monotonic() - t0, 3)
                res.attempts = attempt
                return res
            except Exception as e:  # noqa: BLE001
                retry, wait_hint = self._classify(e)
                if not retry or attempt > self.max_retries:
                    raise
                delay = wait_hint if wait_hint is not None else min(self.max_backoff_s, 2 ** attempt)
                delay = min(self.max_backoff_s, delay) * random.uniform(1.0, 1.25)
                log.warning("%s/%s: %s — повтор %d/%d через %.1f с",
                            self.provider, self.model, type(e).__name__, attempt, self.max_retries, delay)
                time.sleep(delay)

    # ---------- провайдеры ----------
    def _call_anthropic(self, system, user, max_tokens, effort) -> LLMResult:
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        if effort:
            kwargs["output_config"] = {"effort": effort}
        # Серверные fallbacks на другую модель намеренно НЕ включаем: в бенчмарке
        # ответ должен принадлежать именно запрошенной модели. Отказ фиксируем как stop_reason=refusal.
        r = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        usage = {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
        return LLMResult(text=text, stop_reason=r.stop_reason, usage=usage, latency_s=0, attempts=0,
                         response_id=r.id, settings={"max_tokens": max_tokens, "effort": effort})

    def _call_openai(self, system, user, max_tokens, temperature) -> LLMResult:
        kwargs = dict(model=self.model,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        # у OpenAI новые модели принимают max_completion_tokens; Together/Groq — max_tokens
        kwargs["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        r = self._client.chat.completions.create(**kwargs)
        ch = r.choices[0]
        u = r.usage
        usage = {"input_tokens": getattr(u, "prompt_tokens", None), "output_tokens": getattr(u, "completion_tokens", None)}
        return LLMResult(text=ch.message.content or "", stop_reason=ch.finish_reason, usage=usage, latency_s=0,
                         attempts=0, response_id=r.id, settings={"max_tokens": max_tokens, "temperature": temperature})

    # ---------- ошибки ----------
    def _classify(self, e: Exception) -> tuple[bool, float | None]:
        """(повторять ли, пауза из retry-after или None)."""
        sdk = self._sdk
        if isinstance(e, (sdk.APIConnectionError, sdk.APITimeoutError)):
            return True, None
        if isinstance(e, sdk.APIStatusError):
            if e.status_code not in RETRYABLE_STATUS:
                return False, None
            ra = e.response.headers.get("retry-after") if e.response is not None else None
            try:
                return True, float(ra) if ra else None
            except ValueError:
                return True, None
        return False, None
