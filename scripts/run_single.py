"""Одиночный вызов модели на одной задаче.

Примеры:
  python scripts/run_single.py --task MECH-BEAM-0001 --prompt guarded --dry-run
  python scripts/run_single.py --task ELEC-DC-0001 --provider anthropic --model claude-opus-5-5 --prompt cot --effort medium
  python scripts/run_single.py --task THERM-GAS-0001 --provider openai --model gpt-5 --prompt zero_shot
  python scripts/run_single.py --task THERM-GAS-0001 --provider groq --model llama-3.3-70b-versatile --temperature 0
Результат дописывается в $LOG_DIR/runs.jsonl (по умолчанию runs/runs.jsonl).
"""
import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from bench.answer import compare, parse_final_answer  # noqa: E402
from bench.io import load_tasks  # noqa: E402
from bench.llm import PROVIDERS, LLMClient  # noqa: E402
from bench.prompts import VARIANTS, build_prompt  # noqa: E402
from bench.schema import RunRecord  # noqa: E402


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="id задачи")
    ap.add_argument("--tasks-file", type=Path, help="конкретный .jsonl (по умолчанию все в data/tasks)")
    ap.add_argument("--provider", choices=PROVIDERS, default="anthropic")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--prompt", choices=VARIANTS, default="zero_shot")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--temperature", type=float, help="только OpenAI-совместимые провайдеры")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], help="только Anthropic")
    ap.add_argument("--dry-run", action="store_true", help="показать промпт и не вызывать API")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    task = next((t for t in load_tasks(args.tasks_file) if t.id == args.task), None)
    if task is None:
        print(f"задача {args.task} не найдена", file=sys.stderr)
        return 2
    system, user, phash = build_prompt(task, args.prompt)

    if args.dry_run:
        print(f"=== SYSTEM ({args.prompt}, hash {phash}) ===\n{system}\n\n=== USER ===\n{user}")
        return 0

    rec = RunRecord(run_id=uuid.uuid4().hex[:12], task_id=task.id, provider=args.provider, model=args.model,
                    prompt_variant=args.prompt, prompt_hash=phash)
    try:
        res = LLMClient(args.provider, args.model).complete(
            system, user, max_tokens=args.max_tokens, temperature=args.temperature, effort=args.effort)
        rec.response_text, rec.stop_reason, rec.usage = res.text, res.stop_reason, res.usage
        rec.latency_s, rec.attempts, rec.settings = res.latency_s, res.attempts, res.settings
        try:
            rec.parsed = parse_final_answer(res.text)
            rec.correct, rec.checks = compare(task, rec.parsed)
        except Exception as e:  # noqa: BLE001
            rec.parse_error, rec.correct = str(e), False
    except Exception as e:  # noqa: BLE001
        rec.error = f"{type(e).__name__}: {e}"
        logging.error("вызов не удался: %s", rec.error)

    log_dir = ROOT / os.getenv("LOG_DIR", "runs")
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(rec.model_dump_json() + "\n")

    print(json.dumps({"task": task.id, "correct": rec.correct, "stop_reason": rec.stop_reason,
                      "checks": [c.model_dump() for c in rec.checks], "parse_error": rec.parse_error,
                      "error": rec.error, "usage": rec.usage, "latency_s": rec.latency_s},
                     ensure_ascii=False, indent=2))
    return 0 if rec.error is None else 1


if __name__ == "__main__":
    sys.exit(main())
