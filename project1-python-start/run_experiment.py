"""로컬 모델 2개 x 질문 10개 x 2회 = 본 실험 40회를 실행하고 전부 기록한다.

실행:  uv run python run_experiment.py
결과:  results/runs.jsonl  (한 줄 = 한 번의 호출)

한 건만 시험해 볼 때:  uv run python run_experiment.py --only Q01
"""

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

from ollama import Client

import config

BASE = Path(__file__).parent
NS = 1_000_000_000  # 나노초 -> 초
MIB = 1_048_576     # 바이트 -> MiB


def get_ps_entry(client: Client, model_tag: str) -> dict:
    """답변 직후, 모델이 아직 메모리에 있을 때 실행 상태를 조회한다.

    size_vram은 '이 시점의 값'이지 최대 VRAM 사용량이 아니다. 관측 시점을 함께 남긴다.
    """
    info = {"vram_observed_at": datetime.now().isoformat(timespec="seconds")}
    try:
        listed = client.ps().models
    except Exception as exc:
        info["ps_error"] = f"{type(exc).__name__}: {exc}"
        return info

    if not listed:
        info["ps_note"] = "실행 중인 모델 목록이 비어 있음 (이미 메모리에서 내려갔을 수 있음)"
        return info

    for m in listed:
        if getattr(m, "model", None) == model_tag:
            size_vram = getattr(m, "size_vram", None)
            details = getattr(m, "details", None)
            info.update({
                "size_vram_bytes": size_vram,
                "size_vram_mib": round(size_vram / MIB, 1) if size_vram else None,
                "digest": getattr(m, "digest", None),
                "quantization_level": getattr(details, "quantization_level", None),
                "parameter_size": getattr(details, "parameter_size", None),
                "context_length": getattr(m, "context_length", None),
            })
            return info

    info["ps_note"] = f"목록에 {model_tag} 없음 (다른 모델이 올라와 있는지 확인)"
    return info


def ask_once(client: Client, model_tag: str, question: dict,
             repeat: int, is_warmup: bool) -> dict:
    """한 번 호출하고, 성공/실패와 상관없이 기록 한 줄을 만들어 돌려준다."""
    user_msg = config.build_user_message(question)

    record = {
        "run_id": f"{model_tag}|{question['id']}|r{repeat}" + ("|warmup" if is_warmup else ""),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_tag,
        "question_id": question["id"],
        "question_type": question["type"],
        "question_format": question.get("format"),
        "repeat": repeat,
        "is_warmup": is_warmup,
        "options": config.OPTIONS,
        "system_prompt": config.SYSTEM_PROMPT,
        "user_message": user_msg,
    }

    # 질문마다 messages를 새로 만든다. 이전 질문의 대화 이력을 이어 붙이지 않는다.
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    start = time.perf_counter()
    try:
        response = client.chat(
            model=model_tag,
            messages=messages,
            options=config.OPTIONS,
        )
    except Exception as exc:
        # 실패도 기록에 남긴다. 지우거나 성공 응답으로 대체하지 않는다.
        record.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
            "elapsed_sec": round(time.perf_counter() - start, 3),
            "traceback": traceback.format_exc()[-1000:],
        })
        return record

    elapsed = time.perf_counter() - start

    load_ns = getattr(response, "load_duration", None)
    eval_count = getattr(response, "eval_count", None)
    eval_ns = getattr(response, "eval_duration", None)
    prompt_eval_count = getattr(response, "prompt_eval_count", None)

    # 생성 속도: eval_duration이 없거나 0 이하면 계산하지 않고 사유를 남긴다.
    if eval_count and eval_ns and eval_ns > 0:
        tokens_per_sec = round(eval_count / (eval_ns / NS), 2)
        tps_note = None
    else:
        tokens_per_sec = None
        tps_note = "eval_count 또는 eval_duration 통계 없음/0 이하"

    record.update({
        "status": "success",
        "response_text": response.message.content,
        "elapsed_sec": round(elapsed, 3),
        "load_duration_sec": round(load_ns / NS, 3) if load_ns else None,
        "eval_count": eval_count,
        "eval_duration_sec": round(eval_ns / NS, 3) if eval_ns else None,
        "tokens_per_sec": tokens_per_sec,
        "tokens_per_sec_note": tps_note,
        "prompt_eval_count": prompt_eval_count,
    })
    record.update(get_ps_entry(client, model_tag))
    return record


def append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="특정 질문 ID만 실행 (예: Q01)")
    parser.add_argument("--out", default=None, help="결과 파일 경로 덮어쓰기")
    args = parser.parse_args()

    questions = json.loads((BASE / config.QUESTIONS_PATH).read_text(encoding="utf-8"))
    if args.only:
        questions = [q for q in questions if q["id"] == args.only]
        if not questions:
            raise SystemExit(f"{args.only} 질문을 찾을 수 없습니다.")

    out_path = BASE / (args.out or config.RESULTS_PATH)
    out_path.parent.mkdir(exist_ok=True)

    client = Client()

    for model_tag in config.MODELS:
        print(f"\n=== {model_tag} ===")

        # 워밍업: 첫 호출의 로딩 지연을 본 실험과 섞지 않기 위해 먼저 돌린다.
        for _ in range(config.WARMUP_PER_MODEL):
            rec = ask_once(client, model_tag, questions[0], repeat=0, is_warmup=True)
            append(out_path, rec)
            print(f"  warmup   status={rec['status']} "
                  f"elapsed={rec.get('elapsed_sec')}s load={rec.get('load_duration_sec')}s")

        # 본 실험: 질문 10개 x 2회
        for q in questions:
            for r in range(1, config.REPEATS + 1):
                rec = ask_once(client, model_tag, q, repeat=r, is_warmup=False)
                append(out_path, rec)
                print(f"  {q['id']} r{r}  status={rec['status']} "
                      f"elapsed={rec.get('elapsed_sec')}s "
                      f"in={rec.get('prompt_eval_count')} out={rec.get('eval_count')} "
                      f"tps={rec.get('tokens_per_sec')} "
                      f"vram={rec.get('size_vram_mib')}MiB")

    print(f"\n저장 완료 -> {out_path}")


if __name__ == "__main__":
    main()
