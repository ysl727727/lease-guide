"""실험 결과를 집계한다.

실행 방법:
  uv run python aggregate.py            -> 성능 지표
  uv run python aggregate.py quality    -> 품질 점수
  uv run python aggregate.py answers    -> 답변 원문 (전부)
  uv run python aggregate.py answers Q06 -> 답변 원문 (Q06만)
"""

import csv
import json
import sys


# 질문 유형표 (집계할 때 씁니다)
QTYPE = {
    "Q01": "정상", "Q02": "정상", "Q03": "정상", "Q04": "정상", "Q05": "정상",
    "Q06": "경계", "Q07": "경계", "Q08": "경계",
    "Q09": "범위밖", "Q10": "범위밖",
}

# 집계할 성능 지표 (jsonl의 키 이름, 화면에 표시할 이름)
METRICS = [
    ("elapsed_sec", "전체 응답 시간(초)"),
    ("load_duration_sec", "모델 로딩 시간(초)"),
    ("tokens_per_sec", "생성 속도(tok/s)"),
    ("eval_count", "출력 토큰 수"),
    ("prompt_eval_count", "입력 토큰 수"),
    ("size_vram_mib", "VRAM(MiB)"),
]


def load_runs():
    """results/runs.jsonl을 읽어서 리스트로 돌려준다."""
    runs = []
    f = open("results/runs.jsonl", encoding="utf-8")
    for line in f:
        if line.strip() != "":
            runs.append(json.loads(line))
    f.close()
    return runs


def get_models(runs):
    """기록에 등장하는 모델 이름을 순서대로 모은다."""
    models = []
    for r in runs:
        if r["model"] not in models:
            models.append(r["model"])
    return models


def show_performance():
    runs = load_runs()

    for model in get_models(runs):
        # 이 모델의 본 실험 기록만 고른다 (워밍업 제외)
        my_runs = []
        for r in runs:
            if r["model"] == model and r["is_warmup"] == False:
                my_runs.append(r)

        # 성공한 것만 따로 고른다
        success = []
        for r in my_runs:
            if r["status"] == "success":
                success.append(r)

        print("")
        print("=== " + model + " ===")
        print("호출 성공 수 / 전체 시도 수: " + str(len(success)) + " / " + str(len(my_runs)))

        # 지표마다 평균을 낸다
        for key, label in METRICS:
            values = []
            for r in success:
                v = r.get(key)
                # 값이 없으면(None) 평균에 넣지 않는다. 0으로 채우지 않는다.
                if v is not None:
                    values.append(v)

            if len(values) == 0:
                print("  " + label + ": 집계 불가 (유효 측정값 0건)")
            else:
                avg = sum(values) / len(values)
                print("  " + label + ": 평균 " + str(round(avg, 2))
                      + "  (n=" + str(len(values))
                      + ", 최소 " + str(min(values))
                      + " / 최대 " + str(max(values)) + ")")

        # 워밍업은 따로 보여준다
        for r in runs:
            if r["model"] == model and r["is_warmup"] == True:
                print("  [워밍업, 본 집계 제외] 로딩 " + str(r.get("load_duration_sec"))
                      + "s / 응답 " + str(r.get("elapsed_sec")) + "s")

        # 실패한 회차
        fails = []
        for r in my_runs:
            if r["status"] == "error":
                fails.append(r)
        if len(fails) == 0:
            print("  실패 회차: 없음")
        else:
            print("  실패 회차:")
            for r in fails:
                print("    - " + r["question_id"] + " r" + str(r["repeat"]) + ": " + r["error_type"])

        # 실행 조건 (첫 성공 기록 기준)
        if len(success) > 0:
            first = success[0]
            print("  양자화: " + str(first.get("quantization_level")))
            print("  실제 context: " + str(first.get("context_length")))
            print("  생성 설정: " + str(first.get("options")))


def show_quality():
    # 채점표를 읽는다
    rows = []
    f = open("results/scores.csv", encoding="utf-8")
    for row in csv.DictReader(f):
        rows.append(row)
    f.close()

    # 모델 이름 모으기
    models = []
    for row in rows:
        if row["model"] not in models:
            models.append(row["model"])

    # 질문 ID 모으기
    questions = []
    for row in rows:
        if row["question_id"] not in questions:
            questions.append(row["question_id"])
    questions.sort()

    print("")
    print("질문별 달성률 (반복 평균)   채점 건수 " + str(len(rows)))
    print("질문  유형    " + "   ".join(models))

    # 나중에 유형별 평균을 내려고 저장해 둔다
    all_rates = {}

    for q in questions:
        line = q + "  " + QTYPE.get(q, "") + "  "
        for m in models:
            rates = []
            for row in rows:
                if row["question_id"] == q and row["model"] == m:
                    rate = int(row["score"]) / int(row["max_score"]) * 100
                    rates.append(rate)
            if len(rates) > 0:
                avg = sum(rates) / len(rates)
                all_rates[(q, m)] = avg
                line = line + "  " + str(round(avg, 1)) + "%"
        print(line)

    print("")
    print("유형별 평균 달성률")
    for t in ["정상", "경계", "범위밖"]:
        line = "  " + t + ": "
        for m in models:
            rates = []
            for q in questions:
                if QTYPE.get(q) == t and (q, m) in all_rates:
                    rates.append(all_rates[(q, m)])
            if len(rates) > 0:
                avg = sum(rates) / len(rates)
                line = line + m + " " + str(round(avg, 1)) + "%   "
        print(line)

    print("")
    print("전체 평균 달성률 (질문 10개 동일 가중)")
    for m in models:
        rates = []
        for q in questions:
            if (q, m) in all_rates:
                rates.append(all_rates[(q, m)])
        avg = sum(rates) / len(rates)
        print("  " + m + ": " + str(round(avg, 1)) + "%")


def show_answers(only):
    runs = load_runs()
    for r in runs:
        if r["is_warmup"] == True:
            continue
        if only is not None and r["question_id"] != only:
            continue
        print("")
        print("[" + r["question_id"] + " / " + str(r["repeat"]) + "회] " + r["model"])
        print("출력토큰 " + str(r.get("eval_count")) + "  응답시간 " + str(r.get("elapsed_sec")) + "s")
        print("-" * 50)
        print(r.get("response_text", "(응답 없음)"))
        print("=" * 50)


# 여기서부터 실행 시작
if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    mode = "performance"

if mode == "quality":
    show_quality()
elif mode == "answers":
    if len(sys.argv) > 2:
        show_answers(sys.argv[2])
    else:
        show_answers(None)
else:
    show_performance()
