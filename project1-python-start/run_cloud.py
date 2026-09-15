"""Cloud API 비교 실험.

미리 정한 질문 5개(Q01, Q03, Q05, Q06, Q10)를 Cloud 모델에 각 1회 적용한다.
로컬 모델과 같은 system 프롬프트, 같은 자료를 사용한다.

실행:  uv run python run_cloud.py
결과:  results/cloud.jsonl

주의: API 키는 실행할 때 입력받는다. 코드나 저장소에 넣지 않는다.
"""

import getpass
import json
import time
from datetime import datetime

from openai import OpenAI

import config

# 결과를 보기 전에 미리 정한 5개 (notes.md에 선정 이유 기록)
CLOUD_QUESTIONS = ["Q01", "Q03", "Q05", "Q06", "Q10"]

# 사용할 Cloud 모델과 단가 (1M 토큰당 USD). 실제 단가는 실행 시점에 확인할 것.
CLOUD_MODEL = "gpt-5.6-luna"
PRICE_INPUT = 0.20
PRICE_OUTPUT = 1.20

OUT_PATH = "results/cloud.jsonl"


# 질문 파일 읽기
f = open(config.QUESTIONS_PATH, encoding="utf-8")
all_questions = json.load(f)
f.close()

# 미리 정한 5개만 고르기
questions = []
for q in all_questions:
    if q["id"] in CLOUD_QUESTIONS:
        questions.append(q)

print("Cloud 비교 대상 " + str(len(questions)) + "개: " + str(CLOUD_QUESTIONS))

# API 키 입력받기 (화면에 표시되지 않음)
api_key = getpass.getpass("OpenAI API 키를 입력하세요: ")
client = OpenAI(api_key=api_key)

out = open(OUT_PATH, "a", encoding="utf-8")
total_cost = 0.0

for q in questions:
    user_msg = config.build_user_message(q)

    record = {
        "run_id": CLOUD_MODEL + "|" + q["id"] + "|r1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": CLOUD_MODEL,
        "question_id": q["id"],
        "question_type": q["type"],
        "repeat": 1,
        "is_warmup": False,
        "system_prompt": config.SYSTEM_PROMPT,
        "user_message": user_msg,
    }

    # 로컬과 같은 방식: 질문마다 messages를 새로 만든다
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=CLOUD_MODEL,
            messages=messages,
            temperature=1,
            max_completion_tokens=config.OPTIONS["num_predict"],
        )
    except Exception as e:
        # 실패도 기록한다
        record["status"] = "error"
        record["error_type"] = type(e).__name__
        record["error_message"] = str(e)[:500]
        record["elapsed_sec"] = round(time.perf_counter() - start, 3)
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("  " + q["id"] + "  status=error  " + type(e).__name__)
        continue

    elapsed = time.perf_counter() - start

    in_tokens = response.usage.prompt_tokens
    out_tokens = response.usage.completion_tokens
    cost = in_tokens / 1000000 * PRICE_INPUT + out_tokens / 1000000 * PRICE_OUTPUT
    total_cost = total_cost + cost

    record["status"] = "success"
    record["response_text"] = response.choices[0].message.content
    record["elapsed_sec"] = round(elapsed, 3)
    record["prompt_eval_count"] = in_tokens
    record["eval_count"] = out_tokens
    record["cost_usd"] = round(cost, 6)
    record["price_note"] = "단가 입력 $" + str(PRICE_INPUT) + " / 출력 $" + str(PRICE_OUTPUT) + " per 1M tokens"

    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("  " + q["id"] + "  status=success  "
          + "elapsed=" + str(round(elapsed, 3)) + "s  "
          + "in=" + str(in_tokens) + " out=" + str(out_tokens) + "  "
          + "cost=$" + str(round(cost, 6)))

out.close()

print("")
print("추정 총비용: $" + str(round(total_cost, 6)))
print("실제 사용 내역은 OpenAI 대시보드에서 따로 확인할 것")
print("저장 완료 -> " + OUT_PATH)
