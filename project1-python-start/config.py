"""실험 조건을 한곳에 모아둔 파일.

여기 있는 값은 본 실험 시작 전에 확정하고, 40회가 끝날 때까지 바꾸지 않는다.
"""

# ── 비교할 로컬 후보 2개 (ollama list의 전체 태그) ──────────────
MODELS = [
    "gemma3:4b",
    "qwen3:4b-instruct-2507-q4_K_M",
]

# ── 모든 질문·모든 모델에 동일하게 적용되는 지시문 ────────────────
SYSTEM_PROMPT = (
    "당신은 전월세 계약을 처음 하는 임차인을 돕는 안내 어시스턴트입니다.\n"
    "- 제공된 [참고 자료]에 있는 내용만 근거로 답하세요.\n"
    "- 자료에 없는 절차, 금액, 기간, 기관명을 추정하거나 만들어내지 마세요.\n"
    "- 답변은 간결하게, 근거를 먼저 쓰고 결론을 마지막 줄에 쓰세요.\n"
    "- 자료만으로 답할 수 없으면 모른다고 밝히고 어디에 확인하면 되는지 안내하세요.\n"
    "- 계약의 안전성 판단이나 법적 판단은 하지 말고, 안내 범위를 벗어난다고 밝히세요."
)

# ── 생성 설정 ────────────────────────────────────────────────────
OPTIONS = {
    "temperature": 0,
    "num_predict": 512,
    "num_ctx": 4096,
    "seed": 42,
}

REPEATS = 2           # 질문당 본 실험 횟수
WARMUP_PER_MODEL = 1  # 모델당 워밍업 (본 집계에서 제외)

QUESTIONS_PATH = "prompts/questions.json"
RESULTS_PATH = "results/runs.jsonl"


def build_user_message(question: dict) -> str:
    """참고 자료를 프롬프트에 직접 넣는다. 검색 단계가 없으므로 RAG가 아니다."""
    return f"{question['context']}\n\n[질문] {question['question']}"