"""선택 실습 B — 문장 임베딩과 코사인 유사도 확인.

실행:  uv run python embedding_test.py

흐름: 문장 -> 임베딩 벡터 -> 코사인 유사도 -> 의미 유사도 확인

주의:
- 여기서 쓰는 임베딩은 '문장 비교용'이며, 생성 모델(qwen3/gemma3)이 답을 만들 때
  사용하는 내부 토큰 표현과는 다른 것이다.
- 같은 임베딩 모델과 같은 설정으로 얻은 벡터끼리만 비교한다.
- 유사도 값은 '의미가 가깝다'는 뜻이지 '답이 사실이다'는 뜻이 아니다.
"""

from sentence_transformers import SentenceTransformer, util

# 한국어를 지원하는 다국어 임베딩 모델
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

print("임베딩 모델을 불러오는 중입니다. 처음 실행하면 다운로드가 필요합니다.")
model = SentenceTransformer(MODEL_NAME)
print("모델: " + MODEL_NAME)
print("")


def show_similarity(title, base, candidates):
    """기준 문장 하나와 후보 문장들의 유사도를 출력한다."""
    print("=" * 60)
    print(title)
    print("-" * 60)
    print("기준 문장: " + base)
    print("")

    base_vec = model.encode(base)

    results = []
    for text in candidates:
        vec = model.encode(text)
        score = util.cos_sim(base_vec, vec).item()
        results.append((score, text))

    # 유사도가 높은 순서로 정렬
    results.sort(reverse=True)

    for score, text in results:
        print("  " + str(round(score, 4)) + "  " + text)
    print("")


# ── 1부. 발제 예시 문장 ────────────────────────────────────
sentence_a = "고양이가 소파 위에 있다."
sentence_b = "소파 위에 고양이가 앉아 있다."
sentence_c = "오늘 주식시장이 상승했다."

print("임베딩 벡터의 차원 수: " + str(len(model.encode(sentence_a))))
print("")

show_similarity(
    "1부. 발제 예시 문장",
    sentence_a,
    [sentence_b, sentence_c],
)

# ── 2부. 프로젝트 자료에 적용 ──────────────────────────────
# 광고 사례 하나를 기준으로, 판정 기준 네 조항 중 어느 것이 의미상 가까운지 본다.
ad_case = "광고에는 남향이라고 적혀 있으나 실제 주된 채광 방향은 북향이다."

criteria = [
    "(가) 광고 시 제시한 옵션의 성능을 실제와 다르게 표시한 경우",
    "(나) 광고한 관리비 총 금액이 광고 시점 직전 월 관리비와 현저하게 차이가 나는 경우",
    "(다) 방향이 광고 시 제시한 방향과 90도 이상 차이가 나는 경우",
    "(라) 주요 교통시설과의 거리를 실제 도보거리가 아니라 직선거리로 표시·광고하는 경우",
]

show_similarity(
    "2부. 광고 사례와 판정 기준의 유사도\n(정답 조항: (다) 방향 90도 이상 차이)",
    ad_case,
    criteria,
)

# 다른 사례로 한 번 더
ad_case2 = "지하철역까지 도보 5분이라고 광고했으나 직선거리 기준이었고 실제로는 13분 걸린다."

show_similarity(
    "3부. 다른 광고 사례\n(정답 조항: (라) 직선거리 표시)",
    ad_case2,
    criteria,
)

print("=" * 60)
print("해석 시 주의")
print("-" * 60)
print("유사도가 높다는 것은 문장의 의미가 가깝다는 뜻이며,")
print("해당 조항이 실제로 적용된다는 판단과는 다르다.")
print("이 값을 판정의 정확도나 사실성 점수로 해석하지 않는다.")
