# review_response — 심사 피드백 재측정 작업

SDN 라우팅 논문(OSPF vs QoS-DQN vs Risk-DQN)의 심사 지적사항을 **다시 측정해서**
확인하거나 반증한 기록이다. 원고 숫자를 그대로 믿지 않고 코드에서 다시 뽑는 것이
목적이고, 그 과정에서 원고의 결함과 내 자신의 잘못된 가설을 여러 건 찾아 적어 뒀다.

## 1. 가장 먼저 — 폴더 배치

**이 폴더는 단독으로 동작하지 않는다.** 스크립트 140개 중 **68개**가 다음 줄로
원본 코드를 import 한다:

```python
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))
```

따라서 원본 `SDN_DQN_new_env/` 의 **형제 폴더**로 두어야 한다:

```
ROUTING TEST/
├── SDN_DQN_new_env/        <- 원본 (이 저장소에 없음, 별도로 받아야 함)
├── Risk-aware-DQN-routing/ <- 원본 (별도 git 저장소)
└── review_response/        <- 이 저장소
```

## 2. 절대 규칙 — 원본은 읽기만 했다

모든 작업은 `review_response/` 안에서만 하고, 산출물은 `review_results/` 로만
나간다. 원본 `SDN_DQN_new_env/`, `Risk-aware-DQN-routing/` 는 **수정 금지**다.

확인 방법:

```bash
cd ../Risk-aware-DQN-routing && git status --porcelain   # 비어 있어야 정상
```

`SDN_DQN_new_env/` 는 git 저장소가 아니라 수정 시각으로 확인한다. 작업 중
시각이 바뀌는 것은 `__pycache__/*.pyc` 뿐이고(우리 스크립트가 원본 모듈을
import 할 때 파이썬이 자동 생성), 소스 파일은 바뀌지 않는다.

## 3. 폴더 구조

| 경로 | 내용 |
|---|---|
| `HANDOVER.md` | **처음부터 어디까지 했고 결과가 무엇인지.** 처음 보면 여기부터. |
| `FINDINGS_*.md` | 주제별 측정 기록 **10개**. 각 항목에 **무엇을 쟀고 무엇을 안 쟀는지** 명시. |
| `RESUME_HERE.md` | 진행 상황과 남은 할 일의 **정본**. 다른 문서와 어긋나면 이게 맞다. |
| `DRAFT_*.md` | 수정 원고 초안 조각. |
| `measure_*.py`, `reeval_*.py`, `diagnose_*.py` | 측정 스크립트. |
| `verify_table_cells.py`, `audit_draft_numbers.py` | 초안 숫자 검증 도구 (아래 순서 주의). |
| `review_results/` | 모든 측정 산출 CSV·로그. |
| `results/`, `models*/` | 재측정에 쓴 입력 데이터와 체크포인트. |

## 4. 검증 도구는 순서가 있다

초안 숫자 검증은 **반드시 이 순서**로 돌린다:

```
초안 편집  ->  verify_table_cells.py  ->  audit_draft_numbers.py
```

커버리지 색인이 `(초안 파일, 줄번호)` 를 키로 쓰기 때문에 거꾸로 돌리면 멀쩡한
칸이 거짓 경보로 뜬다.

목표 상태는 **248칸 불일치 0 · 살아 있는 주장 0건 · 못 찾은 전사 0개**다.

## 5. 실행 환경

Windows + Python 3.11. torch·numpy·networkx·pandas 필요(scipy 는 쓰지 않는다 —
통계는 `review_stats.py` 에 직접 구현).

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONIOENCODING=utf-8 \
    python -u -X utf8 measure_<name>.py
```

- `OMP_NUM_THREADS=1` 을 빼면 결과가 달라진다. torch 스레드 수가 경로 설정
  시간 측정을 최대 4.4배 왜곡하고, 지연 측정값을 26.13 ms 움직인다.
- `KMP_DUPLICATE_LIB_OK=TRUE` 없으면 OpenMP 중복 로드로 죽는다.
- `python -I` (isolated) 는 pandas 를 깨뜨린다.

## 6. 이 저장소에 없는 것

- `SDN_DQN_new_env/`, `Risk-aware-DQN-routing/` — 원본. 별도로 받아야 한다.
- `results/dynamic_link_log.csv` (61MB) — GitHub 파일 경고선(50MB) 초과로 제외.
  `analyze_risk_identification.py` 와 `run_policy.py` 가 읽으므로 그 둘을 돌릴
  때는 따로 받아야 한다.

## 7. 읽는 순서 추천

1. **`HANDOVER.md`** — 전체 진행과 결과 요약, 함정 모음, 이어받는 순서.
2. `RESUME_HERE.md` 헤더 — 지금까지 닫은 항목과 남은 공저자 결정.
3. 관심 주제의 `FINDINGS_*.md`.
4. 거기 적힌 산출 CSV 를 `review_results/` 에서 대조.

각 FINDINGS 문서는 결론만 적지 않고 **기각된 가설도 같이** 적어 뒀다. "내가 이렇게
예상했는데 측정해 보니 아니었다" 는 단락이 있으면 그게 본문이다.
