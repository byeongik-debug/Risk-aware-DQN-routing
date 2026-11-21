# DATA_INDEX — 측정 데이터 어디에 뭐가 있는지

`review_results/` 에 산출물 **245개(69MB)** 가 있다. 색인 없이는 못 찾으므로 주제별로
정리한다. 숫자를 확인하려면 **FINDINGS 문서 → 이 색인 → CSV** 순서로 가면 된다.

- **기계가 읽을 전체 색인: `review_results/_data_index.csv`** (245행). 열은
  `file, bytes, rows, writers, readers, header` 이고 `header` 에 그 CSV 의 전체 열
  이름이 들어 있다. 어느 파일에 어떤 열이 있는지 찾을 때 이걸 grep 하면 된다.
- 주의: `writers` 열은 **스크립트 본문에서 파일명을 grep 해서 추정**한 것이다. 파일명을
  언급만 하는 스크립트가 섞일 수 있다(예: `table_cell_verification.csv` 는 실제로
  `verify_table_cells.py` 가 쓰고 `audit_draft_numbers.py` 는 읽기만 한다). 정확히 알아야
  하면 해당 스크립트의 `OUT_CSV` 를 보면 된다.
- `_*_log.txt` 30개는 스크립트 산출이 아니라 **쉘에서 리다이렉트한 실행 로그**다.
  근거로 쓰지 말고 재현 확인용으로만 본다.

---

## 1. 입력 데이터 (우리가 만든 게 아니라 읽은 것)

| 경로 | 크기 | 쓰는 쪽 |
|---|---|---|
| `results/dynamic_link_log.csv` | **61MB** | `analyze_risk_identification.py`, `run_policy.py` |
| `results/` 나머지 | — | 원본 실험의 런별 결과 CSV |
| `models/`, `models*/` | — | 재측정에 쓴 체크포인트 |
| `_originals/SDN_DQN_new_env/` | 73MB | **스크립트 140개 중 68개가 import 한다.** 환경·토폴로지·위험계산·체크포인트 |
| `_originals/Risk-aware-DQN-routing/` | 23MB | 어느 스크립트도 참조하지 않는 **참고용** 원본 저장소 |

`_originals/` 는 읽기 전용 사본이다. 배치 방법은 `README.md` 1절.

---

## 2. 초안 숫자 검증 (가장 먼저 돌려 볼 것)

| 파일 | 행 | 내용 |
|---|---|---|
| `draft_number_provenance.csv` | 1349 | 초안 숫자 **전건** 출처 판정(AGG_CSV / DOC_ONLY / NO_SOURCE …) |
| `table_cell_verification.csv` | 266 | 표 칸 **재집계** 결과. 인쇄값 대 계산값 대 차이 |

목표 상태는 **248칸 불일치 0 · 살아 있는 주장 0건 · 전사 못 찾은 것 0개**이고 현재
그 상태다. 재실행 순서는 `verify_table_cells.py` → `audit_draft_numbers.py` (역순 금지).

---

## 3. 표 10 (가중치/인자 절제) · 평가 규약

| 파일 | 행 | 내용 |
|---|---|---|
| `table10_f1.csv` | 12 | 표 10 F1 열 |
| `_table10_rawlog.pkl` | 22.5MB | 표 10 원시 롤아웃. `build_table10_f1.py`, `measure_ic_redundancy.py` 가 읽는다 |
| `table10_welch.csv` | 65 | 팔별 Welch 검정 |
| `table10_factor_runs.csv` / `table10_extra_runs.csv` | 35 / 34 | 인자 절제 런별 결과 |
| `table10_fill.csv` / `table10_sigma_ref.csv` | 15 / 24 | 표 10 빈칸 채움과 산포 기준 |
| `outcome_membership.csv` | 69 | **결과값 붕괴** — 모수 69개가 결과값 40개로 뭉치는 소속 |
| `eval_seed_offset.csv` | 3 | 계통 +1.249% 의 정체(규약 아님, 표본 하나의 위치) |
| `july_nov_protocol.csv` | 25 | 7월(비쌍) 대 11월(공통난수) 규약 비교 |
| `eval_protocol_noise.csv` / `..._from_log.csv` | 90 / 90 | 평가 규약 산포 직접 측정 |
| `policy_classes.csv` | 148 | 다섯 라벨이 실제로는 정책 셋이라는 군집 결과 |
| `cluster_n_reanalysis.csv` | 57 | 족 크기 재분석 |

**읽을 때 주의**: "±" 는 성능 산포가 아니다(`outcome_membership.csv` 를 보면 최빈 하나에
23개가 몰려 있다). 지연 절대값 368.8 / 348.7 / 360.8 은 **서로 다른 평가 규약**의 값이다.

---

## 4. 손실 지표 · 채점 규약 · ospf_bw 메커니즘

| 파일 | 행 | 내용 |
|---|---|---|
| `scoring_convention.csv` | 220 | 홉당 평균 대 경로 총량 대 유실확률, 네 정책 × 네 시나리오 |
| `convention_gap_mechanism.csv` | 80 | 규약 차이의 **원인** 분해(천장·혼잡·압축계수·물리 드리프트) |
| `convention_gap_links.csv` | 37 | 링크 단위 분해 |
| `norisk_conventions.csv` / `..._summary.csv` | 30 / 93 | 표 11 손실 열 종단간 재계산(부호 역전) |
| `pareto_e2e.csv` | 20 | 종단간 기준 지연–손실 교환관계 |
| **`bw_saturation.csv`** | **80** | `ospf_bw` 포화 가설 **기각** 근거(상한 도달 몫·비용값 종수·비용비) |
| **`bw_hop_margin.csv`** | **12** | 홉수별 최소 대역폭비용과 7홉/6홉 여유 8.38~16.67% |
| `ospf_replan.csv` | 100 | 재계획 대 1회 계획 (차이는 재계획이 아니라 `step` 때문) |
| `qos_baseline_handicap.csv` | 140 | QoS 비교군에게 같은 정보를 줬을 때 |
| `risk_dijkstra.csv` / `risk_dijkstra_paths.csv` | 60 / 20 | 위험 Dijkstra 대조군 |

---

## 5. 경로 설정 시간 (표 12)

| 파일 | 행 | 내용 |
|---|---|---|
| `table12_path_setup.csv` | 12 | 표 12 본표 |
| `path_setup_time_threads1.csv` / `..._threads12.csv` | 24 / 24 | **스레드 수가 4.4배 왜곡**함을 보이는 쌍 |
| `decide_segments_runs.csv` (+`_rep2`,`_rep3`) | 833 / 829 / 826 | 결정 구간 분해, 복제 3회 |
| `decide_segments_derived.csv` | 95 | 분해 비중(순전파 52.8% · 마스킹 12.9% · 텐서 4.2% · 미귀속 27.9%) |
| `decision_breakdown_runs.csv` | 2493 | 결정당 비용 런별 |
| `decision_breakdown_summary.csv` | 149 | 위 요약 |
| `decision_breakdown_noparts_*.csv` | 2499 / 138 | 부분 제거 대조 |
| `decision_breakdown_fits.csv` | 3 | 절편 적합(절편 0.51 ms 철회 근거) |

**비중을 인용할 때는 규약을 병기**해야 한다 — 같은 비용이 고립 측정 17.1% / 호출 내
26.99% 로 갈린다(`decide_segments_derived.csv`).

---

## 6. Risk-Blind 고정 경로 · 적응 팔 (학습 몫 2.25%)

| 파일 | 행 | 내용 |
|---|---|---|
| `blind_path_candidates.csv` | **246** | 홉≤8 후보 경로 전수와 기대보상. 6홉은 4개뿐 |
| `blind_path_rivals.csv` | 200 | 고정 경로 대 경쟁 경로 |
| `blind_path_paired.csv` | 980 | 쌍체 + Holm(족 245) |
| `blind_path_decompose.csv` | 6 | 경로 몫 대 **갈아타기 몫** 분해 |
| `policy_paths_all.csv` | 13800 | 팔 69개 × 에피소드별 실제 경로 |
| `arm_path_switch_gain.csv` | 69 | 팔별 갈아타기 이득(+30.45 등) |
| `path_identity_all_arms.csv` / `..._pairs.csv` | 69 / 2346 | 경로 수열 **40종뿐 · 23개가 팔 9개에 걸쳐 동일** |
| `path_identity_modal_arms.csv` / `..._pairs.csv` | 25 / 300 | 최빈 경로 동일성 |
| `cross_paths_arms.csv` / `..._pairs.csv` | 20 / 190 | 교차 통제군(다른 에피소드 관측) |
| `path_diversity.csv` | 90 | 정책별 사용 경로 수(Risk-DQN 1~3개) |
| `reward_scale.csv` | 2099 | 보상 규모 |

**"13/13 유의" 를 팔 간 독립 증거로 쓰면 안 된다** — `path_identity_all_arms.csv` 가
경로 수열 중복을 보여 준다.

---

## 7. 상태 가림 · 위험 블록 여분 · 피해 축

| 파일 | 행 | 내용 |
|---|---|---|
| `state_occlusion_battery.csv` | 120 | 블록 가림 배터리(risk_mean / risk_frozen / dest_mean …) |
| `state_occlusion_rollout.csv` | 70 | 가림별 롤아웃 |
| `occlusion_paths_runs.csv` / `..._detail.csv` | 15 / 26 | 가림 시 실제 경로(통제군 3.2개 대 가린 쪽 1.0개) |
| `episode_metrics_all.csv` | 22000 | 에피소드 단위 전수(섞기 문턱 분석용) |
| `episode_metrics_thr.csv` / `_ref` / `_shuf` / `_shuf20` | 24000 / 14000 / 8000 / 2000 | 문턱·기준·섞기 변형 |
| `decision_flip.csv` / `decision_flip_thr.csv` | 117 / 42 | 위험 칸을 섞었을 때 **결정 전환율** |
| `axis_factorial.csv` / `rank_vs_value_axis.csv` | 15 / 40 | 피해 축이 코사인(방향) 하나임 |
| `rank_vs_value.csv` / `rank_value_geometry.csv` | 60 / 13 | 순위 교란 대 값 교란 |
| `policy_path_distance_arms.csv` / `..._pairs.csv` | 12 / 2346 | 정책 간 경로 거리 |
| `policy_distance.csv` | 110 | 정책 거리 요약 |

**문턱 위치 주장은 철회됐다** — 격자를 바꾸면 갈리는 자리가 흩어진다. 단조 곡선에
계단이 없다.

---

## 8. 시간 창(window) · 시나리오

| 파일 | 행 | 내용 |
|---|---|---|
| `window_policy_test.csv` | 15 | 창 길이별 정책 검정(Holm 0/16) |
| `window_policy_distance.csv` / `window_policy_loo.csv` | 17 / 9 | 창별 거리 · 하나씩 빼기 |
| `window_sweep_significance.csv` / `..._corrected.csv` | 21 / 11 | 스윕 유의성 · 다중비교 보정 |

**고아 파일 주의**: `window_sweep_summary.csv` 와 `results/qos_reference.csv` 의
`dynamic` 342.28 은 어느 분석에도 연결되지 않는다. "고치지" 말 것.

---

## 9. 토폴로지 일반화 (R3-4, 표 14)

| 파일 | 행 | 내용 |
|---|---|---|
| `topology_generalization_runs.csv` | 90 | 재배선 비율별 런 |
| `generalization_norevisit.csv` | 90 | **재방문 배제 디코딩**(60% 실패 → 100%) |
| `generalization_conventions.csv` | 90 | 규약별 재계산(12/12 유의) |
| `table14_final.csv` / `table14_repro_check.csv` | 6 / 4 | 표 14 본표와 재현 확인 |
| `table14_final_runs.csv` | 150 | 런별 |
| `table14_ospf_hops_runs.csv` | 120 | OSPF 홉 비교 |

**체크포인트 주의**: 표 14 는 `770d17d9`, 표 10/11/12 는 `20e74c39` 다. 같은
`risk_only_model_final.pth` 가 cwd 에 따라 다른 모델로 해석된다 → `check_model_paths.py`.

---

## 10. 위험 링크 식별 · 예측 (R3-2)

| 파일 | 행 | 내용 |
|---|---|---|
| `risk_oracle_leak.csv` | 120 | **`link_type` 이 정답과 1:1 누출**(누출항만 0.9743 > 전체 0.9662) |
| `risk_prediction.csv` | 264 | 예측 AUC. 세 시나리오는 과제가 없어 1.000 |

---

## 11. 노드 이동성 (C-1)

| 파일 | 행 | 내용 |
|---|---|---|
| `mobility_proxy.csv` / `_prev` / `_none` | 420 / 420 / 420 | 이동성 대리지표 세 변형 |
| `rwp_churn.csv` / `rwp_embedding_fit.csv` / `rwp_sensitivity.csv` | 25 / 29 / 14 | RWP 타당성 |
| `cross_2x2_tests.csv` / `cross_baselines.csv` / `cross_runs.csv` | 18 / 10 / 10 | 2×2 절제(결손 +0.00%p) |

---

## 12. 없는 것 / 미완

- **링크 재사용 집중도·경로 간 중복도** — `measure_link_reuse.py` 가 작성돼 있고 스모크
  테스트는 통과했으나 **본 실행 산출물이 없다**(중단). 돌리는 법과 검증 기준은
  `HANDOVER.md` 6.2절.
- 재학습이 필요한 "안 쟀다" 칸들 — `FINDINGS_table10_f1.md` 에서
  `grep -n "안 쟀다\|단정하지 않는다"`.
