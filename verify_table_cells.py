# -*- coding: utf-8 -*-
"""초안 표 교체본의 **칸마다** 런 단위 CSV 에서 다시 집계해 대조한다.

왜 이게 따로 필요한가
--------------------
`audit_draft_numbers.py` 는 "이 숫자가 커밋 파일 어딘가에 있나" 만 본다.
그런데 표 칸은 대개 **런 CSV 의 평균**이라 어느 파일에도 그 문자열로
존재하지 않는다(가림 배터리 471.09 ± 0.00 은 `occlusion_paths_runs.csv`
다섯 줄의 평균이다). 그래서 값 대조만으로는
  - 값은 맞지만 **다른 표**에서 온 칸(`350.1 ± 1.5` 사건)
  - 집계 규칙이 다른 칸(± 가 모델 간인지 시드 간인지)
을 못 잡는다. 여기서는 행 식별과 열 매핑을 **명시 선언**하고 평균·표준편차를
다시 계산해 인쇄값과 맞춘다. 허용 오차는 인쇄 자릿수의 반(360.8 이면 0.05)이다.

원본 트리는 **읽기만** 한다. 표 11 의 제안·NoRisk 행은 선언된 출처가
`SDN_DQN_new_env/results/{baseline,norisk}_reference.csv` 이므로
(`table11_learning_models.csv` 가 그렇게 적어 뒀다) 거기까지 내려가 읽는다.
쓰기는 전부 `review_results/` 안에서만 일어난다. 출력:
  review_results/table_cell_verification.csv
"""
import csv
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "review_results")
ORIG = os.path.join(os.path.dirname(HERE), "SDN_DQN_new_env", "results")
OUT = os.path.join(RES, "table_cell_verification.csv")

NUMRE = re.compile(r"[-+±]?\s*\d+(?:\.\d+)?")

# ---------------------------------------------------------------- 선언
# rows : (초안 행을 식별하는 선행 칸 값들, 런 CSV 행 필터)
# cells: (초안 칸 번호 0=첫 칸, CSV 열, 배율, 통계[, 그 칸의 std 열])
#        통계 mean / mean_sd / value(단일행 그대로) / value_sd
#        다섯째가 없으면 spec 의 sd_col 을 쓴다(표 12 처럼 한 열인 경우).
#        range(min~max 범위 칸) / paren_mean(괄호 안 평균)도 쓴다 — 표 13.
# derived: CSV 에 없는 유도 열 {이름: row->float}.
# runs_dir: 런 CSV 가 review_results/ 밖에 있을 때만 적는다(읽기 전용).
SPECS = [
    dict(name=u"표 10 — 인자 절제 (DRAFT_4_6_2)",
         draft="DRAFT_4_6_2_table10.md",
         runs="table10_factor_runs.csv",
         rows=[([u"**Proposed (Full)**"], {"arm": "Full"}),
               ([u"w/o VD"], {"arm": "-dDelay"}),
               ([u"w/o VL"], {"arm": "-dLoss"}),
               ([u"w/o VD, VL"], {"arm": "-VD,VL"}),
               ([u"w/o U"], {"arm": "-Util"}),
               ([u"w/o IC"], {"arm": "-Cong"}),
               ([u"w/o T"], {"arm": "-Type"})],
         cells=[(2, "delay", 1.0, "mean_sd"),
                (3, "loss", 100.0, "mean"),
                (4, "risky_pct", 1.0, "mean"),
                (5, "hop", 1.0, "mean")]),

    # 주의: 이 두 행의 출처는 norisk_conventions.csv(손실 규약 재평가본)가
    # 아니다. 그걸 쓰면 ± 의 층이 달라져(모델 5개 x 시드 5개 중첩) 인쇄값과
    # 안 맞는다 — 2026-11-16 에 내가 그렇게 선언해 거짓 불일치 4건을 만들었다.
    # 선언된 출처는 table11_learning_models.csv 가 적어 둔 원본 결과 파일이고,
    # ± 는 (가) 동결 모델 1개 x 평가 시드 5개 / (나) 독립 학습 모델 5개 다.
    dict(name=u"표 11 — 제안 행 (DRAFT_4_6_3)",
         draft="DRAFT_4_6_3_table11.md",
         runs_dir=ORIG,
         runs="baseline_reference.csv",
         rows=[([u"제안 기법"], {"scenario": "dynamic"})],
         cells=[(1, "delay", 1.0, "value_sd", "delay_std"),
                (2, "loss", 100.0, "value_sd", "loss_std"),
                (3, "hop", 1.0, "value")]),

    dict(name=u"표 11 — NoRisk 행 (DRAFT_4_6_3)",
         draft="DRAFT_4_6_3_table11.md",
         runs_dir=ORIG,
         runs="norisk_reference.csv",
         rows=[([u"NoRisk-DQN <sup>주3</sup>"], {"scenario": "dynamic"})],
         cells=[(1, "delay", 1.0, "value_sd", "delay_std"),
                (2, "loss", 100.0, "value_sd", "loss_std"),
                (3, "hop", 1.0, "value")]),

    dict(name=u"표 11 — QoS-Model 행 (DRAFT_4_6_3)",
         draft="DRAFT_4_6_3_table11.md",
         runs="qos_baseline_handicap.csv",
         rows=[([u"QoS-Model <sup>주4</sup>"],
                {"scenario": "dynamic", "variant": "qos_t0_stale"})],
         cells=[(1, "delay", 1.0, "mean_sd"),
                (2, "published", 100.0, "mean_sd"),
                (3, "hop", 1.0, "mean_sd")]),

    # 이 표의 인쇄값은 감사에서 DOC_ONLY 로 나온다(평균이라 어느 CSV 에도
    # 그 문자열이 없다). 그러나 칸 재집계로는 아래에서 이미 덮인다 —
    # DOC_ONLY 는 "검증 안 됨" 이 아니라 "문자열 대조로는 못 닿음" 이다.
    dict(name=u"표 11 — 위험블록 가림 배터리 (DRAFT_4_6_3)",
         draft="DRAFT_4_6_3_table11.md",
         runs="occlusion_paths_runs.csv",
         rows=[([u"가리지 않음 (통제)"], {"blind": "none"}),
               ([u"전 칸 같은 평균 (정보 0)"], {"blind": "scalar"}),
               ([u"엣지별 평균 (정적 서열 보존)"], {"blind": "edgemean"})],
         cells=[(1, "delay", 1.0, "mean_sd"),
                (2, "loss", 1.0, "mean"),
                (3, "hop", 1.0, "mean"),
                (4, "risky_pct", 1.0, "mean"),
                (5, "n_distinct_paths", 1.0, "mean")]),

    dict(name=u"표 14 — OSPF 행 (DRAFT_4_9)",
         draft="DRAFT_4_9_table14.md",
         runs="table14_final_runs.csv",
         rows=[([u"0.2", u"OSPF"], {"rewire_ratio": "0.2", "method": "OSPF"}),
               ([u"0.4", u"OSPF"], {"rewire_ratio": "0.4", "method": "OSPF"}),
               ([u"0.6", u"OSPF"], {"rewire_ratio": "0.6", "method": "OSPF"})],
         cells=[(2, "success_rate", 1.0, "mean"),
                (3, "delay", 1.0, "mean_sd"),
                (4, "hop", 1.0, "mean_sd"),
                (5, "loss_published", 100.0, "mean_sd"),
                (6, "risky_link_usage", 1.0, "mean_sd"),
                (7, "risky_count", 1.0, "mean")]),

    # CSV 에 "제안" 이라는 method 라벨은 없다 — frozen(20e74c39)/merged(770d17d9)
    # 두 체크포인트 사본이 각각 original/no_revisit 로 들어 있다. 초안 표가
    # 어느 사본에서 왔는지가 그 자체로 미결(메모리: 표 14 는 770d17d9,
    # 초안 캡션은 20e74c39)이므로 둘 다 걸어 인쇄값이 가리는 쪽을 본다.
    dict(name=u"표 14 — 제안 행 = frozen/no_revisit (20e74c39)",
         draft="DRAFT_4_9_table14.md",
         runs="table14_final_runs.csv",
         rows=[([u"0.2", u"**제안**"],
                {"rewire_ratio": "0.2",
                 "method": "Risk-DQN[frozen/no_revisit]"}),
               ([u"0.4", u"**제안**"],
                {"rewire_ratio": "0.4",
                 "method": "Risk-DQN[frozen/no_revisit]"}),
               ([u"0.6", u"**제안**"],
                {"rewire_ratio": "0.6",
                 "method": "Risk-DQN[frozen/no_revisit]"})],
         cells=[(2, "success_rate", 1.0, "mean"),
                (3, "delay", 1.0, "mean_sd"),
                (4, "hop", 1.0, "mean_sd"),
                (5, "loss_published", 100.0, "mean_sd"),
                (6, "risky_link_usage", 1.0, "mean_sd"),
                (7, "risky_count", 1.0, "mean")]),

    # 아래는 **대립 가설 배제용**이고 초안의 주장이 아니다. 인쇄된 원고
    # 표 14 는 merged + *원* 디코딩에서 왔고(FINDINGS_table10_f1 19절),
    # 교체 초안은 본문 통일을 위해 frozen + 재방문배제로 다시 쟀다. 즉
    # merged/no_revisit 조합은 누구의 주장도 아니다 — 인쇄값이 체크포인트에
    # 둔감한 게 아님을 보이려고(판별력) 같이 돌린다. 불일치로 세지 않는다.
    dict(name=u"[대립 가설] 표 14 제안 행이 merged/no_revisit 라면",
         counter=True,
         draft="DRAFT_4_9_table14.md",
         runs="table14_final_runs.csv",
         rows=[([u"0.2", u"**제안**"],
                {"rewire_ratio": "0.2",
                 "method": "Risk-DQN[merged/no_revisit]"}),
               ([u"0.4", u"**제안**"],
                {"rewire_ratio": "0.4",
                 "method": "Risk-DQN[merged/no_revisit]"}),
               ([u"0.6", u"**제안**"],
                {"rewire_ratio": "0.6",
                 "method": "Risk-DQN[merged/no_revisit]"})],
         cells=[(2, "success_rate", 1.0, "mean"),
                (3, "delay", 1.0, "mean_sd"),
                (4, "hop", 1.0, "mean_sd"),
                (5, "loss_published", 100.0, "mean_sd"),
                (6, "risky_link_usage", 1.0, "mean_sd"),
                (7, "risky_count", 1.0, "mean")]),

    # 표 13 은 평균±sd 가 아니다 — **좌표 임베딩 5종의 min~max 범위**이고
    # 첫 칸의 괄호는 유지율 평균이다. "상태가 못 보는 간선 비중" 은 CSV 열이
    # 아니라 novel_edges / n_present 로 만든 유도 열이다.
    dict(name=u"표 13 — 이동성 관측 한계 (DRAFT_4_8)",
         draft="DRAFT_4_8_mobility.md",
         runs="rwp_churn.csv",
         derived={"unseen_pct":
                  lambda r: (float(r["novel_edges"])
                             / float(r["n_present"]) * 100.0)},
         rows=[([u"20"], {"vmax_kmh": "20.0"}),
               ([u"40"], {"vmax_kmh": "40.0"}),
               ([u"60"], {"vmax_kmh": "60.0"}),
               ([u"80"], {"vmax_kmh": "80.0"}),
               ([u"100"], {"vmax_kmh": "100.0"})],
         cells=[(1, "kept_of_50", 1.0, "range"),
                (1, "kept_pct", 1.0, "paren_mean"),
                (2, "novel_edges", 1.0, "range"),
                (3, "unseen_pct", 1.0, "range"),
                (4, "deg0_nodes", 1.0, "range"),
                (5, "sd_connected_pct", 1.0, "range")]),

    # 표 12 는 2026-11-20 까지 세 명세 모두 table12_path_setup.csv 를 봤다. 그 파일은
    # build_table12.py 의 산출물이므로, 그 스크립트가 넣은 결함은 자기 산출물에
    # 대조되어 깨끗하게 통과했다 — 실제로 통과했다(Dynamic 결정 횟수 한 칸이
    # 12스레드 실행에서 왔다). 그래서 **측정값 칸은 측정 산출물에 직접** 댄다.
    # 계산값(배수)만 빌드 산출물에 남기고, 그 입력 두 칸은 아래 두 명세가 이미
    # 독립으로 검증한다.
    dict(name=u"표 12 — 경로 설정 시간, 안 B (DRAFT_4_7)",
         draft="DRAFT_4_7_table12.md",
         runs="path_setup_time_threads1.csv",
         rows=[([u"Normal"], {"scenario": "normal", "method": "OSPF"}),
               ([u"Congestion"], {"scenario": "congestion", "method": "OSPF"}),
               ([u"Unstable"], {"scenario": "unstable", "method": "OSPF"}),
               ([u"Dynamic"], {"scenario": "dynamic", "method": "OSPF"})],
         cells=[(1, "per_path_ms_mean", 1.0, "value_sd")],
         sd_col="per_path_ms_std"),

    dict(name=u"표 12 — 제안 열, 안 B (DRAFT_4_7)",
         draft="DRAFT_4_7_table12.md",
         runs="path_setup_time_threads1.csv",
         rows=[([u"Normal"], {"scenario": "normal", "method": "Risk-DQN"}),
               ([u"Congestion"],
                {"scenario": "congestion", "method": "Risk-DQN"}),
               ([u"Unstable"], {"scenario": "unstable", "method": "Risk-DQN"}),
               ([u"Dynamic"], {"scenario": "dynamic", "method": "Risk-DQN"})],
         cells=[(3, "per_path_ms_mean", 1.0, "value_sd"),
                (4, "decisions_mean", 1.0, "value")],
         sd_col="per_path_ms_std",
         header_hint=u"| Case | OSPF (Dijkstra) | QoS-Model"),

    dict(name=u"표 12 — QoS 열, 안 B (DRAFT_4_7)",
         draft="DRAFT_4_7_table12.md",
         runs="path_setup_time_threads1.csv",
         rows=[([u"Normal"], {"scenario": "normal", "method": "QoS-Dijkstra"}),
               ([u"Congestion"],
                {"scenario": "congestion", "method": "QoS-Dijkstra"}),
               ([u"Unstable"],
                {"scenario": "unstable", "method": "QoS-Dijkstra"}),
               ([u"Dynamic"],
                {"scenario": "dynamic", "method": "QoS-Dijkstra"})],
         cells=[(2, "per_path_ms_mean", 1.0, "value_sd")],
         sd_col="per_path_ms_std",
         header_hint=u"| Case | OSPF (Dijkstra) | QoS-Model"),

    # 배수는 측정값이 아니라 위 두 명세가 검증한 두 칸의 몫이다. 따라서 여기만
    # 빌드 산출물을 본다 — 분자·분모가 독립 검증되므로 세탁 경로가 아니다.
    dict(name=u"표 12 — OSPF 대비 배수(계산값) (DRAFT_4_7)",
         draft="DRAFT_4_7_table12.md",
         runs="table12_path_setup.csv",
         rows=[([u"Normal"], {"scenario": "normal", "method": "Risk-DQN"}),
               ([u"Congestion"],
                {"scenario": "congestion", "method": "Risk-DQN"}),
               ([u"Unstable"], {"scenario": "unstable", "method": "Risk-DQN"}),
               ([u"Dynamic"], {"scenario": "dynamic", "method": "Risk-DQN"})],
         cells=[(5, "ratio_vs_ospf_t1", 1.0, "value")],
         header_hint=u"| Case | OSPF (Dijkstra) | QoS-Model"),
]


# ------------------------------------------------- 유도 칸(비율·차이) 선언
# SCALARS: 이름 -> (디렉터리, CSV, 행 필터, 열, 배율, 집계[value|mean])
# DERIVED: 초안의 유도 칸. 위치 지정 둘 —
#   (ㄱ) 표 행   : rows=[선행 칸] + idx
#   (ㄴ) 본문 산문: anchor(그 줄의 식별 문구) + after(그 뒤부터 센다)
#        + exprs(등장 순서대로의 기대 수식)
SCALARS = {
    "prop_delay": (None, "baseline_reference.csv",
                   {"scenario": "dynamic"}, "delay", 1.0, "value"),
    "prop_loss": (None, "baseline_reference.csv",
                  {"scenario": "dynamic"}, "loss", 100.0, "value"),
    "prop_hop": (None, "baseline_reference.csv",
                 {"scenario": "dynamic"}, "hop", 1.0, "value"),
    "nr_delay": (None, "norisk_reference.csv",
                 {"scenario": "dynamic"}, "delay", 1.0, "value"),
    "nr_loss": (None, "norisk_reference.csv",
                {"scenario": "dynamic"}, "loss", 100.0, "value"),
    "nr_hop": (None, "norisk_reference.csv",
               {"scenario": "dynamic"}, "hop", 1.0, "value"),
    "qos_delay": (RES, "qos_baseline_handicap.csv",
                  {"scenario": "dynamic", "variant": "qos_t0_stale"},
                  "delay", 1.0, "mean"),
    "qos_loss": (RES, "qos_baseline_handicap.csv",
                 {"scenario": "dynamic", "variant": "qos_t0_stale"},
                 "published", 100.0, "mean"),
    "qos_hop": (RES, "qos_baseline_handicap.csv",
                {"scenario": "dynamic", "variant": "qos_t0_stale"},
                "hop", 1.0, "mean"),
}
for _rr in ("0.2", "0.4", "0.6"):
    _k = _rr.replace(".", "")
    SCALARS["ospf_lc" + _k] = (
        RES, "table14_final_runs.csv",
        {"rewire_ratio": _rr, "method": "OSPF"},
        "loss_compounded", 1.0, "mean")
    SCALARS["prop_lc" + _k] = (
        RES, "table14_final_runs.csv",
        {"rewire_ratio": _rr, "method": "Risk-DQN[frozen/no_revisit]"},
        "loss_compounded", 1.0, "mean")

# 경로 일치율 — 한 칸에 숫자가 여러 개라 seq 모드로 센다.
# 칸 넷: qos_both(NoRisk) / qos_reward(상태위험·보상QoS) /
#        qos_state(상태QoS·보상위험) / risk_both(Full)
_CELLS4 = ("qos_both", "qos_reward", "qos_state", "risk_both")
for _c in _CELLS4:
    SCALARS["ag_in_" + _c] = (
        RES, "cross_paths_pairs.csv",
        {"same_cell": "1", "a_cell": _c}, "agree_pct", 1.0, "mean")
for _i in range(4):
    for _j in range(_i + 1, 4):
        SCALARS["ag_%s__%s" % (_CELLS4[_i], _CELLS4[_j])] = (
            RES, "cross_paths_pairs.csv",
            {"same_cell": "0", "a_cell": _CELLS4[_i], "b_cell": _CELLS4[_j]},
            "agree_pct", 1.0, "mean")

# ---- 발표본 PDF 인쇄 상수 (extract_published_pdf.py 가 좌표로 뽑은 산출물)
# 인쇄물의 내용이라 측정 CSV 어디에도 없다. 그래서 PDF 에서 직접 뽑아
# 산출물로 만들었다 — 내가 문서에 "이렇게 적혀 있다" 고 쓰는 것은 출처가
# 아니기 때문이다(32.2.3 세탁 경로).
for _n in ("t5_norisk_delay", "t5_norisk_loss", "t5_prop_delay",
           "t5_prop_loss", "t6_prop_delay", "t6_qos_delay",
           "body_t5_delay_pct", "body_t5_loss_pct"):
    SCALARS["pdf_" + _n] = (RES, "published_pdf_constants.csv",
                            {"name": _n}, "value", 1.0, "value")

# ---- 원본 트리의 NoRisk 5런 (읽기만 한다. d=None 이면 ORIG 로 간다)
for _i in range(1, 6):
    SCALARS["nr_run%02d" % _i] = (
        None, "norisk_dynamic_run%02d_result.csv" % _i, {}, "delay",
        1.0, "value")

# ---- QoS 핸디캡 배터리: 시나리오 x 변종 (지연 · 홉당 손실)
for _sc in ("congestion", "unstable", "dynamic"):
    for _v in ("risk_dqn", "qos_t0_stale", "qos_post_warmup", "qos_replan",
               "qos_dqn"):
        _k = "%s_%s" % (_sc[:4], _v)
        SCALARS["qd_" + _k] = (RES, "qos_baseline_handicap.csv",
                               {"scenario": _sc, "variant": _v},
                               "delay", 1.0, "mean")
        SCALARS["qp_" + _k] = (RES, "qos_baseline_handicap.csv",
                               {"scenario": _sc, "variant": _v},
                               "published", 1.0, "mean")

# ---- 표 10: 스레드 혼입(1스레드 재학습 대 기록 체제)과 평가시드 치우침
SCALARS["u1_delay"] = (RES, "util_thread1_runs.csv",
                       {"arm": "-Util(1thread)"}, "delay", 1.0, "mean")
SCALARS["u1_r03"] = (RES, "util_thread1_runs.csv",
                     {"arm": "-Util(1thread)", "run": "3"}, "delay",
                     1.0, "value")
SCALARS["u12_delay"] = (RES, "table10_factor_runs.csv",
                        {"arm": "-Util"}, "delay", 1.0, "mean")
SCALARS["u12_r03"] = (RES, "table10_factor_runs.csv",
                      {"arm": "-Util", "run": "3"}, "delay", 1.0, "value")
SCALARS["full_delay"] = (RES, "table10_factor_runs.csv",
                         {"arm": "Full"}, "delay", 1.0, "mean")
SCALARS["full_r05"] = (RES, "table10_factor_runs.csv",
                       {"arm": "Full", "run": "5"}, "delay", 1.0, "value")
SCALARS["fso_p1"] = (RES, "full_seed_offset.csv", {"label": "Full-r01"},
                     "delay", 1.0, "mean")
SCALARS["fso_p5"] = (RES, "full_seed_offset.csv", {"label": "Full-r05"},
                     "delay", 1.0, "mean")
SCALARS["eso_cong"] = (RES, "eval_seed_offset.csv", {"label": "-Cong-r03"},
                       "diff_pct", 1.0, "value")
SCALARS["eso_util"] = (RES, "eval_seed_offset.csv", {"label": "-Util-r03"},
                       "diff_pct", 1.0, "value")

# ---- 가림 경로 배터리 (지연 평균·산포)
for _b in ("none", "scalar", "edgemean"):
    SCALARS["occ_" + _b] = (RES, "occlusion_paths_runs.csv",
                            {"blind": _b}, "delay", 1.0, "mean")
    SCALARS["occ_%s_sd" % _b] = (RES, "occlusion_paths_runs.csv",
                                 {"blind": _b}, "delay", 1.0, "sd")

# ---- 2x2 교차 (상태 x 보상) — 몫은 Full 모서리에서의 단순주효과다
for _c in ("risk_both", "qos_both", "qos_state", "qos_reward"):
    for _m in ("delay", "loss", "hop"):
        SCALARS["x_%s_%s" % (_c, _m)] = (
            RES, "cross_2x2.csv", {"cell": _c}, _m + "_mean", 1.0, "value")

# ---- 순위/값 4조건 배터리 (위험 차원 섞기)
for _c in ("intact", "risk_dimshuffle"):
    SCALARS["rv_%s_hop" % _c] = (RES, "_rank_vs_value_4cond_backup.csv",
                                 {"condition": _c}, "hop", 1.0, "mean")

# ---- 표 14 (재배선 비율별 제안 대 OSPF)
for _rr in ("0.2", "0.4", "0.6"):
    _k = _rr.replace(".", "")
    for _tag, _meth in (("o", "OSPF"), ("p", "Risk-DQN[frozen/no_revisit]")):
        SCALARS["%slp%s" % (_tag, _k)] = (
            RES, "table14_final_runs.csv",
            {"rewire_ratio": _rr, "method": _meth},
            "loss_published", 100.0, "mean")
        SCALARS["%srk%s" % (_tag, _k)] = (
            RES, "table14_final_runs.csv",
            {"rewire_ratio": _rr, "method": _meth},
            "risky_link_usage", 1.0, "mean")


# ======================= 한 칸으로 안 떨어지는 통계 =======================
# SCALARS 는 "CSV 하나 · 행 필터 하나 · 열 하나" 까지만 한다. 쌍체 표준편차,
# 최소검출차, 윈도우별 평균의 범위는 그 틀에 안 들어가므로 함수로 적는다.
# 함수 본문이 곧 선언이다 — 값을 적어 두는 것이 아니라 CSV 에서 만든다.
def _rows(fn, d=None):
    return list(csv.DictReader(
        io.open(os.path.join(d or RES, fn), encoding="utf-8-sig")))


def _t14_paired_delay(rr):
    """표 14 — 같은 토폴로지에서 (제안 - OSPF) 지연. 10쌍."""
    o, p = {}, {}
    for r in _rows("table14_final_runs.csv"):
        if r["rewire_ratio"] != rr:
            continue
        if r["method"] == "OSPF":
            o[r["topology"]] = float(r["delay"])
        elif r["method"] == "Risk-DQN[frozen/no_revisit]":
            p[r["topology"]] = float(r["delay"])
    ks = sorted(set(o) & set(p))
    assert len(ks) == 10, (rr, len(ks))
    return [p[k] - o[k] for k in ks]


def _mde(rr):
    """양측 5% · 검정력 50% 의 최소검출차 = t(.975, 9) * sd / sqrt(n).

    이 절이 지연을 근거로 못 쓰는 이유의 수치다. 관측된 차가 이보다
    작으면 '차이 없음' 이 아니라 '잴 수 없음' 이다.
    """
    ds = _t14_paired_delay(rr)
    return 2.262157 * _sd(ds) / len(ds) ** 0.5


def _nov_pair_sd():
    """11월 규약 — 같은 추첨 안에서 두 팔의 차. 그 차의 추첨 간 sd."""
    g = {}
    for r in _rows("eval_protocol_noise.csv"):
        if r["protocol"] == "nov":
            g.setdefault(r["draw"], {})[r["label"]] = float(r["delay"])
    ds = [g[k]["-Util-r03"] - g[k]["-Cong-r03"] for k in sorted(g)]
    assert len(ds) == 15, len(ds)
    return _sd(ds)


def _july_pair_sd():
    """7월 규약 — 비쌍체라 팔별 sd 의 평균에 sqrt(2) 를 싣는다."""
    g = {}
    for r in _rows("eval_protocol_noise.csv"):
        if r["protocol"] == "july":
            g.setdefault(r["label"], []).append(float(r["delay"]))
    sds = [_sd(v) for v in g.values()]
    return sum(sds) / len(sds) * (2 ** 0.5)


def _win_range(col):
    """윈도우별 평균의 최대-최소 (unstable 배터리)."""
    g = {}
    for r in _rows("window_unstable_runs.csv"):
        g.setdefault(r["window"], []).append(float(r[col]))
    ms = [sum(v) / len(v) for v in g.values()]
    return max(ms) - min(ms)


def _nr_sorted():
    """원본 트리의 NoRisk 5런 지연을 오름차순으로. 읽기만 한다."""
    vs = []
    for i in range(1, 6):
        rs = _rows("norisk_dynamic_run%02d_result.csv" % i, ORIG)
        vs.append(float(rs[0]["delay"]))
    return sorted(vs)


_QOS_METRICS = ("delay", "hop", "published", "path_sum", "compounded",
                "cong_live", "cong_live_per_hop")


def _qos_holm(scenario, variant, metric):
    """QoS 핸디캡 배터리의 Holm 보정 p.

    족을 적는 것이 핵심이다. 인쇄된 p 는 **시나리오 안에서** risk_dqn 과
    ospf 를 뺀 변종 5개 x 지표 7개 = 35개 검정의 족에서 나왔다. 족을 달리
    잡으면 같은 데이터로도 다른 p 가 나온다(26절). 그래서 족을 수식에
    같이 적는다.
    """
    import review_stats as RS
    by = {}
    for r in _rows("qos_baseline_handicap.csv"):
        if r["scenario"] == scenario:
            by.setdefault(r["variant"], []).append(r)
    base = by["risk_dqn"]
    vs = [v for v in sorted(by) if v not in ("risk_dqn", "ospf")]
    ps, keys = [], []
    for v in vs:
        for m in _QOS_METRICS:
            ps.append(RS.welch([float(r[m]) for r in by[v]],
                               [float(r[m]) for r in base])[3])
            keys.append((v, m))
    assert len(ps) == 35, len(ps)
    return RS.holm_bonferroni(ps, alpha=0.05)[keys.index((variant, metric))]


COMPUTED = {
    "nr_s0": lambda: _nr_sorted()[0],
    "nr_s1": lambda: _nr_sorted()[1],
    "nr_s2": lambda: _nr_sorted()[2],
    "nr_s3": lambda: _nr_sorted()[3],
    "nr_s4": lambda: _nr_sorted()[4],
    "qholm_pw_pub": lambda: _qos_holm("dynamic", "qos_post_warmup",
                                      "published"),
    "mde02": lambda: _mde("0.2"),
    "mde04": lambda: _mde("0.4"),
    "mde06": lambda: _mde("0.6"),
    "nov_pair_sd": _nov_pair_sd,
    "july_pair_sd": _july_pair_sd,
    "w_own_range": lambda: _win_range("risk_own"),
    "w_ref_range": lambda: _win_range("risk_ref"),
}

_LC = [u"(ospf_lc02-prop_lc02)/ospf_lc02*100",
       u"(ospf_lc04-prop_lc04)/ospf_lc04*100",
       u"(ospf_lc06-prop_lc06)/ospf_lc06*100"]

DERIVED = [
    # 표 11 비율·차이 행 (지연·손실은 %, 홉은 차)
    dict(name=u"표 11 — vs NoRisk 행 (유도)", draft="DRAFT_4_6_3_table11.md",
         row=[u"vs NoRisk-DQN"],
         cells=[(1, u"(prop_delay-nr_delay)/nr_delay*100"),
                (2, u"(prop_loss-nr_loss)/nr_loss*100"),
                (3, u"prop_hop-nr_hop")]),
    dict(name=u"표 11 — vs QoS-Model 행 (유도)",
         draft="DRAFT_4_6_3_table11.md",
         row=[u"vs QoS-Model <sup>주4</sup>"],
         cells=[(1, u"(prop_delay-qos_delay)/qos_delay*100"),
                (2, u"(prop_loss-qos_loss)/qos_loss*100"),
                (3, u"prop_hop-qos_hop")]),
    # 경로 일치율 표 — 한 칸에 숫자가 여러 개다(seq). 같은 조건내 값이
    # 여러 행에 복제돼 있어 행끼리의 일관성까지 같이 검사된다.
    dict(name=u"경로 일치율 표 — 조건내 (seq)",
         draft="DRAFT_4_6_3_table11.md",
         row=[u"조건내(잡음 바닥)"],
         cells=[(1, [u"ag_in_risk_both", u"ag_in_qos_both",
                     u"ag_in_qos_reward", u"ag_in_qos_state"])]),
    dict(name=u"경로 일치율 표 — 둘 다 교체 (seq)",
         draft="DRAFT_4_6_3_table11.md",
         row=[u"표 11 의 두 행 (둘 다 교체)"],
         cells=[(1, [u"ag_qos_both__risk_both"]),
                (2, [u"ag_in_qos_both", u"ag_in_risk_both"])]),
    dict(name=u"경로 일치율 표 — 보상만 교체 (seq)",
         draft="DRAFT_4_6_3_table11.md",
         row=[u"보상만 교체"],
         cells=[(1, [u"ag_qos_both__qos_state", u"ag_qos_reward__risk_both"]),
                (2, [u"ag_in_qos_both", u"ag_in_qos_state",
                     u"ag_in_qos_reward", u"ag_in_risk_both"])]),
    dict(name=u"경로 일치율 표 — 상태만 교체 (seq)",
         draft="DRAFT_4_6_3_table11.md",
         row=[u"상태만 교체"],
         cells=[(1, [u"ag_qos_both__qos_reward", u"ag_qos_state__risk_both"]),
                (2, [u"ag_in_qos_both", u"ag_in_qos_reward",
                     u"ag_in_qos_state", u"ag_in_risk_both"])]),
    # 표 14 복합확률 손실 감소 — 같은 세 값이 DRAFT_5 네 곳에 복제돼 있다.
    dict(name=u"DRAFT_5 판정표 ⑦ (유도)", draft="DRAFT_5_summary_abstract.md",
         anchor=u"발표값은 복합확률 기준", after=u"`loss_compounded`",
         exprs=_LC),
    dict(name=u"DRAFT_5 국문초록 (유도)", draft="DRAFT_5_summary_abstract.md",
         anchor=u"패킷 손실 감소가 유지되었다", after=u"각각", exprs=_LC),
    dict(name=u"DRAFT_5 영문초록 (유도)", draft="DRAFT_5_summary_abstract.md",
         anchor=u"on 30 link configurations not used for training",
         after=u"training", exprs=_LC),
    dict(name=u"DRAFT_5 요약 (유도)", draft="DRAFT_5_summary_abstract.md",
         anchor=u"종단간 손실 감소가 유지되었다",
         after=u"유지되었다", exprs=_LC),

    # =================== 본문 산문에 적은 파생 수치 ===================
    # 아래는 전부 감사에서 경보로 뜬 줄이다(집계 CSV 에 그 문자열로 없다).
    # 표 칸이 아니라 산문이라 "어디서 왔는지" 가 문서에만 있었다. 각 줄의
    # 인쇄값을 런 CSV 재집계로 갚는다.

    # ---- 4.6.1 시간 창
    dict(name=u"4.6.1 위험 측정자 폭", draft="DRAFT_4_6_1_temporal.md",
         anchor=u"재면 폭이", after=u"벌어지는데(",
         exprs=[u"w_own_range", u"w_ref_range"]),
    dict(name=u"4.6.1 차원 섞기 홉", draft="DRAFT_4_6_1_temporal.md",
         anchor=u"으로 짧아진다", after=u"8.002 →",
         exprs=[u"rv_risk_dimshuffle_hop"]),

    # ---- 4.6.2 표 10 (스레드 혼입 · 평가시드 치우침)
    dict(name=u"4.6.2 1스레드 재학습 대 Full", draft="DRAFT_4_6_2_table10.md",
         anchor=u"혼입이 있던 기록값",
         after=u"대비 **",
         exprs=[u"u1_delay-full_delay"]),
    # 한 줄에 "run03" 이 들어 있어 토큰 열에 03 이 끼어든다. 두 명세로 쪼갠다.
    dict(name=u"4.6.2 런별 최대 이동", draft="DRAFT_4_6_2_table10.md",
         anchor=u"움직인다(run03", after=u"최대 **",
         exprs=[u"u12_r03-u1_r03"]),
    dict(name=u"4.6.2 run03 팔 최악", draft="DRAFT_4_6_2_table10.md",
         anchor=u"움직인다(run03", after=u"팔 최악 ",
         exprs=[u"u12_r03"]),
    dict(name=u"4.6.2 공통난수 잡음항", draft="DRAFT_4_6_2_table10.md",
         anchor=u"공통난수) 이 항이",
         after=u"이 항이 **", exprs=[u"nov_pair_sd"]),
    dict(name=u"4.6.2 기준 행 교체 (유도)", draft="DRAFT_4_6_2_table10.md",
         anchor=u"로 내려간다(차", after=u"> **",
         exprs=[u"full_delay",
                u"(4*fso_p1+fso_p5)/5",
                u"full_delay-(4*fso_p1+fso_p5)/5",
                u"(full_delay-(4*fso_p1+fso_p5)/5)/((4*fso_p1+fso_p5)/5)*100"]),
    dict(name=u"4.6.2 정책 셋의 치우침", draft="DRAFT_4_6_2_table10.md",
         anchor=u"절제 팔에 해당하는",
         after=u"치우침",
         exprs=[u"eso_cong", u"eso_util",
                u"(full_r05-fso_p5)/fso_p5*100",
                u"(eso_cong+eso_util+(full_r05-fso_p5)/fso_p5*100)/3"]),
    dict(name=u"4.6.2 옳은 값 둘", draft="DRAFT_4_6_2_table10.md",
         anchor=u"독립 정책은 셋이고",
         after=u"옳은 값은 **",
         exprs=[u"(eso_cong+eso_util+(full_r05-fso_p5)/fso_p5*100)/3",
                u"(eso_cong+eso_util)/2"]),
    dict(name=u"4.6.2 워밍업 후 QoS", draft="DRAFT_4_6_2_table10.md",
         anchor=u"로 제안이 **3.7%", after=u"403.74 →",
         exprs=[u"qd_dyna_qos_post_warmup",
                u"(qd_dyna_risk_dqn-qd_dyna_qos_post_warmup)"
                u"/qd_dyna_qos_post_warmup*100",
                u"qd_dyna_risk_dqn-qd_dyna_qos_post_warmup"]),

    # ---- 4.6.3 표 11 (발표본 산수 · NoRisk 5런 · QoS 핸디캡 · 가림 · 2x2)
    dict(name=u"4.6.3 발표본 지연 산수", draft="DRAFT_4_6_3_table11.md",
         anchor=u"| Delay −23.7% |", after=u"−23.7% |",
         exprs=[u"(pdf_t5_prop_delay-pdf_t5_norisk_delay)"
                u"/pdf_t5_norisk_delay*100",
                u"(pdf_t5_prop_delay-pdf_t5_norisk_delay)"
                u"/pdf_t5_prop_delay*100"]),
    dict(name=u"4.6.3 발표본 손실 산수", draft="DRAFT_4_6_3_table11.md",
         anchor=u"| Loss −47.4% |", after=u"−47.4% |",
         exprs=[u"(pdf_t5_prop_loss-pdf_t5_norisk_loss)"
                u"/pdf_t5_norisk_loss*100",
                u"(pdf_t5_prop_loss-pdf_t5_norisk_loss)"
                u"/pdf_t5_prop_loss*100"]),
    dict(name=u"4.6.3 47.4% 가 되려면", draft="DRAFT_4_6_3_table11.md",
         anchor=u"이거나 제안 손실이",
         after=u"손실이",
         exprs=[u"pdf_t5_prop_loss/(1-pdf_body_t5_loss_pct/100)",
                u"pdf_t5_norisk_loss*(1-pdf_body_t5_loss_pct/100)"]),
    dict(name=u"4.6.3 NoRisk 5런 범위", draft="DRAFT_4_6_3_table11.md",
         anchor=u"런별 5개", after=u"런별 5개",
         exprs=[u"nr_s0", u"nr_s4"]),
    dict(name=u"4.6.3 NoRisk 5런 전체", draft="DRAFT_4_6_3_table11.md",
         anchor=u"326.611 / 332.029",
         exprs=[u"nr_s0", u"nr_s1", u"nr_s2", u"nr_s3", u"nr_s4"]),
    dict(name=u"4.6.3 워밍업 후 (주4 요약)", draft="DRAFT_4_6_3_table11.md",
         anchor=u"가 되어 **제안 기법이 "
                u"3.7% 느리다**", after=u"403.74 →",
         exprs=[u"qd_dyna_qos_post_warmup",
                u"(qd_dyna_risk_dqn-qd_dyna_qos_post_warmup)"
                u"/qd_dyna_qos_post_warmup*100"]),
    dict(name=u"4.6.3 워밍업 후 (본문)", draft="DRAFT_4_6_3_table11.md",
         anchor=u"제안 기법(350.12)이", after=u"403.74 →",
         exprs=[u"qd_dyna_qos_post_warmup", u"qd_dyna_risk_dqn",
                u"(qd_dyna_risk_dqn-qd_dyna_qos_post_warmup)"
                u"/qd_dyna_qos_post_warmup*100"]),
    dict(name=u"4.6.3 매 홉 재계획", draft="DRAFT_4_6_3_table11.md",
         anchor=u"재계획까지 허용하면 **",
         after=u"허용하면 **", exprs=[u"qd_dyna_qos_replan"]),
    dict(name=u"4.6.3 홉당 손실 비유의", draft="DRAFT_4_6_3_table11.md",
         anchor=u"비유의)** 로, 종단간",
         after=u"p_holm =", exprs=[u"qholm_pw_pub"]),
    dict(name=u"4.6.3 부호 역전 (네 값)", draft="DRAFT_4_6_3_table11.md",
         anchor=u"부호까지 바뀐다",
         after=u"바뀐다**(",
         exprs=[u"(qp_cong_qos_t0_stale-qp_cong_risk_dqn)"
                u"/qp_cong_qos_t0_stale*100",
                u"(qp_cong_qos_post_warmup-qp_cong_risk_dqn)"
                u"/qp_cong_qos_post_warmup*100",
                u"(qp_unst_qos_t0_stale-qp_unst_risk_dqn)"
                u"/qp_unst_qos_t0_stale*100",
                u"(qp_unst_qos_post_warmup-qp_unst_risk_dqn)"
                u"/qp_unst_qos_post_warmup*100",
                u"(qp_cong_qos_replan-qp_cong_risk_dqn)"
                u"/qp_cong_qos_replan*100",
                u"(qp_unst_qos_replan-qp_unst_risk_dqn)"
                u"/qp_unst_qos_replan*100"]),
    dict(name=u"4.6.3 학습된 QoS 모델", draft="DRAFT_4_6_3_table11.md",
         anchor=u"보다 **빠르다**",
         after=u"dynamic 지연 **",
         exprs=[u"qd_dyna_qos_dqn", u"qd_dyna_risk_dqn"]),
    dict(name=u"4.6.3 가림 5런 차", draft="DRAFT_4_6_3_table11.md",
         anchor=u"5런으로 올리면 차가",
         after=u"차가 **",
         exprs=[u"occ_scalar-occ_edgemean"]),
    dict(name=u"4.6.3 가림 산포", draft="DRAFT_4_6_3_table11.md",
         anchor=u"산포 ± 62.28", after=u"산포 ±",
         exprs=[u"occ_scalar_sd"]),
    dict(name=u"4.6.3 2x2 손실·홉 몫", draft="DRAFT_4_6_3_table11.md",
         anchor=u"손실·홉도 같은 방향",
         after=u"손실 몫 상태",
         exprs=[u"abs(x_qos_state_loss-x_risk_both_loss)"
                u"/(abs(x_qos_state_loss-x_risk_both_loss)"
                u"+abs(x_qos_reward_loss-x_risk_both_loss))*100",
                u"abs(x_qos_reward_loss-x_risk_both_loss)"
                u"/(abs(x_qos_state_loss-x_risk_both_loss)"
                u"+abs(x_qos_reward_loss-x_risk_both_loss))*100",
                u"abs(x_qos_state_hop-x_risk_both_hop)"
                u"/(abs(x_qos_state_hop-x_risk_both_hop)"
                u"+abs(x_qos_reward_hop-x_risk_both_hop))*100"]),
    dict(name=u"4.6.3 교차 일치율 (산문1)", draft="DRAFT_4_6_3_table11.md",
         anchor=u"같은 조건을 다시",
         after=u"일치율(",
         exprs=[u"ag_qos_both__qos_reward", u"ag_qos_state__risk_both"]),
    dict(name=u"4.6.3 교차 일치율 (산문2)", draft="DRAFT_4_6_3_table11.md",
         anchor=u"은 두 조건의 조건내 "
                u"일치율보다", after=u"(",
         exprs=[u"ag_qos_both__qos_reward", u"ag_qos_state__risk_both"]),

    # ---- 4.9 표 14
    dict(name=u"4.9 MDE (20.6절 인용)", draft="DRAFT_4_9_table14.md",
         anchor=u"부호도 일정하지 않다",
         after=u"MDE", exprs=[u"mde02", u"mde04", u"mde06"]),
    dict(name=u"4.9 손실 %p (세 비율)", draft="DRAFT_4_9_table14.md",
         anchor=u"%p, Holm p = 0.0000 / 0.0015 / 0.0001; 위험 링크",
         exprs=[u"plp04-olp04", u"plp06-olp06"]),
    dict(name=u"4.9 위험 링크 %p", draft="DRAFT_4_9_table14.md",
         anchor=u"%p, Holm p = 0.0000 / 0.0015 / 0.0001; 위험 링크",
         after=u"위험 링크",
         exprs=[u"prk02-ork02", u"prk04-ork04"]),
    dict(name=u"4.9 위험 링크 감소율", draft="DRAFT_4_9_table14.md",
         anchor=u"비율로 적으면", after=u"적으면",
         exprs=[u"-(prk04-ork04)/ork04*100", u"-(prk02-ork02)/ork02*100"]),
    dict(name=u"4.9 MDE (본문)", draft="DRAFT_4_9_table14.md",
         anchor=u"ms)보다 작다", after=u"(",
         exprs=[u"mde02", u"mde04", u"mde06"]),

    # ---- DRAFT_5
    dict(name=u"DRAFT_5 공정 비교군", draft="DRAFT_5_summary_abstract.md",
         anchor=u"공정 비교군(",
         after=u"비교군(",
         exprs=[u"qd_dyna_qos_post_warmup", u"qd_dyna_qos_replan"]),
    dict(name=u"DRAFT_5 기준 행 교체", draft="DRAFT_5_summary_abstract.md",
         anchor=u"요약·ABSTRACT 는 표 10", after=u"(",
         exprs=[u"full_delay-(4*fso_p1+fso_p5)/5",
                u"(full_delay-(4*fso_p1+fso_p5)/5)"
                u"/((4*fso_p1+fso_p5)/5)*100"]),
]

SIGNED = re.compile(u"([+\-−]?)\s*(\d+(?:\.\d+)?)")


def parse_signed(s):
    """부호 있는 첫 숫자 -> (값, 소수자릿수). 없으면 None."""
    s = re.sub(u"<sup>.*?</sup>", u"", s)
    s = s.replace(u"**", u"").replace(u"*", u"")
    m = SIGNED.search(s)
    if not m:
        return None
    sign = -1.0 if m.group(1) in (u"-", u"−") else 1.0
    tok = m.group(2)
    dec = len(tok.split(".")[1]) if "." in tok else 0
    return sign * float(tok), dec


def _sd(vals):
    """표본 표준편차(ddof=1). 값이 하나면 0."""
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def scalars():
    """선언된 이름 -> 값. 원본 트리는 읽기만 한다."""
    out = {}
    for name, (d, fn, filt, col, scale, agg) in SCALARS.items():
        rows = list(csv.DictReader(
            io.open(os.path.join(d or ORIG, fn), encoding="utf-8-sig")))
        sub = [r for r in rows if all(r.get(k) == v for k, v in filt.items())]
        assert sub, (name, fn, filt)
        vals = [float(r[col]) * scale for r in sub]
        if agg == "value":
            out[name] = vals[0]
        elif agg == "mean":
            out[name] = sum(vals) / len(vals)
        elif agg == "sd":
            out[name] = _sd(vals)
        elif agg == "min":
            out[name] = min(vals)
        elif agg == "max":
            out[name] = max(vals)
        elif agg == "n":
            out[name] = float(len(vals))
        else:
            raise ValueError((name, agg))
    # 한 칸으로 안 떨어지는 통계(쌍체 sd · MDE · 범위 · 최대차)는 따로.
    for name, fn in sorted(COMPUTED.items()):
        out[name] = fn()
    # eval 에 넘길 보조 함수 — 빌트인은 막혀 있다.
    out["min"] = min
    out["max"] = max
    out["abs"] = abs
    return out


def verify_derived(results):
    """비율·차이 칸을 자기 출처 칸에서 다시 계산한다."""
    env = scalars()
    print("")
    print(u"=== 유도 칸(비율·차이) — 자기 출처 칸에서 다시 계산 ===")
    for sp in DERIVED:
        dpath = os.path.join(HERE, sp["draft"])
        lines = io.open(dpath, encoding="utf-8").read().splitlines()
        pairs = []                      # (인쇄문자열, 수식, 줄번호)
        if "row" in sp:
            found = table_rows(dpath, sp["row"])
            if not found:
                print(u"  %-28s 초안 행 못 찾음 — 건너뜀" % sp["name"][:28])
                continue
            lineno, fields = found[0]
            for idx, expr in sp["cells"]:
                if idx >= len(fields):
                    continue
                if isinstance(expr, (list, tuple)):
                    # 한 칸에 숫자 여러 개 — 등장 순서대로 짝짓는다.
                    cs = re.sub(u"<sup>.*?</sup>", u"", fields[idx])
                    cs = cs.replace(u"**", u"").replace(u"*", u"")
                    toks = [mm.group(0) for mm in re.finditer(
                        u"[+\\-\u2212]?\\s*\\d+(?:\\.\\d+)?", cs)]
                    for k, e in enumerate(expr):
                        if k < len(toks):
                            pairs.append((toks[k], e, lineno))
                else:
                    pairs.append((fields[idx], expr, lineno))
        else:
            hit = [(i + 1, ln) for i, ln in enumerate(lines)
                   if sp["anchor"] in ln]
            if not hit:
                print(u"  %-28s 닻 못 찾음 — 건너뜀" % sp["name"][:28])
                continue
            lineno, ln = hit[0]
            # after 를 생략하면 줄 전체를 센다(숫자로 시작하는 줄).
            tail = ln.split(sp["after"], 1)[1] if sp.get("after") else ln
            toks = [m.group(0) for m in
                    re.finditer(u"[+\-−]?\s*\d+(?:\.\d+)?", tail)]
            for k, expr in enumerate(sp["exprs"]):
                if k < len(toks):
                    pairs.append((toks[k], expr, lineno))
        for printed, expr, lineno in pairs:
            ps = parse_signed(printed)
            if ps is None:
                continue
            pv, dec = ps
            wv = eval(expr, {"__builtins__": {}}, env)   # noqa: S307
            tol = 0.5 * 10 ** (-dec) * 1.0001
            ok = abs(wv - pv) <= tol
            print(u"  %-28s %-34s 인쇄 %+9.3f | 계산 %+9.4f 차 %.4f%s"
                  % (sp["name"][:28], expr[:34], pv, wv, abs(wv - pv),
                     u"" if ok else u"   <<< 불일치"))
            results.append({"spec": sp["name"], "counter": 0,
                            "draft": sp["draft"], "draft_line": lineno,
                            "row": expr, "csv": "(유도)", "column": "-",
                            "stat": "formula", "n": 0,
                            "printed_mean": "%.6f" % pv,
                            "recomputed_mean": "%.6f" % wv,
                            "printed_sd": "", "recomputed_sd": "",
                            "tol": "%.6f" % tol,
                            "diff_mean": "%.6f" % abs(wv - pv),
                            "diff_sd": "", "match": int(ok)})


def clean(cell):
    return cell.replace(u"**", u"").replace(u"*", u"").strip()


def parse_cell(cell):
    """인쇄된 칸에서 (평균, 표준편차, 평균의 소수자릿수) — 없으면 None."""
    s = re.sub(u"<sup>.*?</sup>", u"", cell)
    s = s.replace(u"**", u"").replace(u"*", u"").replace(u"−", u"-")
    s = s.split(u"(")[0]
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None
    dec = len(nums[0].split(".")[1]) if "." in nums[0] else 0
    sd = float(nums[1]) if (u"±" in s and len(nums) > 1) else None
    return float(nums[0]), sd, dec


def parse_range(cell):
    """`47.7~48.1 (95.8%)` 꼴에서 (하한, 상한, 소수자릿수, 괄호값) 을 뽑는다.

    단일 값(`0.00`, `100.0`)이면 하한=상한이다. 괄호가 없으면 넷째는 None.
    """
    s = re.sub(u"<sup>.*?</sup>", u"", cell)
    s = s.replace(u"**", u"").replace(u"*", u"").replace(u"−", u"-")
    s = s.replace(u"%", u"")
    paren = None
    if u"(" in s:
        inner = s[s.index(u"(") + 1:]
        pn = re.findall(r"\d+(?:\.\d+)?", inner)
        paren = float(pn[0]) if pn else None
        s = s.split(u"(")[0]
    s = s.replace(u"~", u" ").replace(u"∼", u" ")
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None
    dec = len(nums[0].split(".")[1]) if "." in nums[0] else 0
    lo = float(nums[0])
    hi = float(nums[1]) if len(nums) > 1 else lo
    return lo, hi, dec, paren


def mean_sd(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, sd


def colval(row, col, scale, derived):
    """CSV 열 또는 선언된 유도 열의 값."""
    if derived and col in derived:
        return derived[col](row) * scale
    return float(row[col]) * scale


def table_rows(path, leading, header_hint=None):
    """초안에서 선행 칸이 leading 과 같은 표 행들을 찾는다."""
    txt = io.open(path, encoding="utf-8").read().split(u"\n")
    start = 0
    if header_hint:
        for i, line in enumerate(txt):
            if line.startswith(header_hint):
                start = i
                break
    out = []
    for i in range(start, len(txt)):
        line = txt[i]
        if not line.lstrip().startswith(u"|"):
            continue
        f = [c.strip() for c in line.strip().strip(u"|").split(u"|")]
        if len(f) <= len(leading):
            continue
        if all(clean(f[j]) == clean(leading[j]) for j in range(len(leading))):
            out.append((i + 1, f))
    return out


def main():
    results = []
    for sp in SPECS:
        dpath = os.path.join(HERE, sp["draft"])
        rpath = os.path.join(sp.get("runs_dir", RES), sp["runs"])
        runs = list(csv.DictReader(io.open(rpath, encoding="utf-8-sig")))
        print("")
        print(u"=== %s   <- %s (%d행) ===" % (sp["name"], sp["runs"], len(runs)))
        for leading, filt in sp["rows"]:
            sub = [r for r in runs
                   if all(r.get(k) == v for k, v in filt.items())]
            found = table_rows(dpath, leading, sp.get("header_hint"))
            if not sub or not found:
                print(u"  %-30s 런 %d개 · 초안 행 %d개 — 건너뜀"
                      % (u"/".join(leading)[:30], len(sub), len(found)))
                continue
            lineno, fields = found[0]
            for cell in sp["cells"]:
                idx, col, scale, stat = cell[:4]
                sdcol = cell[4] if len(cell) > 4 else sp.get("sd_col")
                if idx >= len(fields):
                    continue
                if stat in ("range", "paren_mean"):
                    rr = parse_range(fields[idx])
                    if rr is None:
                        continue
                    plo, phi, dec, pparen = rr
                    vals = [colval(r, col, scale, sp.get("derived"))
                            for r in sub]
                    tol = 0.5 * 10 ** (-dec) * 1.0001
                    if stat == "range":
                        wlo, whi = min(vals), max(vals)
                        ok = abs(wlo - plo) <= tol and abs(whi - phi) <= tol
                        dm = max(abs(wlo - plo), abs(whi - phi))
                        pm, wm = plo, wlo
                        shown = (u"%.4f~%.4f" % (plo, phi),
                                 u"%.4f~%.4f" % (wlo, whi))
                    else:
                        if pparen is None:
                            continue
                        wm2 = sum(vals) / len(vals)
                        ok = abs(wm2 - pparen) <= tol
                        dm = abs(wm2 - pparen)
                        pm, wm = pparen, wm2
                        shown = (u"(%.4f)" % pparen, u"(%.4f)" % wm2)
                    mark = u"" if ok else u"   <<< 불일치"
                    print(u"  %-26s %-16s 인쇄 %-19s | 재집계 %-19s "
                          u"(n=%d) 차 %.4f%s"
                          % (u"/".join(leading)[:26], col, shown[0], shown[1],
                             len(vals), dm, mark))
                    results.append({"spec": sp["name"],
                                    "counter": int(bool(sp.get("counter"))),
                                    "draft": sp["draft"],
                                    "draft_line": lineno,
                                    "row": u"/".join(leading),
                                    "csv": sp["runs"], "column": col,
                                    "stat": stat, "n": len(vals),
                                    "printed_mean": "%.6f" % pm,
                                    "recomputed_mean": "%.6f" % wm,
                                    "printed_sd": ("%.6f" % phi
                                                   if stat == "range" else ""),
                                    "recomputed_sd": ("%.6f" % max(vals)
                                                      if stat == "range"
                                                      else ""),
                                    "tol": "%.6f" % tol,
                                    "diff_mean": "%.6f" % dm,
                                    "diff_sd": "", "match": int(ok)})
                    continue
                pr = parse_cell(fields[idx])
                if pr is None:
                    continue
                pmean, psd, dec = pr
                if stat.startswith("value"):
                    want_m = float(sub[0][col]) * scale
                    want_sd = (float(sub[0][sdcol]) * scale
                               if stat == "value_sd" else None)
                    n = 1
                else:
                    if sp.get("nest"):
                        groups = {}
                        for r in sub:
                            groups.setdefault(r[sp["nest"]], []).append(
                                float(r[col]) * scale)
                        vals = [sum(v) / len(v) for v in groups.values()]
                    else:
                        vals = [float(r[col]) * scale for r in sub]
                    want_m, want_sd = mean_sd(vals)
                    if stat == "mean":
                        want_sd = None
                    n = len(vals)
                tol = 0.5 * 10 ** (-dec) * 1.0001
                dm = abs(want_m - pmean)
                ok = dm <= tol
                dsd = None
                if psd is not None and want_sd is not None:
                    dsd = abs(want_sd - psd)
                    ok = ok and dsd <= tol
                mark = u"" if ok else u"   <<< 불일치"
                print(u"  %-26s %-16s 인쇄 %9.4f%s | 재집계 %9.4f%s "
                      u"(n=%d) 차 %.4f%s"
                      % (u"/".join(leading)[:26], col, pmean,
                         (u" ±%.3f" % psd) if psd is not None else u"",
                         want_m,
                         (u" ±%.3f" % want_sd) if want_sd is not None else u"",
                         n, dm, mark))
                results.append({"spec": sp["name"],
                                "counter": int(bool(sp.get("counter"))),
                                "draft": sp["draft"],
                                "draft_line": lineno,
                                "row": u"/".join(leading), "csv": sp["runs"],
                                "column": col, "stat": stat, "n": n,
                                "printed_mean": "%.6f" % pmean,
                                "recomputed_mean": "%.6f" % want_m,
                                "printed_sd": ("%.6f" % psd
                                               if psd is not None else ""),
                                "recomputed_sd": ("%.6f" % want_sd
                                                  if want_sd is not None
                                                  else ""),
                                "tol": "%.6f" % tol,
                                "diff_mean": "%.6f" % dm,
                                "diff_sd": ("%.6f" % dsd
                                            if dsd is not None else ""),
                                "match": int(ok)})

    verify_derived(results)

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    live = [r for r in results if not r["counter"]]
    ctr = [r for r in results if r["counter"]]
    bad = [r for r in live if not r["match"]]
    print("")
    print(u"=== 종합: 초안 주장 칸 %d개 검증, 불일치 %d개 ==="
          % (len(live), len(bad)))
    if ctr:
        diff = sum(1 for r in ctr if not r["match"])
        print(u"    (대립 가설 %d칸 중 %d칸이 어긋남 — 인쇄값이 체크포인트"
              u" 사본을 가린다는 뜻이니 정상이다)" % (len(ctr), diff))
    for r in bad:
        print(u"  %s  %s  %s %s: 인쇄 %s±%s vs 재집계 %s±%s (허용 %s)"
              % (r["draft"], r["row"], r["column"], r["stat"],
                 r["printed_mean"], r["printed_sd"],
                 r["recomputed_mean"], r["recomputed_sd"], r["tol"]))
    print(u"저장: " + OUT)


if __name__ == "__main__":
    main()
