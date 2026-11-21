# -*- coding: utf-8 -*-
"""표 11 NoRisk 행의 부호 역전을 **표현 몫 / 보상 몫** 으로 쪼갠다 (2x2).

문제
----
표 11 의 `NoRisk` 행은 "위험 인식을 뗐다" 로 읽히지만, 실제로 `use_risk=False`
는 **둘을 동시에** 바꾼다 (`network_env.py`):
  * 상태: 엣지 위험 50칸 -> delay/loss 정규화 100칸 (125 -> 175 차원)
  * 보상: 위험·시나리오 기반 보상 -> QoS 기반 보상
    (`reward -= delay*0.015`, `-= loss*8.0`, 혼잡 -2.0, 걸음 -0.08,
     그리고 종단 `total_risk*6.0` 벌점이 **사라진다**)
그래서 NoRisk 와 제안 팔의 차이를 '위험 인식의 기여' 로 읽을 수 없다.

설계
----
`CrossEnv` 로 두 인자를 떼어 2x2 를 만든다. 네 칸 모두 같은 하네스
(`evaluate_cell` -> `eval_ablation_risky.evaluate`), 같은 `EVAL_SEED`,
같은 200 에피소드다. 규약을 섞지 않는 것이 중요하다 — 평가 규약이 다르면
팔 간 차에 +-11.6 ms 가 실린다.

                    보상=위험            보상=QoS
    상태=위험       risk_both (Full)     qos_reward
    상태=QoS        qos_state            qos_both (NoRisk)

읽을 때 지켜야 할 한계 셋
-------------------------
1. **실효 표본.** 칸마다 모델 5개지만 서로 다른 값은 그보다 적다. Full 은
   5런 중 4개가 소수 4자리까지 같다. 칸별 '서로 다른 값의 수' 를 같이 찍는다.
2. **보상 크기 혼입.** 같은 전이에서 위험 보상(-20.68/-29.53/-29.00/-29.73)과
   QoS 보상(-1.78/-13.32/-10.90/-14.35)의 **크기 자체가 다르다**. 그래서
   보상 인자의 효과는 '무엇을 보상하는가' 와 '보상 크기' 를 섞는다.
   비교 가능성을 위해 고치지 않았고, 한계로 남긴다.
3. **가법성은 측정이지 메커니즘이 아니다.** 상호작용이 작다는 것은 이
   격자에서 두 주효과가 거의 더해진다는 뜻일 뿐이다.
"""
import csv
import io
import os

import review_stats as S

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "review_results")
SRC = ("cross_runs.csv", "cross_baselines.csv")
OUT_CELLS = os.path.join(RESULT_DIR, "cross_2x2.csv")
OUT_TESTS = os.path.join(RESULT_DIR, "cross_2x2_tests.csv")

CELL_ORDER = ["risk_both", "qos_reward", "qos_state", "qos_both"]
CELL_LABEL = {"risk_both": "상태=위험 / 보상=위험  (Full)",
              "qos_reward": "상태=위험 / 보상=QoS",
              "qos_state": "상태=QoS  / 보상=위험",
              "qos_both": "상태=QoS  / 보상=QoS   (NoRisk)"}
METRICS = ["delay", "loss", "hop"]


def load():
    cells = {}
    for name in SRC:
        p = os.path.join(RESULT_DIR, name)
        if not os.path.exists(p):
            print("!! 없다: %s" % p)
            continue
        with io.open(p, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                cells.setdefault(row["cell"], []).append(row)
    return cells


def vals(rows, metric):
    return [float(r[metric]) for r in rows]


def mean(v):
    return sum(v) / len(v)


def main():
    cells = load()
    # 규약 혼입 거부: 네 칸이 같은 에피소드 수·같은 평가 시드여야 한다
    prot = {}
    for c in cells:
        for r in cells[c]:
            prot.setdefault((r.get("episodes", "?"),
                             r.get("eval_seed", "?")), []).append(c)
    if not prot:
        print("읽을 행이 없다 — `eval` 과 `eval_base` 를 먼저 돌릴 것.")
        return
    if any(k[0] == "?" or k[1] == "?" for k in prot):
        print("!! CSV 에 episodes/eval_seed 열이 없다 — 어느 규약으로 쟀는지"
              " 알 수 없으므로 2x2 를 만들지 않는다.")
        print("   패치된 스크립트로 `eval` 과 `eval_base` 를 다시 돌릴 것.")
        return
    if len(prot) > 1:
        print("!! 평가 규약이 섞였다 — 2x2 를 만들면 안 된다:")
        for k in sorted(prot, key=str):
            print("   에피소드 %s / EVAL_SEED %s : %s"
                  % (k[0], k[1], ", ".join(sorted(set(prot[k])))))
        print("   같은 규약으로 다시 채점할 것"
              " (`eval` 과 `eval_base` 둘 다 --eval-episodes 200).")
        return
    (EPS, SEED), = prot.keys()
    print("평가 규약: 에피소드 %s  EVAL_SEED %s  (네 칸 동일)" % (EPS, SEED))

    missing = [c for c in CELL_ORDER if c not in cells]
    if missing:
        print("2x2 가 아직 안 찼다. 없는 칸: %s" % ", ".join(missing))
        print("  `python train_state_reward_cross.py eval` 와"
              " `... eval_base` 를 둘 다 돌려야 한다.")
        return

    print("=" * 84)
    print("표 11 NoRisk 부호 역전의 분해 — 상태 표현 2 x 보상 함수 2")
    print("=" * 84)

    crows = []
    for c in CELL_ORDER:
        r = cells[c]
        d = vals(r, "delay")
        line = "  %-34s n=%d" % (CELL_LABEL[c], len(d))
        for m in METRICS:
            v = vals(r, m)
            mu, sd = S.mean_std(v)
            line += "  %s %.4f+-%.4f" % (m, mu, sd)
        nd = len(set("%.6f" % x for x in d))
        line += "  서로다른지연값 %d" % nd
        print(line)
        row = dict(cell=c, label=CELL_LABEL[c], n=len(d),
                   n_distinct_delay=nd, state=r[0]["state"],
                   reward=r[0]["reward"], state_size=r[0]["state_size"])
        for m in METRICS:
            mu, sd = S.mean_std(vals(r, m))
            row[m + "_mean"] = "%.6f" % mu
            row[m + "_sd"] = "%.6f" % sd
        crows.append(row)

    print("\n  (서로 다른 값의 수가 n 보다 작으면 실효 표본은 그만큼 작다.)")

    # ── 주효과와 상호작용 ────────────────────────────────────────────────
    print("\n" + "-" * 84)
    print("주효과 / 상호작용  (지연, ms. 음수 = 그 교체가 지연을 줄인다)")
    print("-" * 84)

    trows, pv, keys = [], [], []

    def test(name, a, b, metric="delay"):
        """b - a 의 차이와 Welch. a 가 기준."""
        xs, ys = vals(cells[a], metric), vals(cells[b], metric)
        diff, t, df, p = S.welch(xs, ys)
        trows.append(dict(metric=metric, comparison=name,
                          base=a, arm=b,
                          base_mean="%.6f" % mean(xs),
                          arm_mean="%.6f" % mean(ys),
                          diff="%+.6f" % (mean(ys) - mean(xs)),
                          t=("%.4f" % t) if t == t else "",
                          df=("%.2f" % df) if df == df else "",
                          p_raw=("%.6g" % p) if p == p else "",
                          p_holm=""))
        pv.append(p)
        keys.append(len(trows) - 1)
        return mean(ys) - mean(xs)

    for m in METRICS:
        s_at_risk = test("상태교체 @보상=위험", "risk_both", "qos_state", m)
        s_at_qos = test("상태교체 @보상=QoS", "qos_reward", "qos_both", m)
        r_at_risk = test("보상교체 @상태=위험", "risk_both", "qos_reward", m)
        r_at_qos = test("보상교체 @상태=QoS", "qos_state", "qos_both", m)
        both = test("둘 다 교체 (NoRisk 행)", "risk_both", "qos_both", m)
        inter = (s_at_qos - s_at_risk)
        tot = abs(s_at_risk) + abs(r_at_risk)
        print("\n  [%s]" % m)
        print("    상태 교체  보상=위험일 때 %+.4f / 보상=QoS일 때 %+.4f"
              % (s_at_risk, s_at_qos))
        print("    보상 교체  상태=위험일 때 %+.4f / 상태=QoS일 때 %+.4f"
              % (r_at_risk, r_at_qos))
        print("    둘 다      %+.4f   (주효과 합 %+.4f, 상호작용 %+.4f)"
              % (both, s_at_risk + r_at_risk, inter))
        if tot > 0:
            print("    차이의 몫  상태 %.1f%% / 보상 %.1f%%"
                  % (100.0 * abs(s_at_risk) / tot,
                     100.0 * abs(r_at_risk) / tot))
        trows.append(dict(metric=m, comparison="상호작용(차이의 차이)",
                          base="", arm="", base_mean="", arm_mean="",
                          diff="%+.6f" % inter, t="", df="",
                          p_raw="", p_holm="(검정 안 함)"))

    adj = S.holm_bonferroni(pv)
    for i, k in enumerate(keys):
        trows[k]["p_holm"] = "%.6g" % adj[i] if adj[i] == adj[i] else ""

    print("\n" + "-" * 84)
    print("Welch + Holm (족 크기 %d — 세 지표 x 다섯 비교)" % len(pv))
    print("-" * 84)
    print("  %-8s %-24s %12s %10s %10s %9s"
          % ("지표", "비교", "차이", "t", "p", "p_holm"))
    for r in trows:
        if not r["p_raw"]:
            print("  %-8s %-24s %12s %10s %10s %9s"
                  % (r["metric"], r["comparison"], r["diff"], "", "",
                     r["p_holm"]))
            continue
        print("  %-8s %-24s %12s %10s %10s %9s"
              % (r["metric"], r["comparison"], r["diff"], r["t"],
                 r["p_raw"], r["p_holm"]))

    for path, rows, flds in (
            (OUT_CELLS, crows, list(crows[0].keys())),
            (OUT_TESTS, trows, list(trows[0].keys()))):
        with io.open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=flds, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("\n-> %s (%d행)" % (path, len(rows)))

    print("\n한계 (해석에 반드시 붙일 것)")
    print("  1. 칸마다 모델 5개지만 서로 다른 값은 더 적다(위 표 참조).")
    print("  2. 보상 인자는 '무엇을 보상하는가' 와 '보상 크기' 를 섞는다 —")
    print("     같은 전이에서 위험 보상과 QoS 보상의 절댓값 자체가 다르다.")
    print("  3. 상호작용이 작다는 것은 이 격자에서 두 주효과가 거의"
          " 더해진다는 측정이고, 메커니즘 주장이 아니다.")


if __name__ == "__main__":
    main()
