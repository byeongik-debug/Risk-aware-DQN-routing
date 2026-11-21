# -*- coding: utf-8 -*-
"""C-1 축소형의 도달률이 **왜** 방법을 못 가리는지 — 구조인가 검정력인가.

`FINDINGS_mobility_proxy.md` 6-2-3절을 "도달률로는 아무 방법도 구별할 수
없게 **설계되어** 있었다"로 닫았는데, 그 문장이 `node` 단위에서 과했다.
고립(`stuck_isolated`)의 정의는 "자기 노드가 떠났다"가 아니라 **"서 있는
노드의 차수가 0"**이고(`reeval_mobility_proxy.py:276-277`), 어디에 서는지는
라우팅이 고른다. 즉 고립은 라우팅과 끊겨 있지 않다.

그래서 이 스크립트는 넷을 분리해 잰다. 6-2-5절의 모든 숫자가 여기서 나온다.

  --routes   원본 규칙에서 실패 경로가 정말 하나인지 420행 전수로 센다.
  --hop      노출 가설의 **전제**(제안 경로가 더 길다)를 짝 검정한다.
  --power    원인과 결과에 각각 셀이 몇 개 필요한지 — 짝 분산으로 센다.
  --degree   차수 가설을 반증한다(밟은 노드의 차수). 모델 적재가 필요하다.

읽는 법: `--hop`이 확정되고 `--power`에서 고립률 필요 n이 현행(20)보다 크면,
`node`의 무력함은 **구조가 아니라 검정력**이다. `--degree`가 0에 가까우면
기제는 차수가 아니다. 기제를 재지 않고 주장하지 않기 위한 절제들이다.

주의 1 — `mobility_proxy*.csv`의 실패 경로 열과 `success`는 **이미 퍼센트**다
(`reeval_mobility_proxy.py:392-393`이 `count/episodes*100`으로 쓴다). 셀 집계는
행 평균이고, 개수로 착각해 에피소드로 다시 나누면 20/12배 부풀어 나온다.

주의 2 — `--degree` 탐침은 에피소드마다 재시드하고(`seed*1000+ep`) 본 측정은
(시나리오, 시드)마다 한 번 시드한다. 절대값이 CSV와 다르므로 탐침 숫자는
**방법 간 대조에만** 쓴다. 본 표에 넣지 말 것.

주의 3 — `review_stats.required_n`과 `min_detectable_effect`는 두 표본(Welch)
용이다. 셀끼리 뺀 짝 설계에는 짝 분산을 써야 하므로 여기서 따로 계산한다.
"""

import argparse
import csv
import io
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from review_stats import (_z_quantile, fmt_p, holm_bonferroni,  # noqa: E402
                          mean_std, t_critical, t_two_sided_p)

RESULTS = os.path.join(HERE, "review_results")
RULE_CSV = {"all": "mobility_proxy.csv",
            "prev": "mobility_proxy_prev.csv",
            "none": "mobility_proxy_none.csv"}
ROUTES = ("invalid", "step_limit", "stuck_isolated", "stuck_nopath",
          "stuck_decode")
BASELINES = ("ospf_bw_replan", "ospf_delay_replan")
# 도달률이 포화하지 않는 조건만 쓴다. churn 0.0은 회귀 검사용 대조다.
CONDS = (("link", "0.0"), ("node", "0.2"), ("node", "0.5"),
         ("node_protected", "0.2"), ("node_protected", "0.5"))
NODE_CONDS = (("node", "0.2"), ("node", "0.5"))
CELLS_NOW = 20          # 시드 5 x 시나리오 4
POWER = 0.80


def load(rule="none"):
    path = os.path.join(RESULTS, RULE_CSV[rule])
    with io.open(path, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("scenario")]


def paired_t(d):
    """차 목록 -> (평균, MDE, p). reeval_mobility_proxy.py:1434와 같은 식."""
    if len(d) < 3:
        return float("nan"), float("nan"), float("nan")
    m, sd = mean_std(d)
    se = sd / math.sqrt(len(d))
    if se == 0.0:
        return m, 0.0, (0.0 if m else float("nan"))
    df = len(d) - 1
    return m, t_critical(df) * se, t_two_sided_p(m / se, df)


def cells(rows, unit, p, field):
    """(시나리오, 시드) -> {방법: 값}. 짝을 깨지 않으려고 열쇠로 모은다."""
    out = {}
    for r in rows:
        if r["churn_unit"] != unit or r["churn_p"] != p:
            continue
        if r[field] == "":
            continue
        out.setdefault((r["scenario"], r["seed"]), {})[r["variant"]] = \
            float(r[field])
    return out


def diffs(rows, unit, p, field, base):
    key = cells(rows, unit, p, field)
    return [v["risk_dqn"] - v[base] for _, v in sorted(key.items())
            if "risk_dqn" in v and base in v]


def report_routes(rule):
    """실패 경로가 하나인가. 0 아닌 **행 수**를 세는 것이 요점이다 —

    열이 퍼센트라 총합은 뜻이 없지만, "한 행도 0이 아닌 적이 없다"는
    단위와 무관한 사실이고 항등식도 아니다.
    """
    rows = load(rule)
    print("실패 경로 전수 — 규칙=%s, %d행 (행당 %s에피소드)"
          % (rule, len(rows), rows[0]["episodes"]))
    print("  %-14s %9s %9s %12s" % ("경로", "행평균%", "최대%", "0아닌행"))
    for k in ROUTES:
        v = [float(r[k]) for r in rows]
        print("  %-14s %8.4f%% %8.2f%% %8d / %d"
              % (k, mean_std(v)[0], max(v), sum(1 for x in v if x > 0),
                 len(v)))
    print("  넷이 0아닌행 0이면 실패 경로는 하나다 — 행동 마스킹도, 목적지")
    print("  단절도, 디코딩 규칙도 한 번도 실패를 만들지 않았다는 뜻이다.")


def report_hop(rule):
    """노출 가설의 전제. 상대를 **둘 다** 본다 —

    품질 21비교는 `ospf_delay`만 상대로 잡는다(목적 함수 계열이 같아서
    공정 비교군이다). 그 선택이 정당해도, 유의한 격차가 `ospf_bw` 쪽에
    있으면 비교군 선택이 그 격차를 가린다. 그래서 둘 다 싣는다.
    """
    rows = load(rule)
    print("홉 짝 비교 (risk_dqn - 기준선) — 규칙=%s, 셀=시나리오x시드" % rule)
    for field in ("hop", "paired_hop"):
        print("  [%s]%s" % (field,
                            " (셋 다 도달한 짝만)" if "paired" in field
                            else " (성공 에피소드 평균)"))
        res = []
        for unit, p in CONDS:
            for base in BASELINES:
                d = diffs(rows, unit, p, field, base)
                if len(d) < 3:
                    continue
                m, mde, pv = paired_t(d)
                res.append(("%-14s %-4s vs %-18s" % (unit, p, base),
                            m, mde, len(d), pv))
        adj = holm_bonferroni([r[4] for r in res])
        for r, a in zip(res, adj):
            aa = a[0] if isinstance(a, (tuple, list)) else a
            print("    %s n=%2d %+7.3f 홉  MDE %.3f  p=%-9s Holm %-9s %s"
                  % (r[0], r[3], r[1], r[2], fmt_p(r[4]), fmt_p(aa),
                     "생존" if aa < 0.05 else ""))
    print("  `hop`은 성공한 에피소드만 평균한다 — 고립은 짧게 끊긴 실패를")
    print("  빼므로 고립이 많은 쪽을 짧게 보이게 만든다. 즉 생존자 편향은")
    print("  이 차이를 **줄이는** 방향이고, 그래도 남으면 전제는 확정이다.")


def report_power(rule):
    """원인과 결과에 셀이 각각 몇 개 필요한가. 짝 분산으로 센다."""
    rows = load(rule)
    za, zb = _z_quantile(1 - 0.05 / 2.0), _z_quantile(POWER)
    print("짝 설계 필요 셀 수 — 규칙=%s, 검정력 %.2f, 양측 0.05, 현행 %d개"
          % (rule, POWER, CELLS_NOW))
    for field, lab, unit_lab in (("hop", "홉(원인)", "홉"),
                                 ("stuck_isolated", "고립률(결과)", "%p")):
        print("  [%s]" % lab)
        for unit, p in NODE_CONDS:
            for base in BASELINES:
                d = diffs(rows, unit, p, field, base)
                if len(d) < 3:
                    continue
                m, sd = mean_std(d)
                if m == 0.0 or sd == 0.0:
                    print("    %-14s %-4s vs %-18s 차 %+.3f %s — 필요 n "
                          "정의 불가" % (unit, p, base, m, unit_lab))
                    continue
                n = math.ceil(((za + zb) * sd / abs(m)) ** 2)
                shown = "%d" % n if n < 10 ** 6 else ">1e6"
                print("    %-14s %-4s vs %-18s 차 %+7.3f %-2s 짝sd %6.3f  "
                      "필요 n=%s"
                      % (unit, p, base, m, unit_lab, sd, shown))
    print("  원인이 현행 셀로 보이고 결과가 안 보이면, `node`의 무력함은")
    print("  구조가 아니라 **검정력**이다 — 표본을 늘리면 가려진다.")
    print("  반대로 `link`와 `node_protected`는 실패가 없거나 불가능하므로")
    print("  표본을 늘려도 달라지지 않는다. 그쪽이 구조다.")


def report_degree(episodes):
    """차수 가설의 절제. 밟은 노드의 차수를 직접 센다.

    본 측정과 **같은** `walk()`을 쓰고 `env.visited`만 읽는다(관찰 전용).
    규칙은 원본(`none`)으로 고정한다 — 내 배제 규칙이 경로를 바꾸면
    차수 비교가 그 인공물을 재게 된다.
    """
    import random

    import numpy as np
    import torch

    import reeval_loss_conventions as RLC
    import reeval_mobility_proxy as M
    from topology import DESTINATION, EDGES, NUM_NODES

    deg = dict((n, 0) for n in range(NUM_NODES))
    for e in EDGES:
        deg[e[0]] += 1
        deg[e[1]] += 1
    print("기준 위상 차수: 평균 %.2f, 최소 %d, 최대 %d / 출발 %d(차수 %d) "
          "목적 %d(차수 %d)"
          % (sum(deg.values()) / float(NUM_NODES), min(deg.values()),
             max(deg.values()), M.SOURCE, deg[M.SOURCE], DESTINATION,
             deg[DESTINATION]))

    model = RLC.load_risk_model()
    device = next(model.parameters()).device

    def make_pick(cost):
        if cost is None:
            def pick(e):
                bad = M.blocked_by(e, "none")
                valid = [a for a in e.get_valid_actions() if a not in bad]
                if not valid:
                    return None
                st = e.get_state()
                with torch.no_grad():
                    q = model(torch.FloatTensor(st)
                              .unsqueeze(0).to(device))[0]
                m = torch.full_like(q, -1e9)
                for a in valid:
                    m[a] = q[a]
                return int(torch.argmax(m).item())
            return pick

        def pick(e, _c=cost):
            path = M.plan_live(e, M.blocked_by(e, "none"), _c)
            return path[1] if len(path) >= 2 else None
        return pick

    print("\n%d에피소드 x %d시드 x %d시나리오, 규칙=배제없음"
          % (episodes, len(M.SEEDS), len(M.SCENARIOS)))
    print("  %-18s %7s %11s %10s %13s"
          % ("방법", "고립%", "걸음(전체)", "밟은차수", "고립지점차수"))
    store = {}
    for unit, p in (("node", 0.20), ("node", 0.50)):
        print("  -- unit=%s churn=%.2f" % (unit, p))
        for variant, cost in M.VARIANTS:
            pick = make_pick(cost)
            iso, cnt, steps, degs, idg = 0, 0, [], [], []
            for seed in M.SEEDS:
                for sc in M.SCENARIOS:
                    for ep in range(episodes):
                        random.seed(seed * 1000 + ep)
                        np.random.seed(seed * 1000 + ep)
                        env = M.make_env(sc, p, (seed, ep), unit)
                        _, reason = M.walk(env, pick)
                        cnt += 1
                        steps.append(len(env.visited))
                        degs.extend(deg[n] for n in env.visited)
                        if reason == "stuck_isolated":
                            iso += 1
                            idg.append(deg[env.current_node])
            row = (100.0 * iso / cnt, mean_std(steps)[0], mean_std(degs)[0],
                   mean_std(idg)[0] if idg else float("nan"))
            store[(unit, p, variant)] = row
            print("  %-18s %6.2f%% %11.3f %10.3f %13.3f" % ((variant,) + row))
    print("\n  제안 - ospf_bw 대조:")
    for unit, p in (("node", 0.20), ("node", 0.50)):
        a = store[(unit, p, "risk_dqn")]
        b = store[(unit, p, "ospf_bw_replan")]
        print("    %s %.2f  고립 %+.2f %%p | 걸음 %+.3f | 밟은차수 %+.3f "
              "| 고립지점차수 %+.3f"
              % (unit, p, a[0] - b[0], a[1] - b[1], a[2] - b[2],
                 a[3] - b[3]))
    print("  밟은차수 차가 0에 가깝고 고립지점차수가 제안 쪽에서 더 높으면")
    print("  차수 가설은 반증이다. 걸음 차가 CSV의 홉 차와 같은 크기로")
    print("  남으면 그 차는 생존자 편향이 만든 것이 아니다.")
    print("  이 탐침은 에피소드마다 재시드한다 — 절대값을 본 표에 쓰지 말 것.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rule", default="none", choices=sorted(RULE_CSV),
                    help="어느 디코딩 규칙의 CSV를 읽을지 (기본 none=원본)")
    ap.add_argument("--routes", action="store_true")
    ap.add_argument("--hop", action="store_true")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--degree", type=int, metavar="EPISODES", default=0,
                    help="차수 탐침을 셀당 N에피소드로 돌린다(모델 적재)")
    a = ap.parse_args()
    any_csv = a.routes or a.hop or a.power
    if not any_csv and not a.degree:
        a.routes = a.hop = a.power = True
    for flag, fn in ((a.routes, report_routes), (a.hop, report_hop),
                     (a.power, report_power)):
        if flag:
            fn(a.rule)
            print()
    if a.degree:
        report_degree(a.degree)


if __name__ == "__main__":
    main()
