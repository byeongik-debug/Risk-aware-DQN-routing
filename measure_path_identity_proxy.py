# -*- coding: utf-8 -*-
"""경로 '동일' 판정이 **대리 판정**인지 잰다.

왜 필요한가
-----------
23.9.6.15 의 가장 강한 새 주장은 "`intact` 정책이 risk 비용 Dijkstra 경로를
**69.8%** 재현한다" 다. 그런데 그 판정은 **지연 6자리 일치 + 홉 일치**로
했고 노드열을 직접 비교한 게 아니다. 서로 다른 경로가 같은 홉에서 같은
지연을 가지면 69.8% 는 **과대추정**이다.

이게 형식적 걱정이 아닌 이유: 23.9.6.14 가 누산 때문에 **링크 지연이
clip 상한(120)에 포화**하는 것을 쟀다. 포화한 링크는 **정확히 같은 값**을
가지므로, 포화 링크만 다른 두 경로는 지연이 **진짜로 같아진다**.

무엇을 재는가
-------------
1. 홉 계급(6~MAXH) 안에서 지연이 1e-6 이내로 붙은 **경로 쌍 수**.
2. **risk-Dijkstra 경로와 같은 홉·같은 지연을 갖는 다른 경로 수** — 이게
   69.8% 판정의 거짓양성 기회를 직접 잰다.
3. clip 상한에 붙은 링크 수(포화가 원인인지 보려고).

틱을 맞추는 것이 중요하다. 정책은 `reset` 1 + `WARMUP` 10 + h 틱째에
도착하므로(23.9.6.14), 측정도 **WARMUP + h** 에서 한다. h 를 6~10 으로
훑어 포화가 깊어질수록 충돌이 늘어나는지 본다.

한계
----
- 충돌 '기회' 를 재는 것이고, 정책이 실제로 그 충돌 경로를 골랐는지는
  아니다. 충돌이 0 이면 대리 판정이 정확하다는 **충분조건**이고, 0 이
  아니면 69.8% 의 상한만 말해 준다.
- 홉 MAXH 를 넘는 경로는 안 본다(정책 홉 평균 6.95~8.20).

출력: review_results/path_identity_proxy.csv
환경변수: PIP_ENVS(기본 30) · PIP_MAXH(기본 12) · PIP_TOL(기본 1e-6)
"""
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import occlude_state_blocks as OC  # noqa: E402

sys.path.append(OC.ORIG)
sys.path.append(HERE)

import networkx as nx  # noqa: E402
from network_env import NetworkEnv  # noqa: E402
from topology import EDGES, SOURCE, DESTINATION  # noqa: E402

N_ENV = int(os.environ.get("PIP_ENVS", "30"))
MAXH = int(os.environ.get("PIP_MAXH", "12"))
TOL = float(os.environ.get("PIP_TOL", "1e-6"))
HOPS = tuple(range(6, 11))  # 도착 틱 WARMUP+h 의 h
CLIP = 120.0
# 23.9.6.15 [1] 은 **네 기준 경로 전부**에 동일 비율을 찍었다. 특히
# `=hop` 이 0.0 -> 14.0% 로 오른 것이 "손상되면 최단홉으로 간다" 의
# 근거이고, 홉 6 경로는 **4개뿐**이라 거기 충돌이 있으면 그 주장이
# 흔들린다. 그래서 네 비용을 다 본다.
REFS = ("risk", "delay", "hop", "bandwidth")


def build_paths():
    """홉 -> 경로 리스트(노드열). 위상만의 함수라 한 번만 만든다."""
    g = nx.Graph()
    g.add_edges_from(EDGES)
    out = {}
    for pth in nx.all_simple_paths(g, SOURCE, DESTINATION, cutoff=MAXH):
        out.setdefault(len(pth) - 1, []).append(tuple(pth))
    return out


def main():
    OC.check_env()
    os.makedirs(OC.OUT_DIR, exist_ok=True)
    paths = build_paths()
    bar = "=" * 92
    print(bar)
    print("경로 동일 판정이 대리 판정인지 — 지연 충돌을 직접 센다")
    print("  env %d · 홉 상한 %d · 허용오차 %.1e" % (N_ENV, MAXH, TOL))
    print("  홉별 경로 수: %s"
          % ", ".join("h%d %d" % (h, len(paths[h]))
                      for h in sorted(paths)))
    print(bar)

    # 경로를 **정규화 엣지열**로 미리 바꿔 둔다(틱마다 재계산 안 하게).
    probe = NetworkEnv(use_risk=True, scenario=OC.SCENARIO,
                       risk_window=OC.WINDOW)
    probe.reset()
    ekeys = {}
    for h in sorted(paths):
        ekeys[h] = [tuple(probe.normalize_edge(p[i], p[i + 1])
                          for i in range(len(p) - 1))
                    for p in paths[h]]

    rows = []
    for tick_h in HOPS:
        tot_pairs = 0
        tot_paths = 0
        risk_hits = dict((ct, []) for ct in REFS)
        sat = []
        for ei in range(N_ENV):
            env = NetworkEnv(use_risk=True, scenario=OC.SCENARIO,
                             risk_window=OC.WINDOW)
            env.reset()
            for _ in range(OC.WARMUP + tick_h):
                env.update_dynamic_qos()
            qd = dict((e, env.qos[e]["delay"]) for e in env.qos)
            sat.append(sum(1 for e in qd if qd[e] >= CLIP - 1e-9))
            # 홉 계급별 충돌(판정과 무관한 전역 배경값)
            for h in sorted(ekeys):
                ds = sorted(sum(qd[e] for e in ek) for ek in ekeys[h])
                tot_paths += len(ds)
                tot_pairs += sum(1 for i in range(len(ds) - 1)
                                 if ds[i + 1] - ds[i] < TOL)
            # 네 기준 경로 각각: 같은 홉·같은 지연인 **다른** 경로 수
            for ct in REFS:
                rp = env.ospf_path(cost_type=ct)
                rh = len(rp) - 1
                rdel = sum(qd[env.normalize_edge(rp[i], rp[i + 1])]
                           for i in range(rh))
                if rh not in ekeys:
                    risk_hits[ct].append(-1)
                    continue
                k = 0
                for ek in ekeys[rh]:
                    if abs(sum(qd[e] for e in ek) - rdel) < TOL:
                        k += 1
                risk_hits[ct].append(k - 1)  # 자기 자신 제외
        print("")
        print("  도착 틱 WARMUP+%d" % tick_h)
        print("    포화 링크(>=%.0fms) 평균 %.2f / 50"
              % (CLIP, sum(sat) / len(sat)))
        print("    홉 계급 안 인접 충돌 쌍 %d (경로 %d 중) — 배경값"
              % (tot_pairs, tot_paths))
        for ct in REFS:
            ok = [v for v in risk_hits[ct] if v >= 0]
            print("    %-10s 같은 홉·같은 지연인 다른 경로: 평균 %.3f ·"
                  " 최대 %d · 0인 env %d/%d"
                  % (ct, sum(ok) / max(len(ok), 1),
                     max(ok) if ok else -1,
                     sum(1 for v in ok if v == 0), len(ok)))
            rows.append([tick_h, ct, N_ENV, MAXH, "%.1e" % TOL,
                         "%.3f" % (sum(sat) / len(sat)), tot_pairs,
                         tot_paths, "%.4f" % (sum(ok) / max(len(ok), 1)),
                         max(ok) if ok else -1,
                         sum(1 for v in ok if v == 0), len(ok)])

    path = os.path.join(OC.OUT_DIR, "path_identity_proxy.csv")
    with io.open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arrival_tick_h", "cost_type", "n_env",
                    "max_hop", "tol", "mean_saturated_links",
                    "collision_pairs", "paths_scored", "mean_extra_paths",
                    "max_extra_paths", "n_env_zero", "n_env_ok"])
        w.writerows(rows)
    print("")
    print("저장: %s" % path)
    print(bar)


if __name__ == "__main__":
    main()
