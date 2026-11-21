# -*- coding: utf-8 -*-
"""표 13(노드 이동성)을 Random Waypoint 로 채울 수 있는가 — 전제부터 측정.

원고 4.8절은 Random Waypoint, vmax 20~100 km/h 를 명시한다. 그런데 RWP 는
**노드 좌표 · 통신 반경 · 필드 크기 · 스텝 시간**을 요구하고, `topology.py` 는
그 넷 중 **아무것도 정의하지 않는다** — 25개 노드와 손으로 적은 간선 50개,
간선별 타입뿐이다(`EDGE_TYPES`). 프로젝트 안에서 좌표가 나오는 유일한 곳은
`animate_risk_dqn.py` 의 `nx.spring_layout` 이고 그건 **그림용**이다.

그래서 이 스크립트는 세 가지를 센다. 주장하지 않고 센다.

(1) 주어진 50간선 그래프가 **통신 반경 모형(unit-disk graph)으로 표현되는가.**
    좌표를 여러 방법으로 찍고 반경 R 을 훑어 실제 간선 집합과의 F1 을 최대화한다.
    F1 이 1.0 에 못 미치면, RWP 시뮬레이션은 논문의 토폴로지가 아닌 **다른 네트워크**를
    움직이는 것이 된다. (측정 결과는 F1 0.99 로 **거의 맞는다** — 이 항목은 통과다.)

(2) 그 좌표로 RWP 를 돌렸을 때 **상태 벡터가 표현할 수 없는 간선이 몇 개 생기는가.**
    상태의 위험 50칸은 모듈 전역 `EDGES` 순서에 묶여 있다(`network_env.get_state`).
    노드가 움직이면 원래 50개에 없는 간선이 생기는데 **담을 칸이 없다.**
    링크 소멸은 reeval_mobility_proxy.py 가 이미 다뤘지만(risk=1.0 으로 표시),
    **간선 생성**은 그 축소형도 다루지 못한다.

(3) 답이 **정하지 않은 자유 파라미터에 얼마나 끌려다니는가.** 좌표·필드·스텝시간을
    그럴듯한 범위에서 바꿔 같은 수치를 다시 낸다. 폭이 크면 표 13 의 칸은
    이동성의 성질이 아니라 **내가 고른 상수**를 보고하는 것이다.

출력: review_results/rwp_embedding_fit.csv, review_results/rwp_churn.csv
"""
import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, ORIG)

import networkx as nx                            # noqa: E402
import numpy as np                               # noqa: E402

from topology import EDGES, NUM_NODES, SOURCE, DESTINATION   # noqa: E402

OUT_DIR = os.path.join(HERE, "review_results")
TRUE_EDGES = frozenset(frozenset(e) for e in EDGES)
R_GRID = 400                 # 반경 훑기 해상도
PAIRS = [(i, j) for i in range(NUM_NODES) for j in range(i + 1, NUM_NODES)]
PI = np.array([i for i, _ in PAIRS])
PJ = np.array([j for _, j in PAIRS])
TRUTH = np.array([frozenset((i, j)) in TRUE_EDGES for i, j in PAIRS])
SEED = 20261114

# 원고가 주지 않아 내가 고르는 상수들 — (3) 에서 민감도를 잰다
FIELD_M = float(os.environ.get("FIELD_M", "1000"))    # 정사각 필드 한 변 (m)
DT_S = float(os.environ.get("DT_S", "1.0"))           # 한 스텝의 물리 시간 (s)
HORIZON = int(os.environ.get("HORIZON", "10"))        # 에피소드 스텝 수 (평균 홉 ~8)
EPISODES = int(os.environ.get("EPISODES", "200"))
VMAX_KMH = (20.0, 40.0, 60.0, 80.0, 100.0)


def true_graph():
    g = nx.Graph()
    g.add_nodes_from(range(NUM_NODES))
    g.add_edges_from(EDGES)
    return g


def normalize(pos):
    """좌표를 단위 정사각형으로 옮긴다 (반경 비교를 공정하게 하려고)."""
    P = np.array([pos[i] for i in range(NUM_NODES)], dtype=float)
    P -= P.min(axis=0)
    span = P.max(axis=0)
    span[span == 0] = 1.0
    P /= span.max()
    return P


def layered_layout():
    """주석의 계층 구분을 그대로 좌표로 — 손으로 만든 후보."""
    layers = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12],
              [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23], [24]]
    pos = {}
    for yi, row in enumerate(layers):
        for xi, n in enumerate(row):
            pos[n] = (xi / max(1, len(row) - 1), yi / (len(layers) - 1))
    return pos


def hop_distances():
    """노드쌍 홉 거리 행렬. unit-disk 후보의 목표 거리로 쓴다."""
    g = true_graph()
    D = np.zeros((NUM_NODES, NUM_NODES))
    sp = dict(nx.all_pairs_shortest_path_length(g))
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            D[i, j] = sp[i][j]
    return D


def stress_layout(seed, iters=600, lr=0.05):
    """SMACOF 류 응력 최소화 — scipy 없이. 유클리드 거리를 홉 거리에 맞춘다.

    scipy 가 이 환경에서 못 쓰이므로(numpy 2.4 와 불일치) kamada_kawai 대신
    쓴다. 통신 반경 모형의 전제가 "거리가 가까우면 연결"이므로, 홉 거리를
    유클리드로 재현하는 좌표가 unit-disk 후보로 가장 유리하다.
    """
    D = hop_distances()
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(NUM_NODES, 2))
    with np.errstate(divide="ignore"):
        W = np.where(D > 0, 1.0 / (D ** 2), 0.0)
    for _ in range(iters):
        diff = P[:, None, :] - P[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2)) + 1e-9
        grad = ((W * (dist - D) / dist)[:, :, None] * diff).sum(axis=1)
        P -= lr * grad
    return {i: tuple(P[i]) for i in range(NUM_NODES)}


def candidate_layouts():
    g = true_graph()
    out = [("layered", layered_layout())]
    for s in range(20):
        out.append(("spring%02d" % s, nx.spring_layout(g, seed=s)))
    for s in range(8):
        out.append(("stress%02d" % s, stress_layout(s)))
    return [(n, normalize(p)) for n, p in out]


def unit_disk_fit(P):
    """반경 R 을 훑어 실제 간선 집합과의 F1 최댓값을 찾는다."""
    d = np.hypot(P[PI, 0] - P[PJ, 0], P[PI, 1] - P[PJ, 1])
    truth = TRUTH
    n_true = int(truth.sum())
    best = None
    for R in np.linspace(d.min(), d.max(), R_GRID):
        pred = d <= R
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = n_true - tp
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if best is None or f1 > best[0]:
            best = (f1, float(R), tp, fp, fn, prec, rec)
    return best


def present_mask(P, R):
    """반경 R 안의 쌍 — 고정 쌍 테이블 위의 불리언 마스크."""
    d = np.hypot(P[PI, 0] - P[PJ, 0], P[PI, 1] - P[PJ, 1])
    return d <= R


def connected(mask, a, b):
    """BFS. 노드 25개라 인접목록을 매번 만들어도 싸다."""
    adj = [[] for _ in range(NUM_NODES)]
    for k in np.flatnonzero(mask):
        i, j = int(PI[k]), int(PJ[k])
        adj[i].append(j)
        adj[j].append(i)
    seen = {a}
    stack = [a]
    while stack:
        n = stack.pop()
        if n == b:
            return True, adj
        for m in adj[n]:
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return False, adj


def rwp_episode(P0, R, vmin_ms, vmax_ms, field, dt, horizon, rng):
    """Random Waypoint 한 에피소드. 반환: 스텝별 지표 리스트."""
    P = np.array(P0, dtype=float) * field
    R_m = R * field
    wp = np.array([[rng.uniform(0, field), rng.uniform(0, field)]
                   for _ in range(NUM_NODES)])
    spd = np.array([rng.uniform(vmin_ms, vmax_ms) for _ in range(NUM_NODES)])
    rows = []
    for _ in range(horizon):
        for n in range(NUM_NODES):
            d = wp[n] - P[n]
            dist = float(np.hypot(d[0], d[1]))
            step = spd[n] * dt
            if dist <= step or dist == 0.0:
                P[n] = wp[n]
                wp[n] = [rng.uniform(0, field), rng.uniform(0, field)]
                spd[n] = rng.uniform(vmin_ms, vmax_ms)
            else:
                P[n] = P[n] + d / dist * step
        mask = present_mask(P, R_m)
        kept = int((mask & TRUTH).sum())
        novel = int((mask & ~TRUTH).sum())
        conn, adj = connected(mask, SOURCE, DESTINATION)
        deg0 = sum(1 for a in adj if not a)
        rows.append({"kept": kept, "novel": novel, "deg0": deg0,
                     "conn": 1.0 if conn else 0.0,
                     "n_present": int(mask.sum())})
    return rows


def rwp_summary(P0, R, vmax_kmh, field=FIELD_M, dt=DT_S,
                horizon=HORIZON, episodes=EPISODES, seed=SEED):
    vmax = vmax_kmh / 3.6
    vmin = 20.0 / 3.6          # 원고 범위의 하단을 최저 속도로 고정
    acc = {"kept": [], "novel": [], "deg0": [], "conn": [], "n_present": []}
    for ep in range(episodes):
        rng = random.Random("%d|%d|%d|%d|%d" % (seed, ep, int(vmax_kmh),
                                                int(field), int(dt * 10)))
        for r in rwp_episode(P0, R, vmin, vmax, field, dt, horizon, rng):
            for k in acc:
                acc[k].append(r[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def mean_sd(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    g = true_graph()
    print("원 토폴로지: 노드 %d개, 간선 %d개 (좌표 없음)"
          % (g.number_of_nodes(), g.number_of_edges()))
    print("자유 파라미터 (원고가 주지 않음): 좌표 %d자유도, 통신반경, 필드 %gm, 스텝 %gs"
          % (2 * NUM_NODES, FIELD_M, DT_S))

    print("")
    print("=" * 76)
    print("[1] 50간선 그래프가 통신 반경 모형으로 표현되는가")
    print("-" * 76)
    print("%-11s %8s %9s %6s %6s %6s %8s %8s"
          % ("좌표", "최적F1", "반경", "맞춤", "허위", "누락", "정밀도", "재현율"))
    fits = []
    for name, P in candidate_layouts():
        f1, R, tp, fp, fn, prec, rec = unit_disk_fit(P)
        fits.append({"layout": name, "best_f1": round(f1, 6),
                     "best_radius": round(R, 6), "tp": tp, "fp": fp,
                     "fn": fn, "precision": round(prec, 6),
                     "recall": round(rec, 6), "_P": P})
    fits.sort(key=lambda r: -r["best_f1"])
    for r in fits:
        print("%-11s %8.4f %9.4f %6d %6d %6d %8.4f %8.4f"
              % (r["layout"], r["best_f1"], r["best_radius"], r["tp"],
                 r["fp"], r["fn"], r["precision"], r["recall"]))
    with open(os.path.join(OUT_DIR, "rwp_embedding_fit.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layout", "best_f1", "best_radius",
                                          "tp", "fp", "fn", "precision",
                                          "recall"])
        w.writeheader()
        for r in fits:
            w.writerow({k: v for k, v in r.items() if k != "_P"})
    f1s = [r["best_f1"] for r in fits]
    m, sd = mean_sd(f1s)
    print("")
    print("최고 F1 %.4f (%s), 후보 %d개 평균 %.4f ± %.4f, 최저 %.4f"
          % (fits[0]["best_f1"], fits[0]["layout"], len(fits), m, sd,
             fits[-1]["best_f1"]))
    b = fits[0]
    if b["best_f1"] >= 0.98 and b["fn"] == 0:
        print("→ 이 그래프는 **사실상 통신 반경 모형이다** — 최적 좌표에서 50간선을")
        print("  전부 재현하고 허위 %d개만 더한다(누락 0). 즉 좌표가 없다는 것이"
              % b["fp"])
        print("  RWP 를 막는 이유가 **아니다.** (이 줄은 반대 예상을 측정이 꺾은 것이다.)")
    else:
        print("→ 최적 좌표에서도 허위 %d개 · 누락 %d개가 남는다. RWP 로 움직이는 것은"
              % (b["fp"], b["fn"]))
        print("  논문의 토폴로지와 다른 그래프다.")
    return fits


TOPK = int(os.environ.get("TOPK", "5"))


def part2(fits):
    """RWP 를 돌려 '상태가 담을 수 없는 간선'과 도달 가능성을 센다."""
    print("")
    print("=" * 76)
    print("[2] RWP 를 돌리면 상태 50칸이 못 담는 간선이 생기는가 (상위 %d좌표)" % TOPK)
    print("-" * 76)
    print("%-11s %6s %9s %10s %10s %9s %9s"
          % ("좌표", "vmax", "원간선유지", "새간선(칸없음)", "현재간선수",
             "차수0노드", "S-D연결%"))
    rows = []
    for r in fits[:TOPK]:
        for v in VMAX_KMH:
            s = rwp_summary(r["_P"], r["best_radius"], v)
            rows.append({"layout": r["layout"], "best_f1": r["best_f1"],
                         "radius": r["best_radius"], "vmax_kmh": v,
                         "field_m": FIELD_M, "dt_s": DT_S,
                         "horizon": HORIZON, "episodes": EPISODES,
                         "kept_of_50": round(s["kept"], 4),
                         "kept_pct": round(100.0 * s["kept"] / 50.0, 4),
                         "novel_edges": round(s["novel"], 4),
                         "n_present": round(s["n_present"], 4),
                         "deg0_nodes": round(s["deg0"], 4),
                         "sd_connected_pct": round(100.0 * s["conn"], 4)})
            print("%-11s %6.0f %9.2f %10.2f %10.2f %9.2f %9.2f"
                  % (r["layout"], v, s["kept"], s["novel"], s["n_present"],
                     s["deg0"], 100.0 * s["conn"]))
    with open(os.path.join(OUT_DIR, "rwp_churn.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for x in rows:
            w.writerow(x)
    print("")
    nv = [x["novel_edges"] for x in rows]
    kp = [x["kept_pct"] for x in rows]
    print("새 간선 %.1f~%.1f개 (상태에 담을 칸이 없다), 원 간선 유지 %.1f~%.1f%%"
          % (min(nv), max(nv), min(kp), max(kp)))
    return rows


def part3(fits):
    """답이 내가 고른 상수에 얼마나 끌려다니는가."""
    print("")
    print("=" * 76)
    print("[3] 정하지 않은 상수를 바꾸면 같은 칸이 얼마나 움직이는가 (vmax=60 고정)")
    print("-" * 76)
    print("%-11s %8s %7s %10s %12s %10s"
          % ("좌표", "필드m", "스텝s", "원간선유지", "새간선", "S-D연결%"))
    base = fits[0]
    out = []
    for field in (500.0, 1000.0, 2000.0):
        for dt in (0.5, 1.0, 2.0):
            s = rwp_summary(base["_P"], base["best_radius"], 60.0,
                            field=field, dt=dt)
            out.append((field, dt, s))
            print("%-11s %8.0f %7.1f %10.2f %12.2f %10.2f"
                  % (base["layout"], field, dt, s["kept"], s["novel"],
                     100.0 * s["conn"]))
    kept = [s["kept"] for _, _, s in out]
    conn = [100.0 * s["conn"] for _, _, s in out]
    print("")
    print("원 간선 유지 %.2f~%.2f개 (%.1f배), S-D 연결 %.1f~%.1f%%"
          % (min(kept), max(kept),
             (max(kept) / min(kept)) if min(kept) > 0 else float("inf"),
             min(conn), max(conn)))
    print("→ 폭이 크면 표 13 의 칸은 이동성이 아니라 이 상수들을 보고하는 것이다.")

    print("")
    print("좌표 선택만으로도 얼마나 갈리는가 (vmax=60, 필드·스텝 기본값)")
    spread = []
    for r in fits[:TOPK]:
        s = rwp_summary(r["_P"], r["best_radius"], 60.0)
        spread.append((r["layout"], r["best_f1"], s["kept"], s["novel"],
                       100.0 * s["conn"]))
    for nm, f1, k, nv, c in spread:
        print("  %-11s F1=%.4f  원간선유지 %.2f  새간선 %.2f  S-D연결 %.1f%%"
              % (nm, f1, k, nv, c))
    ks = [x[2] for x in spread]
    print("  동등하게 잘 맞는 좌표들 사이에서 원 간선 유지가 %.2f~%.2f개로 갈린다."
          % (min(ks), max(ks)))

    # 이 두 블록은 출력만 하고 CSV 를 안 써서 초안이 인용한 26.3~48.9 /
    # 3.4~47.3 이 출처 없는 값이 돼 있었다(감사 32절 산출물 미보존).
    path = os.path.join(OUT_DIR, "rwp_sensitivity.csv")
    cols = ["block", "layout", "field_m", "dt_s", "vmax", "best_f1",
            "kept", "novel", "conn_pct"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for field, dt, s in out:
            w.writerow(dict(block="field_dt_grid", layout=base["layout"],
                            field_m="%.1f" % field, dt_s="%.1f" % dt,
                            vmax="60.0", best_f1="%.6f" % base["best_f1"],
                            kept="%.6f" % s["kept"],
                            novel="%.6f" % s["novel"],
                            conn_pct="%.6f" % (100.0 * s["conn"])))
        for nm, f1, k, nv, c in spread:
            w.writerow(dict(block="layout_spread", layout=nm,
                            field_m="", dt_s="", vmax="60.0",
                            best_f1="%.6f" % f1, kept="%.6f" % k,
                            novel="%.6f" % nv, conn_pct="%.6f" % c))
    nov = [s["novel"] for _, _, s in out]
    print("  저장: %s (%d행)  원간선유지 %.2f~%.2f (%.1f배) · "
          "새간선 %.2f~%.2f (%.1f배)"
          % (path, len(out) + len(spread), min(kept), max(kept),
             (max(kept) / min(kept)) if min(kept) > 0 else float("inf"),
             min(nov), max(nov),
             (max(nov) / min(nov)) if min(nov) > 0 else float("inf")))


if __name__ == "__main__":
    sys.path.append(HERE)
    FITS = main()
    part2(FITS)
    part3(FITS)
    print("")
    print("쓴 파일: review_results/rwp_embedding_fit.csv, review_results/rwp_churn.csv")
