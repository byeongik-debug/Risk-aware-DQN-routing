# -*- coding: utf-8 -*-
"""22.5절 '팔 라벨이 정책을 가린다' 의 유일한 생존 결과가 Holm 족 크기에 견디는가.

왜 재는가
---------
`cluster_policy_classes.py` 의 팔별 순열검정에서 살아남은 것은 `factor_util`
하나뿐이고 기록된 값은 `p_raw = 0.0005, p_holm = 0.0065`(족 13) 다. 그런데
그 0.0005 는 **측정값이 아니라 분해능 바닥**이다 — B=2000 에서 p 의 최솟값이
(0+1)/(2000+1) = 0.00049975 이기 때문이다. 즉 "0.0065" 는 B 가 작아서 거기
멈춘 상한이고, 실제 p 는 더 작을 수 있다.

다른 자리에서 Holm 족 크기가 소속을 두 번 뒤집은 적이 있으므로
(족 18→30 에서 `pow05n` 역전, 족 12 에서 도달률 10/12 가 원리적 불가),
이 결과도 "족을 더 크게 잡으면 깨지는가" 를 숫자로 답해 둬야 한다.

방법
----
`cluster_policy_classes.main()` 을 그대로 호출해 **같은 배터리·같은 행동
벡터**를 받고(시드 20261114 고정), 거기서 나온 합치행렬만 numpy 로 다시
돌린다. 귀무가설은 원본과 같다 — 크기 k 의 부분집합을 무작위로 뽑아
(내부 평균 − 외부 평균)의 분포를 만든다. 원본은 파이썬 `random.sample`,
여기서는 numpy 라서 몬테카를로 오차만큼 다르다(그래서 B=2000 재현도 같이 찍는다).

족 크기 한계는 Holm 의 첫 단계에서 나온다: 가장 작은 p 는 `p * F < 0.05`
일 때만 유의하므로, 유의성을 유지하는 최대 족 크기는 `ceil(0.05/p) - 1` 이다.

원본 `SDN_DQN_new_env/` 는 읽기만 한다. 출력:
  review_results/policy_perm_family.csv
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# cluster_policy_classes 가 import 시점에 sys.path 에서 HERE 를 지우므로
# 이쪽 모듈을 먼저 집어넣는다.
import review_stats as RS                        # noqa: E402

import numpy as np                               # noqa: E402

import cluster_policy_classes as CPC              # noqa: E402

B = int(os.environ.get("PERMS", "200000"))
B_REPRO = int(os.environ.get("PERMS_REPRO", "2000"))
CHUNK = int(os.environ.get("CHUNK", "20000"))
SEED = 20261114
OUT = os.path.join(HERE, "review_results", "policy_perm_family.csv")


def agreement_matrix(vectors, items, kind, n_states):
    """팔 순서를 items 그대로 유지한 m x m 합치율(%) 행렬."""
    vecs = [np.asarray(vectors[(kind, a, r)], dtype=np.int16)
            for a, r, _ in items]
    m = len(vecs)
    V = np.vstack(vecs)                      # (m, n_states)
    A = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        eq = (V[i][None, :] == V).sum(axis=1) / float(n_states) * 100.0
        A[i] = eq
    np.fill_diagonal(A, 0.0)
    return A


def observed_d(A, idx, m):
    """내부 평균 − 외부 평균 (원본 report() 와 같은 정의)."""
    k = len(idx)
    sub = A[np.ix_(idx, idx)]
    win_sum = sub.sum() / 2.0
    win_mean = win_sum / (k * (k - 1) / 2.0)
    rowsum = A[idx].sum()
    cross_sum = rowsum - 2.0 * win_sum
    cross_mean = cross_sum / float(k * (m - k))
    return win_mean - cross_mean


def null_distribution(A, k, m, n_draws, rng):
    """크기 k 부분집합을 무작위로 뽑은 (내부−외부) 분포. 청크로 벡터화."""
    rowsums = A.sum(axis=1)
    out = np.empty(n_draws, dtype=np.float64)
    done = 0
    win_den = k * (k - 1) / 2.0
    cross_den = float(k * (m - k))
    while done < n_draws:
        c = min(CHUNK, n_draws - done)
        # 비복원 추출 c개: 난수 행렬의 argsort 앞 k개
        pick = np.argsort(rng.random((c, m)), axis=1)[:, :k]
        sub = A[pick[:, :, None], pick[:, None, :]]          # (c, k, k)
        win_sum = sub.sum(axis=(1, 2)) / 2.0
        rs = rowsums[pick].sum(axis=1)
        out[done:done + c] = (win_sum / win_den
                              - (rs - 2.0 * win_sum) / cross_den)
        done += c
    return out


def main():
    batteries, vectors, items = CPC.main()
    labels = [a for a, _, _ in items]
    arms = sorted(set(labels))
    m = len(items)
    rows = []

    for kind in sorted(batteries):
        n_states = len(batteries[kind])
        A = agreement_matrix(vectors, items, kind, n_states)
        print("")
        print("=== 묶음 %s (상태 %d, 체크포인트 %d) ===" % (kind, n_states, m))

        per_arm = []
        for arm in arms:
            idx = [i for i, a in enumerate(labels) if a == arm]
            if len(idx) < 2:
                continue
            k = len(idx)
            d = observed_d(A, np.asarray(idx), m)

            rng = np.random.default_rng(SEED + 1)
            null_r = null_distribution(A, k, m, B_REPRO, rng)
            p_repro = (int((null_r >= d).sum()) + 1) / float(B_REPRO + 1)

            rng = np.random.default_rng(SEED + 2)
            null = null_distribution(A, k, m, B, rng)
            hits = int((null >= d).sum())
            p_raw = (hits + 1) / float(B + 1)
            thresh = float(np.quantile(null, 0.95))
            per_arm.append([arm, k, d, hits, p_raw, p_repro, thresh])

        p_holm = RS.holm_bonferroni([r[4] for r in per_arm])
        fam = len(per_arm)
        print("    팔 %d개, B=%d (바닥 p=%.2e), 재현 B=%d"
              % (fam, B, 1.0 / (B + 1), B_REPRO))
        print("    %-18s %2s %9s %7s %10s %10s %10s %8s"
              % ("arm", "n", "내부우위", "초과", "p_raw", "p_holm", "p(B=2k)",
                 "족한계"))
        for r, ph in zip(per_arm, p_holm):
            arm, k, d, hits, p_raw, p_repro, thresh = r
            f_break = int(np.ceil(0.05 / p_raw)) - 1
            mark = "유의" if ph < 0.05 else ""
            print("    %-18s %2d %+8.3f%%p %7d %10.6f %10.6f %10.6f %8d %s"
                  % (arm, k, d, hits, p_raw, ph, p_repro, f_break, mark))
            rows.append({"battery": kind, "arm": arm, "n": k,
                         "inner_advantage_pp": "%.6f" % d,
                         "perms": B, "exceed_count": hits,
                         "p_raw": "%.8f" % p_raw,
                         "p_holm_family%d" % fam: "%.8f" % ph,
                         "p_raw_B%d" % B_REPRO: "%.6f" % p_repro,
                         "thresh95_pp": "%.6f" % thresh,
                         "max_holm_family_still_sig": f_break})

    fields = ["battery", "arm", "n", "inner_advantage_pp", "perms",
              "exceed_count", "p_raw", "p_holm_family%d" % 13,
              "p_raw_B%d" % B_REPRO, "thresh95_pp",
              "max_holm_family_still_sig"]
    # 족 크기가 13이 아닐 수도 있으니 실제 키로 맞춘다
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    fields = [f for f in fields if f in keys] + sorted(keys - set(fields))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("")
    print("저장: " + OUT)


if __name__ == "__main__":
    main()
