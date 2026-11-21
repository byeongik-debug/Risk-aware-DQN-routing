# -*- coding: utf-8 -*-
"""학습한 W-Current 팔을 Full 팔과 맞춰 본다 — 표 10 W-Current 행의 차용 검사.

배경
----
표 10 가중치 비교군의 원천 `table10_extra_runs.csv` 에는 `W-Equal`·
`W-Temporal` 만 있고 **W-Current 행에 해당하는 팔이 없다**
(`weight_current_*.pth` 도 0개). 그 행의 숫자는 기본 가중치로 학습한 다른
팔에서 가져온 것이다. 차용이 정당한지 재려면 "W-Current 조건 == Full 조건"
이어야 한다.

하네스가 이미 증언하는 것
-------------------------
`eval_table10_extra.probe_weights` 는 가중치 몽키패치가 **죽지 않았는지**
확인하려고 `abs(got - base) > 1e-9` 를 단정한다. `"current"` 를 넣으면 이
단정이 **실패한다** — 기본 가중치와 구별되지 않기 때문이다. 그래서
`X.WEIGHTS` 에는 `"equal"`·`"temporal"` 만 있다. 즉 평가 하네스 쪽에서
W-Current 는 `wkey=None` 과 같은 조건이고, 이 스크립트는 양쪽 모델을 모두
`wkey=None` 으로 굴린다(그게 맞는 조건이다).

재는 것 네 가지
---------------
(ㄱ) **위험함수 수치 동일성** — `CUSTOM_WEIGHTS=None` 과
     `=WEIGHT_CONFIGS["current"]` 로 같은 입력에 위험값을 계산해 차이를
     본다. 0.0 이어야 한다(같은 리터럴, `w["util"]` 처럼 키로 직접 접근).
(ㄴ) **체크포인트 비트 비교** — 파일 md5 와 텐서별 최대 절대차.
     같으면 결정적. **다르면 애매하다**: Full 을 학습한 당시의 torch 스레드
     수를 모르고, 스레드 수는 부동소수 축약 순서를 바꾼다.
(ㄷ) **200에피소드 경로 동일성** — 스레드 수·부동소수 잡음에 둔감하므로
     **이쪽이 판정**이다. 경로가 전부 같으면 정책이 같다.
(ㄹ) **네 지표** — Full 쪽은 `table10_factor_runs.csv` 기록값과 자기검증한다.
     어긋나면 롤아웃 절차가 기록 당시와 달라졌다는 뜻이므로 비교 자체가
     무효다.

쓰는 법
-------
    python compare_wcurrent.py                 # 200 에피소드
    python compare_wcurrent.py --episodes 20   # 빨리 확인만
"""
import argparse
import csv
import hashlib
import io
import os

import torch

import check_path_identity as C
import eval_table10_extra as X                # noqa: F401  (하네스 증언용)
import train_weight_compare as W

RESULT_DIR = C.RESULT_DIR
WC_DIR = os.path.join(RESULT_DIR, "wcurrent", "models")
OUT = os.path.join(RESULT_DIR, "wcurrent_compare.csv")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def risk_identity():
    """(ㄱ) current 가중치와 기본 가중치가 같은 위험값을 내는가."""
    rc = C.risk_calculator
    rc.ZEROED_FACTORS = []
    rc.CUSTOM_WEIGHTS = None
    base = rc.calculate_link_risk(**X.PROBE)
    rc.CUSTOM_WEIGHTS = dict(W.WEIGHT_CONFIGS["current"])
    got = rc.calculate_link_risk(**X.PROBE)
    rc.CUSTOM_WEIGHTS = None
    return base, got


def full_dir():
    """표 10 이 실제로 쓴 Full 모델 디렉터리를 CSV 에서 읽는다."""
    p = os.path.join(RESULT_DIR, "table10_factor_runs.csv")
    with io.open(p, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["arm"] == "Full":
                return row["model_dir"]
    raise SystemExit("table10_factor_runs.csv 에 Full 행이 없다")


def tensor_diff(pa, pb):
    """두 체크포인트의 텐서별 최대 절대차와 키 불일치."""
    a = torch.load(pa, map_location="cpu", weights_only=True)
    b = torch.load(pb, map_location="cpu", weights_only=True)
    if sorted(a.keys()) != sorted(b.keys()):
        return None, sorted(set(a) ^ set(b))
    m = 0.0
    for k in a:
        d = (a[k].double() - b[k].double()).abs().max().item()
        if d > m:
            m = d
    return m, []


def agree(pa, pb):
    """에피소드별 경로가 같은 비율(%). 길이가 다르면 측정 결함이다."""
    assert len(pa) == len(pb), "에피소드 수가 다르다 %d/%d" % (len(pa), len(pb))
    same = sum(1 for x, y in zip(pa, pb) if x == y)
    return 100.0 * same / len(pa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    a = ap.parse_args()

    base, got = risk_identity()
    print("[ㄱ] 위험함수  기본 %.17g / current %.17g   차이 %.3g"
          % (base, got, abs(got - base)))
    print("     -> %s" % ("수치 동일(비트)" if got == base else "다르다!!"))
    print("     (하네스의 probe_weights 는 'current' 를 몽키패치 사망으로"
          " 판정한다 — 기본과 구별이 안 되기 때문이다.)")

    fd = full_dir()
    print("\nFull 모델 디렉터리(표 10 이 쓴 것): %s" % fd)

    rw = C.T.RISK_WINDOW
    rec = C.load_recorded()
    rows = []
    for run in range(1, 11):
        wp = os.path.join(WC_DIR, "weight_current_run%02d.pth" % run)
        fp = os.path.join(fd, "factor_full_run%02d.pth" % run)
        if not os.path.exists(wp):
            continue
        if not os.path.exists(fp):
            print("  run%02d: Full 쪽 모델이 없다 — 건너뛴다" % run)
            continue

        hw, hf = md5(wp), md5(fp)
        td, keydiff = tensor_diff(wp, fp)
        print("\n[ㄴ] run%02d 체크포인트" % run)
        print("     W-Current md5 %s" % hw)
        print("     Full      md5 %s" % hf)
        if keydiff:
            print("     !! 키가 다르다: %s" % keydiff)
        print("     파일 동일? %s   텐서 최대차 %.3g"
              % (hw == hf, (td if td is not None else float("nan"))))

        print("[ㄷ][ㄹ] 롤아웃 %d 에피소드 (양쪽 wkey=None)" % a.episodes)
        pw, mw = C.rollout_paths(wp, rw, None, [], a.episodes)
        pf, mf = C.rollout_paths(fp, rw, None, [], a.episodes)
        ag = agree(pw, pf)
        print("     경로 일치 %.2f%%   서로 다른 경로 W-Current %d / Full %d"
              % (ag, len(set(map(tuple, pw))), len(set(map(tuple, pf)))))
        print("     W-Current  지연 %.4f 손실 %.6f 홉 %.4f Risky %.4f"
              % (mw["delay"], mw["loss"], mw["hop"], mw["risky_pct"]))
        print("     Full       지연 %.4f 손실 %.6f 홉 %.4f Risky %.4f"
              % (mf["delay"], mf["loss"], mf["hop"], mf["risky_pct"]))

        # Full 쪽 자기검증: 기록 CSV 와 맞는가
        r = rec.get(("Full", run))
        sc = ""
        if r is not None and a.episodes == 200:
            d = abs(float(r["delay"]) - mf["delay"])
            sc = ("일치(%.2g)" % d) if d < 1e-4 else "어긋남(%.3g)" % d
            print("     자기검증(Full 대 기록 CSV): %s" % sc)

        rows.append(dict(run=run, episodes=a.episodes,
                         md5_wcurrent=hw, md5_full=hf,
                         file_same=int(hw == hf),
                         tensor_maxdiff=("%.6g" % td) if td is not None else "",
                         path_agree_pct="%.4f" % ag,
                         n_paths_wcurrent=len(set(map(tuple, pw))),
                         n_paths_full=len(set(map(tuple, pf))),
                         delay_wcurrent="%.6f" % mw["delay"],
                         delay_full="%.6f" % mf["delay"],
                         loss_wcurrent="%.6f" % mw["loss"],
                         loss_full="%.6f" % mf["loss"],
                         hop_wcurrent="%.4f" % mw["hop"],
                         hop_full="%.4f" % mf["hop"],
                         risky_wcurrent="%.4f" % mw["risky_pct"],
                         risky_full="%.4f" % mf["risky_pct"],
                         selfcheck_full=sc,
                         risk_probe_same=int(got == base)))

    if not rows:
        print("\n비교할 W-Current 모델이 아직 없다 (%s)" % WC_DIR)
        return

    flds = list(rows[0].keys())
    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)
    print("\n-> %s (%d행)" % (OUT, len(rows)))

    ags = [float(r["path_agree_pct"]) for r in rows]
    print("\n[판정] 경로 일치율 %s" % ", ".join("%.1f%%" % v for v in ags))
    if all(v >= 99.9999 for v in ags):
        print("  경로가 전부 같다 -> 두 조건이 같은 정책을 낸다. 표 10 의"
              " W-Current 행을 Full 에서 차용한 것은 **수치적으로 정당**하다.")
        print("  단, 각주에 출처는 적어야 한다: 그 숫자는 가중치 비교 실험에서"
              " 나온 것이 아니다.")
    else:
        print("  경로가 갈린다 -> 차용을 정당화할 수 없다. W-Current 5런을"
              " 전부 학습해 표 10 행을 **실측으로 교체**해야 한다.")


if __name__ == "__main__":
    main()
