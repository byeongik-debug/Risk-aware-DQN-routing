# -*- coding: utf-8 -*-
"""초안에 적힌 숫자가 커밋된 측정 파일에서 오는지 전수 대조한다.

왜 재는가
---------
이 작업에서 같은 종류의 결함을 세 번 잡았다.
  - 표 11 `NoRisk 427.2` — 커밋된 어느 CSV 로도 재현되지 않았다(실제 332.86).
  - `제안 350.1 ± 1.5` — 값은 맞지만 **다른 표**(발표본 표 6)의 행이었다.
  - `-18.4%` — 발표본 PDF 어디에도 인쇄돼 있지 않았다.
셋 다 "초안에 적힌 숫자가 어느 커밋 파일에서 왔는지" 를 확인하지 않아서
생겼다. 사람 눈으로 세 번 잡았으니 나머지는 기계로 센다.

검사 둘
-------
[1] 값 대조. `DRAFT_*.md` 의 숫자 토큰이
      (A) 집계 CSV  : review_results/*.csv 중 200KB 이하 (143개, 1.9MB)
      (B) 원시 로그  : 200KB 초과 (6개, 15.6MB · 에피소드 단위)
      (C) 기록 문서  : FINDINGS*.md + RESUME_HERE.md
    안의 어떤 수치로 **반올림되는지** 본다(초안 350.1 ↔ CSV 350.124972).
    A 에서 맞으면 출처가 있다. A 는 없고 B 만 맞으면 **출처로 치지 않는다**
    (수백만 개 중 우연히 같은 값일 수 있다).

    **C 만 맞는 것도 출처로 치지 않는다.** C 는 내가 쓰는 문서이므로 어떤
    숫자를 거기에 적기만 하면 그 숫자가 NO_SOURCE 에서 벗어나는 **세탁
    경로**가 생긴다. 추측이 아니다 — 32절을 작성하자 RAWLOG_ONLY 3건이
    그날 DOC_ONLY 로 바뀌고 경보 목록에서 사라졌다(1109/75/3 ->
    1110/77/0). 그래서 "계산해 적어 둔 값일 수 있다" 는 이유로 줬던 면제를
    해제하고 경보에 올린다. 정당한 유도값은 면제로 통과시키지 않고
    `verify_table_cells.py` 의 수식 재계산으로 갚는다. 셋 다 없으면 출처 불명.

    반올림 동점 주의: 원본이 정확히 57.555 면 초안의 half-up 57.56 과
    파이썬 `%.2f` 의 57.55 가 갈린다. 둘 다 맞다고 센 뒤 `AGG_CSV_TIE` 로
    구분해 표시한다(실제로 표 14 에서 한 번 걸렸다).

[2] 표 행 교차 출처 — **작동하지 않는다. 기본 비활성이다**(`--rowcheck`).
    의도는 [1] 이 못 잡는 것(값은 맞지만 **다른 표**에서 온 칸, 350.1 사건)을
    한 행의 칸들이 공유하는 출처 파일의 교집합으로 잡는 것이었다. 그런데
    표 칸은 대개 **런 CSV 의 평균**이라 어느 파일에도 그 문자열로 존재하지
    않는다 — 그래서 교집합이 **구조적으로** 빈다. 실제로 이 검사가 띄운 가림
    배터리 18칸을 손으로 다시 집계해 보니 전부 맞았다(거짓 경보 100%).
    남겨 둔 것은 기록용이고, 이 일을 제대로 하는 것은 따로 만든
    `verify_table_cells.py` 다(행·열을 선언하고 런 CSV 에서 재집계).

원본 `SDN_DQN_new_env/`, `Risk-aware-DQN-routing/` 는 열지 않는다. 출력:
  review_results/draft_number_provenance.csv
"""
import csv
import glob
import io
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "review_results")
OUT = os.path.join(RES, "draft_number_provenance.csv")
BIG = 200 * 1024
MAXROWS = 40

# 섹션/표 번호, 쪽·행 번호는 claim 이 아니다.
NUM = re.compile(r"(?<![\d.])(\d{1,7}\.\d{1,8})(?![\d.])")
REF_AFTER = re.compile(u"^(절|장|쪽|행|번|항)")
REF_BEFORE = re.compile(u"(FINDINGS[_A-Za-z0-9]*\\s*|§|절\\s*|표\\s*|그림\\s*|"
                        u"항목\\s*|RESUME[_A-Za-z]*\\s*|py:|:)$")
OLD_FLAGS = [u"~~", u"철회", u"인쇄본", u"발표본", u"구버전", u"대조표",
             u"금지", u"폐기", u"원고 재현", u"잘못", u"오기"]
ANYNUM = re.compile(r"\d+\.\d+|\d+")


def numeric_tokens(text):
    """참조번호로 보이는 것을 뺀 숫자 토큰 목록."""
    out = []
    for m in NUM.finditer(text):
        tok = m.group(1)
        before = text[max(0, m.start() - 12):m.start()]
        after = text[m.end():m.end() + 2]
        if REF_AFTER.match(after) or REF_BEFORE.search(before):
            continue
        out.append(tok)
    return out


def tie_key(s, d):
    """s 가 소수 d+1 째에서 정확한 동점이면 half-up 포맷을, 아니면 None."""
    if "." not in s:
        return None
    frac = s.split(".", 1)[1]
    if len(frac) <= d or frac[d] != "5" or any(c != "0" for c in frac[d + 1:]):
        return None
    q = Decimal(1).scaleb(-d)
    return str(Decimal(s).quantize(q, rounding=ROUND_HALF_UP))


def corpus_round_hits(paths, want):
    """want: {소수자릿수: {포맷문자열}} → (정상적중, 동점적중) 각각
    {포맷문자열: set(파일명)}."""
    hits = {}
    ties = {}
    decs = sorted(want)
    for p in paths:
        name = os.path.basename(p)
        try:
            txt = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception as exc:                       # noqa: BLE001
            sys.stderr.write("skip %s (%s)\n" % (name, exc))
            continue
        for m in ANYNUM.finditer(txt):
            s = m.group(0)
            try:
                v = float(s)
            except ValueError:
                continue
            av = abs(v)
            for d in decs:
                key = "%.*f" % (d, av)
                if key in want[d]:
                    hits.setdefault(key, set()).add(name)
                    continue
                tk = tie_key(s.lstrip("+-"), d)
                if tk is not None and tk in want[d]:
                    ties.setdefault(tk, set()).add(name)
    return hits, ties


def digest_report(drafts, agg=()):
    u"""REVISION_TABLE.md 전사 검사.

    이 문서는 공저자에게 건너가는 요약이고, **초안에서 숫자를 옮겨 적은 것**
    이다. 그래서 출처 말뭉치(CSV)로 대조하면 안 된다 — 초안이 산문으로 적은
    값은 CSV 문자열로 안 닿는 것이 정상이고(그건 [1] 이 칸 재집계로 갚는다),
    여기서 물어야 하는 것은 **옮겨 적다가 틀렸는가**다. 그래서 말뭉치를
    `DRAFT_*.md` 로 두고 같은 반올림 규칙으로 대조한다.
    """
    p = os.path.join(HERE, "REVISION_TABLE.md")
    if not os.path.exists(p):
        return 0
    rows, want = [], {}
    for i, line in enumerate(io.open(p, encoding="utf-8").read().split("\n")):
        for tok in numeric_tokens(line):
            # **반올림 결함 쌍**만 제외한다: `360.77→360.76` 처럼 화살표가
            # 공백 없이 붙고, 양쪽 소수자릿수가 같고, 끝자리만 1 다른 형태.
            # 이 왼쪽 값은 내가 고쳐 **지운 값**이라 초안에 없는 것이 정상이다.
            # 그 외의 화살표(`r 0.54→0.80`, `−18.4% → +5.2%`)는 초안이 실제로
            # 쓰는 값이므로 그대로 검사한다 — 화살표 전체를 면제하면
            # 전사 검사에 구멍이 난다.
            m = re.search(re.escape(tok) + u"→[-+−]?([0-9]+\\.[0-9]+)",
                          line)
            if m:
                rhs = m.group(1)
                if (len(rhs.split(".")[1]) == len(tok.split(".")[1])
                        and abs(round((float(tok) - float(rhs))
                                      * 10 ** len(rhs.split(".")[1]))) == 1):
                    continue
            d = len(tok.split(".", 1)[1])
            key = "%.*f" % (d, abs(float(tok)))
            want.setdefault(d, set()).add(key)
            rows.append((i + 1, tok, d, key, line.strip()))
    hits, ties = corpus_round_hits(drafts, want)
    miss = [r for r in rows if r[3] not in hits and r[3] not in ties]
    # 초안이 **반올림한 값**만 적어서 원본 정밀도가 초안에 없는 경우가 있다
    # (표 14 `57.56` 의 원본 57.555). 이건 전사 오류가 아니라 초안보다 더
    # 정밀한 인용이므로, 남은 것만 집계 CSV 로 한 번 더 받는다.
    src = {}
    if miss and agg:
        w2 = {}
        for _ln, _tok, d, k, _line in miss:
            w2.setdefault(d, set()).add(k)
        h2, t2 = corpus_round_hits(agg, w2)
        for k in list(h2) + list(t2):
            src[k] = sorted(h2.get(k, ())) or sorted(t2.get(k, ()))
        miss = [r for r in miss if r[3] not in src]
    print()
    print(u"=== [3] REVISION_TABLE.md 전사 검사 (말뭉치 = 초안) ===")
    print(u"  숫자 %d개 · 서로 다른 값 %d개 · 초안에서 못 찾은 것 %d개"
          % (len(rows), sum(len(v) for v in want.values()), len(miss)))
    if src:
        print(u"  (초안엔 없고 집계 CSV 로 받은 원본 정밀도 %d개: %s)"
              % (len(src), u" · ".join(u"%s←%s" % (k, v[0])
                                       for k, v in sorted(src.items()))))
    for ln, tok, _d, _k, line in miss[:MAXROWS]:
        print(u"  L%-5d %-12s %s" % (ln, tok, line[:90]))
    return len(miss)


def cell_coverage():
    """verify_table_cells.py 가 재집계로 맞춘 (초안, 줄) -> 인쇄값 목록.

    **말뭉치가 아니다.** 값 대조에는 쓰지 않는다 — 그렇게 쓰면 초안 숫자가
    자기 자신과 맞아 순환이 된다(실제로 한 번 걸렸다). 여기서는 어느 줄의
    어느 인쇄값이 런 CSV 재집계로 확인됐는지만 가져온다.
    """
    p = os.path.join(RES, "table_cell_verification.csv")
    cov = {}
    if not os.path.exists(p):
        return cov
    for r in csv.DictReader(io.open(p, encoding="utf-8-sig")):
        if r.get("counter") == "1" or r.get("match") != "1":
            continue
        key = (r["draft"], int(r["draft_line"]))
        for k in ("printed_mean", "printed_sd"):
            v = (r.get(k) or "").strip()
            if v:
                try:
                    cov.setdefault(key, []).append(abs(float(v)))
                except ValueError:
                    pass
    return cov


def main():
    drafts = sorted(glob.glob(os.path.join(HERE, "DRAFT_*.md")))
    claims = []          # (draft, lineno, token, dec, old_ctx, line, is_table)
    for p in drafts:
        name = os.path.basename(p)
        for i, line in enumerate(
                io.open(p, encoding="utf-8").read().split(u"\n"), 1):
            old = any(f in line for f in OLD_FLAGS)
            tbl = line.lstrip().startswith(u"|")
            for tok in numeric_tokens(line):
                dec = len(tok.split(".")[1])
                claims.append((name, i, tok, dec, old, line.strip(), tbl))

    want = {}
    for _, _, tok, dec, _, _, _ in claims:
        want.setdefault(dec, set()).add("%.*f" % (dec, float(tok)))
    uniq = sorted(set(t for _, _, t, _, _, _, _ in claims), key=float)
    print(u"초안 %d개 · 숫자 주장 %d건 · 서로 다른 값 %d개 · 소수자릿수 %s"
          % (len(drafts), len(claims), len(uniq), sorted(want)))

    # 자기 산출물을 말뭉치에 넣으면 안 된다. table_cell_verification.csv 는
    # 초안의 **인쇄값**을 printed_mean 열에 그대로 담고 있어서, 넣는 순간
    # 모든 표 칸이 "출처 있음" 으로 바뀌어 NO_SOURCE 가 가려진다(실제로
    # 2026-11-16 에 AGG_CSV 1109 -> 1114 로 부풀었다).
    SELF = set(os.path.basename(OUT).split()
               + ["draft_number_provenance.csv",
                  "table_cell_verification.csv"])
    allcsv = [p for p in sorted(glob.glob(os.path.join(RES, "*.csv")))
              if os.path.basename(p) not in SELF]
    agg = [p for p in allcsv if os.path.getsize(p) <= BIG]
    raw = [p for p in allcsv if os.path.getsize(p) > BIG]
    docs = (sorted(glob.glob(os.path.join(HERE, "FINDINGS*.md")))
            + [os.path.join(HERE, "RESUME_HERE.md")])
    COV = cell_coverage()
    print(u"칸 재집계 색인: 초안 %d줄 (말뭉치 아님 — 값 대조에 쓰지 않는다)"
          % len(COV))
    print(u"대조 말뭉치: 집계 CSV %d · 원시 로그 %d · 기록 문서 %d"
          % (len(agg), len(raw), len(docs)))

    hit_a, tie_a = corpus_round_hits(agg, want)
    print(u"  A 집계 CSV 끝: %d/%d 일치 (+ 동점 %d)"
          % (len(hit_a), len(uniq), len(tie_a)))
    hit_b, _ = corpus_round_hits(raw, want)
    print(u"  B 원시 로그 끝: %d/%d" % (len(hit_b), len(uniq)))
    hit_c, tie_c = corpus_round_hits(docs, want)
    print(u"  C 기록 문서 끝: %d/%d (+ 동점 %d)"
          % (len(hit_c), len(uniq), len(tie_c)))

    rows = []
    for name, ln, tok, dec, old, line, tbl in claims:
        key = "%.*f" % (dec, float(tok))
        a = sorted(hit_a.get(key, ()))
        at = sorted(tie_a.get(key, ()))
        c = sorted(hit_c.get(key, ())) + sorted(tie_c.get(key, ()))
        b = sorted(hit_b.get(key, ()))
        if a:
            verdict = "AGG_CSV"
        elif at:
            verdict = "AGG_CSV_TIE"
        elif c:
            verdict = "DOC_ONLY"
        elif b:
            verdict = "RAWLOG_ONLY"
        else:
            verdict = "NO_SOURCE"
        chk = 0
        for pv in COV.get((name, ln), ()):
            if abs(pv - abs(float(tok))) <= 0.5 * 10 ** (-dec) * 1.0001:
                chk = 1
                break
        rows.append({"draft": name, "line": ln, "token": tok,
                     "decimals": dec, "in_table": int(tbl),
                     "old_context": int(bool(old)), "verdict": verdict,
                     "cell_checked": chk,
                     "n_agg_csv": len(a) + len(at), "n_doc": len(c),
                     "n_rawlog": len(b),
                     "agg_csv_examples": ";".join((a + at)[:3]),
                     "doc_examples": ";".join(c[:2]),
                     "context": line[:220]})

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("")
    print(u"=== [1] 값 대조 — 판정 분포 (주장 건수) ===")
    for v in ("AGG_CSV", "AGG_CSV_TIE", "DOC_ONLY", "RAWLOG_ONLY",
              "NO_SOURCE"):
        tot = sum(1 for r in rows if r["verdict"] == v)
        newc = sum(1 for r in rows if r["verdict"] == v and not r["old_context"])
        print(u"  %-12s %4d건  (구버전 인용 문맥 제외: %4d건)" % (v, tot, newc))

    print("")
    # DOC_ONLY 도 경보에 포함한다. 말뭉치 C 는 **내가 쓰는 문서**이므로,
    # 어떤 숫자를 FINDINGS/RESUME 에 적기만 하면 NO_SOURCE 에서 벗어난다 —
    # 세탁 경로다. 실제로 32절을 쓰자 RAWLOG_ONLY 3건이 DOC_ONLY 로 옮겨가
    # 경보 목록에서 사라졌다. 그러므로 집계 CSV 에 닿지 않는 것은 전부 한
    # 계정으로 부른다.
    print(u"=== 집계 CSV 로 못 닿음 · 구버전 인용 아님 (살아 있는 주장) ===")
    print(u"  (DOC_ONLY 도 포함한다 — 말뭉치 C 는 내가 쓰는 문서라 거기에"
          u" 적는 것은 출처가 아니다)")
    reach = [r for r in rows
             if r["verdict"] in ("NO_SOURCE", "RAWLOG_ONLY", "DOC_ONLY")
             and not r["old_context"]]
    paid = [r for r in reach if r["cell_checked"]]
    bad = [r for r in reach if not r["cell_checked"]]
    print(u"  집계 CSV 로 못 닿은 주장 %d건 중 %d건은 칸 재집계로 갚았다"
          u" (런 CSV 에서 다시 계산해 일치)." % (len(reach), len(paid)))
    for r in sorted(bad, key=lambda r: (r["draft"], r["line"])):
        print(u"  %-26s L%-5d %-12s %-12s %s"
              % (r["draft"], r["line"], r["token"], r["verdict"],
                 r["context"][:100]))
    print(u"  총 %d건" % len(bad))

    # ---------------- [2] 표 행 교차 출처 (기본 비활성) ----------------
    if "--rowcheck" not in sys.argv:
        print("")
        print(u"=== [2] 표 행 교차 출처 — 건너뜀 (거짓 경보 100%). "
              u"칸 단위 검증은 verify_table_cells.py ===")
        digest_report(drafts, agg)
        print(u"저장: " + OUT)
        return
    print("")
    print(u"=== [2] 표 행 교차 출처 — 한 행의 칸들이 한 CSV 에서 같이 나오나 ===")
    flagged = 0
    checked = 0
    for p in drafts:
        name = os.path.basename(p)
        for i, line in enumerate(
                io.open(p, encoding="utf-8").read().split(u"\n"), 1):
            if not line.lstrip().startswith(u"|"):
                continue
            if any(f in line for f in OLD_FLAGS):
                continue
            toks = numeric_tokens(line)
            sets = []
            for tok in toks:
                d = len(tok.split(".")[1])
                key = "%.*f" % (d, float(tok))
                s = set(hit_a.get(key, ())) | set(tie_a.get(key, ()))
                if s:
                    sets.append((tok, s))
            if len(sets) < 2:
                continue
            checked += 1
            inter = set.intersection(*[s for _, s in sets])
            if inter:
                continue
            flagged += 1
            if flagged <= MAXROWS:
                print(u"  %-26s L%-5d 칸 %d개 공통 파일 0" % (name, i, len(sets)))
                print(u"      %s" % line.strip()[:150])
                for tok, s in sets:
                    print(u"        %-10s <- %s" % (tok, ";".join(sorted(s)[:3])))
    print(u"  검사한 표 행 %d개 중 공통 출처 없음 **%d개**" % (checked, flagged))
    print("")
    print(u"저장: " + OUT)


if __name__ == "__main__":
    main()
