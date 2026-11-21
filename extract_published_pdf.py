# -*- coding: utf-8 -*-
u"""발표본 PDF 의 인쇄 상수를 **좌표로** 뽑아 CSV 로 고정한다.

왜 필요한가. 4.6.3절 초안은 발표본 표 5·표 6 의 인쇄값(427.2 · 0.13822 ·
345.3 · 0.08256 · 350.1 · 403.7 …)과 본문 백분율(23.7% · 47.4%)을 놓고
산수가 자기 표와 안 맞는다고 말한다. 그런데 그 인쇄값들은 **커밋된 집계
CSV 어디에도 없다** — 측정값이 아니라 인쇄물의 내용이기 때문이다. 그래서
감사에서 전부 경보로 뜬다. 내가 문서에 "PDF 에 이렇게 적혀 있다" 고 쓰는
것은 출처가 아니다(그게 32.2.3 의 세탁 경로다). 그러므로 PDF 에서 직접
기계적으로 뽑아 산출물로 만든다.

원본은 열기만 한다. 쓰기는 review_results/ 로만 한다.

좌표를 쓰는 이유: 표 5 는 한글 PDF 의 2열 조판이라 extract_text() 로는 행이
섞여 나온다("NoRisk-\n427.2 0.13822 7.00\nDQN"). 열 머리말 Delay/Loss/Hop
의 x0 를 읽고 데이터 토큰을 같은 x0 에 배정하면 섞임과 무관하게 확정된다.
배정에 쓴 x0 자체도 CSV 에 남긴다(초안이 인용하는 330.5 / 368 / 416.9).
"""
import csv
import io
import os
import re
import sys

import pdfplumber

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "review_results")
PDF = os.path.join(os.path.dirname(HERE), "SDN_DQN_new_env",
                   u"동적 QoS 환경을 위한 "
                   u"시간적 위험 기반 "
                   u"심층강화학습 라우팅"
                   u".pdf")

NUM = re.compile(u"^\\(?[+−-]?\\d+(?:\\.\\d+)?%?\\)?$")


def _f(tok):
    t = tok.replace(u"−", u"-").replace(u"%", u"")
    t = t.replace(u"(", u"").replace(u")", u"").replace(u"+", u"")
    return float(t)


def words(pg, x0=None, x1=None, top=None, bot=None):
    out = []
    for w in pg.extract_words():
        if x0 is not None and w["x0"] < x0:
            continue
        if x1 is not None and w["x0"] > x1:
            continue
        if top is not None and w["top"] < top:
            continue
        if bot is not None and w["top"] > bot:
            continue
        out.append(w)
    return sorted(out, key=lambda w: (round(w["top"], 1), w["x0"]))


def table5(pg, rows):
    u"""표 5 — 열 머리말 x0 로 배정한다."""
    head = {}
    for w in words(pg, x0=300, top=140, bot=150):
        if w["text"] in (u"Delay", u"Loss", u"Hop"):
            head[w["text"]] = w["x0"]
    assert sorted(head) == [u"Delay", u"Hop", u"Loss"], head

    for col, x in sorted(head.items(), key=lambda kv: kv[1]):
        rows.append(dict(name=u"t5_colx_" + col.lower(), page=pg.page_number,
                         table=u"표 5", row=u"(열 머리말)",
                         col=col, value=u"%.2f" % x, x0=u"%.2f" % x,
                         top=u"%.2f" % 145.25, kind=u"col_x0"))

    # 데이터 토큰: 열 x0 에 1.5pt 안으로 붙는 숫자만.
    cells = {}
    for w in words(pg, x0=300, top=150, bot=210):
        if not NUM.match(w["text"]):
            continue
        for col, x in head.items():
            if abs(w["x0"] - x) <= 1.5:
                cells.setdefault(round(w["top"], 2), {})[col] = w
    tops = sorted(cells)
    assert len(tops) == 2, tops

    for t in tops:
        lab = u" ".join(w["text"] for w in words(pg, x0=285, x1=325,
                                                 top=t - 8, bot=t + 8))
        lab = lab.replace(u"NoRisk- DQN", u"NoRisk-DQN")
        lab = lab.replace(u"Propose d", u"Proposed")
        key = u"norisk" if lab.startswith(u"NoRisk") else u"prop"
        for col, w in sorted(cells[t].items(), key=lambda kv: kv[1]["x0"]):
            rows.append(dict(name=u"t5_%s_%s" % (key, col.lower()),
                             page=pg.page_number, table=u"표 5", row=lab,
                             col=col, value=w["text"], x0=u"%.2f" % w["x0"],
                             top=u"%.2f" % w["top"], kind=u"cell"))
    return [r["name"] for r in rows]


def table6(p6, p7, rows):
    u"""표 6 — 머리말·첫 행은 6쪽 바닥, 둘째 행은 7쪽 머리에 걸쳐 있다."""
    a = [w for w in words(p6, x0=395, top=620) if NUM.match(w["text"])]
    got = [w["text"] for w in a]
    assert got == [u"350.1", u"0.08355", u"1.5", u"0.00057"], got
    b = [w for w in words(p7, x1=260, bot=130) if NUM.match(w["text"])]
    got = [w["text"] for w in b]
    assert got == [u"403.7", u"0.12896", u"2.4", u"0.00177",
                   u"(+15.3%)", u"(+54.3%)"], got

    spec = [(u"t6_prop_delay", u"Risk-DQN", u"Delay", a[0], a[2]),
            (u"t6_prop_loss", u"Risk-DQN", u"Loss", a[1], a[3]),
            (u"t6_qos_delay", u"QoS-Model", u"Delay", b[0], b[2]),
            (u"t6_qos_loss", u"QoS-Model", u"Loss", b[1], b[3])]
    for name, row, col, m, s in spec:
        pg = p6 if name.startswith(u"t6_prop") else p7
        rows.append(dict(name=name, page=pg.page_number, table=u"표 6",
                         row=row, col=col, value=m["text"],
                         x0=u"%.2f" % m["x0"], top=u"%.2f" % m["top"],
                         kind=u"cell"))
        rows.append(dict(name=name + u"_sd", page=pg.page_number,
                         table=u"표 6", row=row, col=col + u" (sd)",
                         value=s["text"], x0=u"%.2f" % s["x0"],
                         top=u"%.2f" % s["top"], kind=u"cell_sd"))
    for name, w in ((u"t6_qos_delay_pct", b[4]), (u"t6_qos_loss_pct", b[5])):
        rows.append(dict(name=name, page=p7.page_number, table=u"표 6",
                         row=u"QoS-Model", col=u"(증감률)",
                         value=u"%.1f" % _f(w["text"]), x0=u"%.2f" % w["x0"],
                         top=u"%.2f" % w["top"], kind=u"cell_pct"))


# (이름, 쪽, 단 영역(x0,x1,top,bot), 정규식, 그룹)
# 2단 조판이라 쪽 전체를 평탄화하면 좌·우 단이 한 줄에서 섞여 "평"+"균"이
# 끊긴다. 단 영역의 단어만 모아 붙인다.
_P6 = u"평균Delay와PacketLoss가각각" \
      u"(\\d+\\.\\d+)%와(\\d+\\.\\d+)%감소"
_P7 = u"평균Delay를(\\d+\\.\\d+)%,PacketLoss를" \
      u"(\\d+\\.\\d+)%감소"
BODY = [
    (u"body_t5_delay_pct", 6, (283, None, 230, None), _P6, 1),
    (u"body_t5_loss_pct", 6, (283, None, 230, None), _P6, 2),
    (u"body_t6_delay_pct", 7, (None, 260, 135, 215), _P7, 1),
    (u"body_t6_loss_pct", 7, (None, 260, 135, 215), _P7, 2),
]


def body(pages, rows):
    u"""본문 문장의 백분율 — 단 영역을 붙여 평탄화한 뒤 문장째로 찾는다."""
    for name, pno, box, pat, grp in BODY:
        pg = pages[pno - 1]
        ws = words(pg, x0=box[0], x1=box[1], top=box[2], bot=box[3])
        flat = re.sub(u"\\s+", u"", u"".join(w["text"] for w in ws))
        m = re.search(pat, flat)
        assert m, (name, pno, flat[:80])
        rows.append(dict(name=name, page=pno, table=u"본문",
                         row=m.group(0), col=u"", value=m.group(grp),
                         x0=u"", top=u"", kind=u"body_pct"))


def main():
    assert os.path.exists(PDF), PDF
    rows = []
    with pdfplumber.open(PDF) as pdf:
        pages = pdf.pages
        print(u"PDF %d쪽" % len(pages))
        table5(pages[5], rows)
        table6(pages[5], pages[6], rows)
        body(pages, rows)

    out = os.path.join(RES, "published_pdf_constants.csv")
    cols = ["name", "page", "table", "row", "col", "value", "x0", "top",
            "kind"]
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for r in rows:
        print(u"  %-22s %-8s %-14s %-10s %s"
              % (r["name"], r["table"], r["row"][:14], r["col"], r["value"]))
    print(u"\n%d행 -> %s" % (len(rows), out))


if __name__ == "__main__":
    sys.exit(main())
