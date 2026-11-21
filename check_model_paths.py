"""체크포인트 상대경로 가드 — 맨 파일명이 cwd에 따라 다른 모델을 집는지 검사한다.

왜 필요한가. `run_topology_generalization_dynamic.py:34`는
`MODEL_PATH = Path("risk_only_model_final.pth")`이다. 디렉터리 이름이 없으므로
어떤 모델을 읽는지는 **실행한 위치**가 정한다. 이 저장소에는 그 이름의 파일이
서로 다른 내용으로 여러 벌 있어서, 같은 명령이 위치에 따라 다른 모델을 평가한다.
표 14가 표 10/11/12와 다른 체크포인트로 만들어진 경로가 바로 이것이다.

판정은 AST로 한다. 문자열이 `.pth`로 끝나는 것만으로는 위험하지 않다 —
`os.path.join(BASE_DIR, "risk_only_model_final.pth")`는 스크립트 위치에 못 박혀
있으므로 안전하다. 위험한 것은 그 문자열이 **단독으로** 경로가 되는 자리,
즉 `X = "name.pth"` 또는 `Path("name.pth")`다. 앵커(`BASE_DIR`, `__file__`,
`/` 연산, 인자 둘 이상의 join)가 붙은 자리는 '고정'으로 따로 센다.

하는 일:
1. 모든 `.py`를 파싱해서 `.pth` 문자열 상수를 찾고 단독/고정으로 가른다.
2. 그 이름의 파일을 저장소 전체에서 찾아 md5·크기·수정일을 적는다.
3. **단독 참조가 있고** 내용이 다른 사본이 둘 이상인 이름을 '모호'로 올린다.
4. 단독 참조마다 "제 디렉터리에서 그냥 실행하면 무엇을 집는가"를 해석한다.
5. `--weights`를 주면 후보를 torch로 열어 같은 원소 비율을 보고한다
   (재저장인지 따로 학습된 다른 모델인지 가른다).

모호한 이름이 하나라도 있으면 종료코드 1이다. 원본 코드는 읽기만 한다.

재현:
    PYTHONIOENCODING=utf-8 python -X utf8 check_model_paths.py
    PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE python -X utf8 \
        check_model_paths.py --weights
"""

import argparse
import ast
import collections
import hashlib
import math
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
BACKUP_MARK = "model_backup"
SELF = os.path.basename(__file__)
ANCHOR_NAMES = ("BASE_DIR", "BASE", "ROOT", "__file__", "MODEL_DIR",
                "MODELS_DIR", "OUT_DIR", "DIR")


def walk(root, include_backups):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if not include_backups and BACKUP_MARK in dirpath.replace(os.sep, "/"):
            continue
        for name in filenames:
            yield os.path.join(dirpath, name)


def digest(path):
    h = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inventory(include_backups):
    """{basename: [(md5, size, 날짜, 상대경로), ...]} — 모든 .pth 사본."""
    found = collections.defaultdict(list)
    for path in walk(ROOT, include_backups):
        if not path.endswith(".pth"):
            continue
        stat = os.stat(path)
        found[os.path.basename(path)].append(
            (digest(path), stat.st_size,
             time.strftime("%Y-%m-%d", time.localtime(stat.st_mtime)),
             os.path.relpath(path, ROOT).replace(os.sep, "/")))
    return {k: sorted(v) for k, v in found.items()}


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def is_anchored(node, parents):
    """이 문자열 상수가 디렉터리에 못 박혀 있는가."""
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.BinOp) and isinstance(
                parent.op, (ast.Div, ast.Add, ast.Mod)):
            return True
        if isinstance(parent, ast.JoinedStr):
            return True
        if isinstance(parent, ast.Call):
            name = dotted(parent.func)
            if name.endswith("join") and len(parent.args) > 1:
                return True
            if name.split(".")[-1] in ("Path", "PurePath") \
                    and len(parent.args) > 1:
                return True
            return False
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.Return,
                               ast.keyword, ast.Dict, ast.List, ast.Tuple,
                               ast.Expr, ast.Subscript, ast.Compare)):
            return False
        parent = parents.get(parent)
    return False


def scan_references(include_backups):
    """{basename: {"bare": [...], "anchored": n}} — AST 기준."""
    refs = collections.defaultdict(
        lambda: {"bare": [], "anchored": 0, "unparsed": 0})
    unparsed = []
    for path in walk(ROOT, include_backups):
        if not path.endswith(".py") or os.path.basename(path) == SELF:
            continue
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
            tree = ast.parse(text)
        except (OSError, SyntaxError, ValueError) as exc:
            unparsed.append((rel, str(exc)[:60]))
            continue
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        lines = text.split("\n")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) \
                    or not isinstance(node.value, str):
                continue
            value = node.value
            if not value.endswith(".pth") or "/" in value:
                continue
            if any(ch in value for ch in (chr(92), " ", ":", "{", "%")):
                continue   # 경로 조각이거나 사람이 읽는 문장이다
            if is_anchored(node, parents):
                refs[value]["anchored"] += 1
            else:
                lineno = getattr(node, "lineno", 0)
                src = lines[lineno - 1].strip() if 0 < lineno <= len(lines) \
                    else ""
                refs[value]["bare"].append((rel, lineno, src))
    return refs, unparsed


def resolve_from(cwd_rel, copies):
    """cwd_rel에서 맨 이름을 열면 어느 사본인가."""
    for md5, size, mtime, rel in copies:
        if os.path.dirname(rel) == cwd_rel.rstrip("/"):
            return (md5, rel)
    return None


def compare_weights(rels):
    """후보 쌍의 같은 원소 비율 — 재저장인지 딴 학습인지 가른다."""
    try:
        import torch
    except ImportError as exc:
        print("    torch 없음, 가중치 비교 생략 (%s)" % exc)
        return
    loaded = {}
    for rel in rels:
        try:
            blob = torch.load(os.path.join(ROOT, rel), map_location="cpu")
        except Exception as exc:                            # noqa: BLE001
            print("    로드 실패 %s (%s)" % (rel, str(exc)[:60]))
            continue
        loaded[rel] = blob.get("model_state_dict", blob) \
            if isinstance(blob, dict) else blob
    keys = list(loaded)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = loaded[keys[i]], loaded[keys[j]]
            shared = [k for k in a if k in b and getattr(a[k], "shape", None)
                      == getattr(b[k], "shape", 1)]
            total = same = zeros = 0
            max_abs = 0.0
            sq_diff = sq_ref = 0.0
            for k in shared:
                x = a[k].flatten().float()
                y = b[k].flatten().float()
                eq = x == y
                total += int(x.numel())
                same += int(eq.sum())
                zeros += int((eq & (y == 0)).sum())
                max_abs = max(max_abs, float((x - y).abs().max()))
                sq_diff += float(((x - y) ** 2).sum())
                sq_ref += float((y ** 2).sum())
            frac = same / total * 100.0 if total else float("nan")
            # 규모 대비 차이: ||x-y|| / ||y||. 원소별 비율은 분모가 0에 가까운
            # 원소 하나에 끌려가므로 쓰지 않는다.
            rel = math.sqrt(sq_diff / sq_ref) if sq_ref else float("nan")
            print("    %s" % os.path.dirname(keys[i]))
            print("    vs %s" % os.path.dirname(keys[j]))
            print("      공통 키 %d개, 동일 원소 %d/%d (%.2f%%)"
                  % (len(shared), same, total, frac))
            print("      그 중 둘 다 정확히 0인 원소 %d개 (%.1f%%)"
                  % (zeros, zeros / same * 100.0 if same else float("nan")))
            print("      규모 대비 차이 ||x-y||/||y|| = %.3f, 최대 절대차 %.4f"
                  % (rel, max_abs))
            if total and same / total < 0.99:
                print("      -> 재저장이 아니다. 따로 학습된 다른 모델이다.")
            elif total:
                print("      -> 사실상 같은 가중치다 (재저장).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", action="store_true",
                        help="모호한 이름의 후보를 torch로 열어 비교")
    parser.add_argument("--include-backups", action="store_true",
                        help="model_backup/ 안도 검사 (기본 제외)")
    parser.add_argument("--only", default="", help="이 이름만 검사")
    args = parser.parse_args()

    inv = inventory(args.include_backups)
    refs, unparsed = scan_references(args.include_backups)
    names = sorted(set(inv) | set(refs))
    if args.only:
        names = [n for n in names if n == args.only]

    print("=" * 78)
    print("체크포인트 상대경로 가드")
    print("저장소 %s" % ROOT)
    print("백업 디렉터리 %s" % ("포함" if args.include_backups else "제외"))
    print("=" * 78)
    bare_total = sum(len(v["bare"]) for v in refs.values())
    anchored_total = sum(v["anchored"] for v in refs.values())
    print(".pth 파일 이름 %d종 / 코드의 .pth 문자열: 단독 %d곳, 고정 %d곳"
          % (len(inv), bare_total, anchored_total))
    if unparsed:
        print("파싱 실패 %d개 (검사에서 빠짐): %s"
              % (len(unparsed), ", ".join(r for r, _ in unparsed[:4])))
    print()

    ambiguous, bare_safe = [], []
    for name in names:
        copies = inv.get(name, [])
        bare = refs.get(name, {}).get("bare", [])
        if not bare:
            continue
        if len({c[0] for c in copies}) > 1:
            ambiguous.append(name)
        else:
            bare_safe.append((name, len(bare), len(copies)))

    if bare_safe:
        print("단독 참조지만 사본 내용이 하나뿐 — 지금은 안전 (%d종)"
              % len(bare_safe))
        for name, nbare, ncopy in bare_safe:
            note = "" if ncopy else "  (저장소에 사본 없음 — 학습 산출물)"
            print("   %-34s 단독 %2d곳, 사본 %d벌%s"
                  % (name, nbare, ncopy, note))
        print("   사본이 하나라도 더 생기면 그 순간 모호해진다.")
        print()

    if not ambiguous:
        print("모호한 이름 없음 — 맨 파일명이 가리키는 모델이 어디서 실행해도 같다.")
        return 0

    for name in ambiguous:
        copies = inv[name]
        by_md5 = collections.OrderedDict()
        for md5, size, mtime, rel in copies:
            by_md5.setdefault(md5, []).append((size, mtime, rel))
        print("=" * 78)
        print("모호: %s — 사본 %d벌, 서로 다른 내용 %d종"
              % (name, len(copies), len(by_md5)))
        print("=" * 78)
        for md5, group in by_md5.items():
            size, mtime, _ = group[0]
            print("  %s  %9d B  %s" % (md5[:8], size, mtime))
            for _, _, rel in group:
                print("      %s" % rel)
        print()
        print("  이 이름을 단독으로 쓰는 자리 — 그 디렉터리에서 실행하면:")
        seen = {}
        for rel, lineno, src in sorted(refs[name]["bare"]):
            cwd = os.path.dirname(rel)
            hit = resolve_from(cwd, copies)
            tag = hit[0][:8] if hit else "없음"
            seen.setdefault(tag, []).append("%s:%d" % (rel, lineno))
        for tag, places in sorted(seen.items()):
            print("    -> %s : %s" % (tag, ", ".join(places[:6])
                                      + (" 외 %d곳" % (len(places) - 6)
                                         if len(places) > 6 else "")))
        resolved = [t for t in seen if t != "없음"]
        if len(resolved) > 1:
            print("  같은 코드가 cwd에 따라 %d개의 다른 모델을 평가한다."
                  % len(resolved))
        if args.weights:
            print("  가중치 비교:")
            compare_weights([group[0][2] for group in by_md5.values()])
        print()

    print("권고 — 스크립트 위치 기준으로 못 박고(`Path(__file__).resolve().parent`),")
    print("평가 출력 CSV에 체크포인트 md5를 한 열로 적어라. 그러면 어떤 표가 어떤")
    print("모델에서 나왔는지 사후에 확인된다 (`reeval_table14_final.py` 참고).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
