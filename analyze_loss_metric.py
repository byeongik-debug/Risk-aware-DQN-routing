"""The reported loss metric is a per-hop mean, so longer paths score better for free.

network_env.calculate_path_metrics returns

    "total_delay": total_delay,              # sum over the hops of the path
    "total_loss":  total_loss / hop_count,   # MEAN over the hops of the path

Delay is an end-to-end path property; loss is a per-link average.  The two are reported
side by side as if they were the same kind of quantity, and the averaged one is not
neutral between routing methods: adding a clean hop lowers it even though the packet now
crosses one more lossy link.  That matters here because the methods under comparison
differ in path length by design -- OSPF takes about 6.0 hops and the DQN variants 6.5 to
9.5 -- so the metric hands the longer-path method an advantage it did not earn.

This script measures how large that effect is, using only data already on disk:

  1. Pool every per-run CSV that reports both loss and hop, and correlate them.  If the
     metric were carrying link-quality information independent of path length, the
     correlation would be weak.
  2. Re-express the published reference tables under two alternative conventions --
     the path sum (sum of per-link rates) and the compounded end-to-end drop probability
     1 - prod(1 - p_i) -- and report how each headline comparison moves.

Step 2 reconstructs from published aggregates, so it is an estimate: the mean of a
nonlinear function is not that function of the mean.  reeval_loss_conventions.py
computes the same three conventions exactly, per episode, from the trained models.
"""

import csv
import glob
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from review_stats import mean_std, t_two_sided_p  # noqa: E402

RESULTS_DIR = os.path.join(BASE_DIR, "..", "SDN_DQN_new_env", "results")
OUT_DIR = os.path.join(BASE_DIR, "review_results")

# Per-run files carrying both loss and hop.  Every variant is pooled: the question is
# whether the metric tracks path length regardless of what produced the path.
RUN_GLOB = "*_run*_result.csv"

# Published aggregate tables, and the column holding the method label (None = one row
# per scenario, the method being the file itself).
REFERENCE_TABLES = (
    ("exp1_reference.csv", "model"),
    ("ospf_reference.csv", None),
    ("qos_reference.csv", None),
    ("norisk_reference.csv", None),
    ("baseline_reference.csv", None),
)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan"), float("nan"), n - 2, float("nan")
    r = sxy / math.sqrt(sxx * syy)
    if abs(r) >= 1.0:
        return r, float("inf"), n - 2, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return r, t, n - 2, t_two_sided_p(t, n - 2)


def path_sum(loss_per_hop, hop):
    """Sum of per-link loss rates: the quantity delay is already reported as."""
    return loss_per_hop * hop


def compounded(loss_per_hop, hop):
    """1 - prod(1 - p_i), the chance a packet is dropped somewhere on the path.

    Per-link rates here reach 0.35, so the sum overshoots 1 and is not a probability;
    the compounded form is the one that stays interpretable.
    """
    return 1.0 - (1.0 - loss_per_hop) ** hop


def collect_runs():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, RUN_GLOB))):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if "loss" not in row or "hop" not in row:
                    continue
                label = (row.get("factor") or row.get("weight_config")
                         or row.get("window") or "?")
                rows.append({
                    "file": os.path.basename(path),
                    "label": str(label),
                    "scenario": row.get("scenario", "dynamic"),
                    "loss": float(row["loss"]),
                    "hop": float(row["hop"]),
                    "delay": float(row["delay"]),
                })
    return rows


def write_correlations(rows, writer):
    """Correlate each loss convention against hop count, within each scenario.

    Pooling the scenarios would answer the wrong question.  Scenario sets both the link
    quality and the path length -- congestion carries high per-link loss and 7-hop paths
    while normal carries low loss and 6-hop paths -- so a pooled correlation is dominated
    by that between-scenario confound and comes out positive, hiding the effect.  The
    dilution is a within-scenario property: holding the network fixed and varying only
    the risk formula, does a longer path score better?  The pooled figure is reported
    last, labelled, so the confound is visible rather than silently averaged in.
    """
    by_scenario = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(row)

    conventions = (
        ("loss_per_hop_as_published", lambda r: r["loss"]),
        ("loss_path_sum", lambda r: path_sum(r["loss"], r["hop"])),
        ("loss_compounded", lambda r: compounded(r["loss"], r["hop"])),
        ("delay_path_sum_as_published", lambda r: r["delay"]),
    )

    groups = [(name, group) for name, group in sorted(by_scenario.items())]
    groups.append(("POOLED (confounded)", rows))

    print("\n=== does each metric just restate path length? (within scenario) ===")
    for scenario, group in groups:
        hops = [r["hop"] for r in group]
        hop_mean, hop_std = mean_std(hops)
        print("\n-- {}  (n={} runs, hop {:.3f} +- {:.3f})".format(
            scenario, len(group), hop_mean, hop_std))
        if len(group) < 3 or hop_std == 0:
            print("   hop count does not vary here; nothing to correlate")
            continue
        print("   {:<30} {:>8} {:>8} {:>9} {:>6} {:>11}".format(
            "metric", "r", "r^2", "t", "df", "p"))
        for name, fn in conventions:
            r, t, df, p = pearson(hops, [fn(x) for x in group])
            print("   {:<30} {:>+8.4f} {:>8.4f} {:>9.3f} {:>6d} {:>11.3g}".format(
                name, r, r * r, t, df, p))
            writer.writerow({
                "section": "correlation_with_hop", "scenario": scenario,
                "metric": name, "n": len(group),
                "r": "{:.6f}".format(r), "r_squared": "{:.6f}".format(r * r),
                "t": "{:.4f}".format(t), "df": df, "p": "{:.6g}".format(p),
            })

    print("\nWhere hop count varies, the published per-hop loss falls as paths lengthen")
    print("while both end-to-end forms rise, as added hops physically must.  An r^2 near")
    print("1 with a negative sign means the metric is path length read backwards.")


def write_reference_tables(writer):
    print("\n=== published reference tables re-expressed (estimates from aggregates) ===")
    for filename, label_col in REFERENCE_TABLES:
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            continue
        method_default = filename.replace("_reference.csv", "").upper()
        print("\n-- {}".format(filename))
        print("{:<10} {:<12} {:>7} {:>10} {:>10} {:>10}".format(
            "method", "scenario", "hop", "published", "path_sum", "compounded"))
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if "loss" not in row or "hop" not in row:
                    continue
                method = row[label_col] if label_col else method_default
                scenario = row.get("scenario", "?")
                loss, hop = float(row["loss"]), float(row["hop"])
                print("{:<10} {:<12} {:>7.3f} {:>10.5f} {:>10.5f} {:>10.5f}".format(
                    method, scenario, hop, loss, path_sum(loss, hop),
                    compounded(loss, hop)))
                writer.writerow({
                    "section": "reference_table", "source": filename,
                    "method": method, "scenario": scenario,
                    "hop": "{:.4f}".format(hop),
                    "loss_published": "{:.6f}".format(loss),
                    "loss_path_sum": "{:.6f}".format(path_sum(loss, hop)),
                    "loss_compounded": "{:.6f}".format(compounded(loss, hop)),
                    "delay_published": row.get("delay", ""),
                })


def write_headline_deltas(writer):
    """How much of each OSPF-vs-QoS-DQN loss claim is path-length dilution."""
    path = os.path.join(RESULTS_DIR, "exp1_reference.csv")
    if not os.path.exists(path):
        return
    by_scenario = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_scenario.setdefault(row["scenario"], {})[row["model"]] = row

    conventions = (
        ("published", lambda loss, hop: loss),
        ("path_sum", path_sum),
        ("compounded", compounded),
    )

    print("\n=== OSPF -> QoS-DQN loss improvement under each convention ===")
    print("{:<12} {:>12} {:>12} {:>12}".format(
        "scenario", "published", "path_sum", "compounded"))
    for scenario in ("normal", "congestion", "unstable", "dynamic"):
        pair = by_scenario.get(scenario, {})
        if "OSPF" not in pair or "QoS-DQN" not in pair:
            continue
        values = {}
        for name, fn in conventions:
            ospf = fn(float(pair["OSPF"]["loss"]), float(pair["OSPF"]["hop"]))
            dqn = fn(float(pair["QoS-DQN"]["loss"]), float(pair["QoS-DQN"]["hop"]))
            values[name] = (ospf - dqn) / ospf * 100
        print("{:<12} {:>11.1f}% {:>11.1f}% {:>11.1f}%".format(
            scenario, values["published"], values["path_sum"], values["compounded"]))
        writer.writerow({
            "section": "headline_improvement", "scenario": scenario,
            "improvement_published_pct": "{:.3f}".format(values["published"]),
            "improvement_path_sum_pct": "{:.3f}".format(values["path_sum"]),
            "improvement_compounded_pct": "{:.3f}".format(values["compounded"]),
        })
    print("\nPositive means QoS-DQN loses fewer packets than OSPF.  The gap between the")
    print("first column and the other two is the part of the published claim that comes")
    print("from QoS-DQN taking more hops rather than from choosing better links.")


def main():
    rows = collect_runs()
    if not rows:
        raise SystemExit("no per-run result files found in " + RESULTS_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    out_csv = os.path.join(OUT_DIR, "loss_metric_hop_dilution.csv")
    fields = ["section", "metric", "source", "method", "scenario", "n", "hop",
              "loss_published", "loss_path_sum", "loss_compounded", "delay_published",
              "improvement_published_pct", "improvement_path_sum_pct",
              "improvement_compounded_pct", "r", "r_squared", "t", "df", "p"]

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        write_correlations(rows, writer)
        write_reference_tables(writer)
        write_headline_deltas(writer)

    hop_mean, hop_std = mean_std([r["hop"] for r in rows])
    print("\npooled runs: n={}, hop {:.3f} +- {:.3f} (range {:.2f}-{:.2f})".format(
        len(rows), hop_mean, hop_std,
        min(r["hop"] for r in rows), max(r["hop"] for r in rows)))
    print("wrote " + out_csv)


if __name__ == "__main__":
    main()
