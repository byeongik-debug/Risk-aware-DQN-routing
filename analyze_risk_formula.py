"""Do the risk formula weights and factors actually change routing behaviour?

Reviewer R1 and R3-1 ask why the risk weights are what they are.  Two experiments in
SDN_DQN_new_env already bear on that and neither was tested for significance:

  train_weight_compare.py   W-Equal and W-Temporal reweight the five factors
  train_factor_ablation.py  -Util .. -Type zero one factor at a time

Both answer the same question, so they are aggregated here against one baseline.
The baseline runs are window5_dynamic_run01-05: factor_ablation_summary.csv reports
Full as 368.7961 +- 21.7457 ms, which is exactly the mean and sample std of those five
files, so they are the per-run values behind both the Full and the W-Current rows.

Only delay, loss and hop are compared.  The variant CSVs carry no risk column, which is
just as well -- risk is computed by the very formula these experiments perturb, so a
risk difference across weight configs would measure the ruler, not the route.  Delay,
loss and hop are physical path properties and stay comparable.
"""

import csv
import glob
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from review_stats import (fmt_p, holm_bonferroni, mean_std,  # noqa: E402
                          min_detectable_effect, required_n, welch)

RESULTS_DIR = os.path.join(BASE_DIR, "..", "SDN_DQN_new_env", "results")
OUT_DIR = os.path.join(BASE_DIR, "review_results")

BASELINE_LABEL = "W-Current (Full)"

# success_rate is 100.0 in every run of every variant, so it has zero variance and no
# test is defined on it.  It is reported once at the end rather than as a row per
# variant, which would be twenty identical lines of nan.
METRICS = ("delay", "loss", "hop")

# label-column, filename glob.  The label column holds the display name the training
# script wrote, so the mapping from file prefix to label stays with the data.
VARIANT_SOURCES = (
    ("weight", "weight_config", "weight_*_dynamic_run*_result.csv"),
    ("factor", "factor", "factor_*_dynamic_run*_result.csv"),
)


def read_rows(pattern):
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                yield os.path.basename(path), row


def load():
    """{group: {label: {metric: [values]}}}, the shared baseline, and success rates."""
    groups = {}
    baseline = {m: [] for m in METRICS}
    success = {}

    for _, row in read_rows("window5_dynamic_run*_result.csv"):
        for metric in METRICS:
            baseline[metric].append(float(row[metric]))
        success.setdefault(BASELINE_LABEL, []).append(float(row["success_rate"]))

    for group, label_col, pattern in VARIANT_SOURCES:
        bucket = groups.setdefault(group, {})
        for name, row in read_rows(pattern):
            label = row.get(label_col)
            if not label:
                raise SystemExit("{}: no '{}' column".format(name, label_col))
            per_metric = bucket.setdefault(label, {m: [] for m in METRICS})
            for metric in METRICS:
                per_metric[metric].append(float(row[metric]))
            success.setdefault(label, []).append(float(row["success_rate"]))

    return groups, baseline, success


def compare(groups, baseline):
    """One record per (group, variant, metric), with Holm adjustment inside each group.

    Each group is one experiment answering one question, so the family of tests to
    correct over is that group: five factor variants times three metrics, or two weight
    configs times three.  Adjusting across both groups together would be conservative
    beyond what the design calls for; adjusting not at all would let one p just under
    0.05 out of fifteen look like a result.
    """
    records = []
    for group in sorted(groups):
        start = len(records)
        for label in sorted(groups[group]):
            for metric in METRICS:
                values = groups[group][label][metric]
                ref = baseline[metric]
                mean, std = mean_std(values)
                ref_mean, ref_std = mean_std(ref)
                diff, t, df, p = welch(values, ref)
                mde = min_detectable_effect(values, ref)
                records.append({
                    "group": group, "variant": label, "metric": metric,
                    "n": len(values), "mean": mean, "std": std,
                    "baseline_n": len(ref), "baseline_mean": ref_mean,
                    "baseline_std": ref_std, "diff": diff,
                    "diff_pct": diff / ref_mean * 100 if ref_mean else float("nan"),
                    "t": t, "df": df, "p": p, "mde": mde,
                    "mde_pct": mde / ref_mean * 100 if ref_mean else float("nan"),
                    "required_n": required_n(values, ref),
                })
        family = records[start:]
        for record, adjusted in zip(family,
                                    holm_bonferroni([r["p"] for r in family])):
            record["p_holm"] = adjusted
    return records


def main():
    groups, baseline, success = load()
    os.makedirs(OUT_DIR, exist_ok=True)

    n_base = len(baseline["delay"])
    if n_base == 0:
        raise SystemExit("no window5_dynamic baseline runs found in " + RESULTS_DIR)

    records = compare(groups, baseline)
    out_csv = os.path.join(OUT_DIR, "risk_formula_significance.csv")
    fields = ["group", "variant", "metric", "n", "mean", "std",
              "baseline_n", "baseline_mean", "baseline_std", "diff", "diff_pct",
              "t", "df", "p", "p_holm", "mde", "mde_pct", "required_n",
              "significant_holm_05"]

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            holm = record["p_holm"]
            row["significant_holm_05"] = "yes" if holm == holm and holm < 0.05 else "no"
            for key in ("mean", "std", "baseline_mean", "baseline_std", "diff",
                        "diff_pct", "t", "df", "p", "p_holm", "mde", "mde_pct"):
                row[key] = "{:.6f}".format(record[key])
            writer.writerow(row)

    last_group = None
    for record in records:
        if record["group"] != last_group:
            last_group = record["group"]
            print("\n=== {} vs {} (dynamic, n={} runs each) ===".format(
                last_group, BASELINE_LABEL, n_base))
            print("{:<12} {:<7} {:>9} {:>9} {:>9} {:>8} {:>8} {:>9} {:>6}".format(
                "variant", "metric", "mean", "diff", "min.det.", "p", "p_holm",
                "runs_needed", ""))
        flag = ""
        if record["p"] == record["p"] and record["p"] < 0.05:
            holm = record["p_holm"]
            flag = "  *" if holm == holm and holm < 0.05 else "  (raw only)"
        need = record["required_n"]
        print("{:<12} {:<7} {:>9.4f} {:>+9.4f} {:>9.4f} {:>8} {:>8} {:>11}{}".format(
            record["variant"], record["metric"], record["mean"], record["diff"],
            record["mde"], fmt_p(record["p"]), fmt_p(record["p_holm"]),
            "{:d}".format(int(need)) if need == need else "n/a", flag))

    print("\nbaseline {} (dynamic, n={}):".format(BASELINE_LABEL, n_base))
    for metric in METRICS:
        mean, std = mean_std(baseline[metric])
        print("  {:<7} {:.4f} +- {:.4f}".format(metric, mean, std))

    rates = sorted({value for values in success.values() for value in values})
    print("  success_rate  {} in all {} runs of all {} variants -- zero variance, so no"
          " test is defined".format(
              rates if len(rates) > 1 else rates[0],
              sum(len(v) for v in success.values()), len(success)))

    tested = [r for r in records if r["p"] == r["p"]]
    raw_hits = [r for r in tested if r["p"] < 0.05]
    holm_hits = [r for r in tested if r["p_holm"] == r["p_holm"] and r["p_holm"] < 0.05]
    print("\n{} comparisons: {} significant before correction, {} after Holm.".format(
        len(tested), len(raw_hits), len(holm_hits)))

    print("\nmin.det. is the smallest difference this sample size could have called")
    print("significant at alpha=0.05.  A diff well under it means the experiment could")
    print("not resolve an effect that size, not that the effect is absent:")
    for metric in METRICS:
        worst = max((r for r in records if r["metric"] == metric),
                    key=lambda r: r["mde"])
        print("  {:<7} resolves nothing below {:.4f} ({:.1f}% of the {:.4f} baseline)"
              .format(metric, worst["mde"], worst["mde_pct"], worst["baseline_mean"]))

    print("\nruns_needed is how many runs per group would resolve the effect actually")
    print("observed, at 80% power.  The largest effects seen and their price:")
    ranked = sorted((r for r in records if r["required_n"] == r["required_n"]),
                    key=lambda r: r["required_n"])
    for record in ranked[:5]:
        print("  {:<12} {:<7} {:>+9.4f} ({:>+6.1f}%)  needs {:>3d} runs per group"
              .format(record["variant"], record["metric"], record["diff"],
                      record["diff_pct"], int(record["required_n"])))
    print("  ... current design uses {} runs per group.".format(n_base))

    print("\nwrote " + out_csv)


if __name__ == "__main__":
    main()
