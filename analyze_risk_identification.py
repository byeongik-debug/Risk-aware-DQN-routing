"""Can the reported risk-identification experiment tell the five metrics apart?

Reviewer R3-2 asks whether the proposed risk metric identifies risky links.  The
committed answer is results/risk_future_detection_accuracy_summary.csv, written by
run_policy.py's analyze_future_danger_detection.  Its five rows -- Proposed Risk,
Delay Only, Loss Only, Utilization Only, Simple QoS Score -- are identical to sixteen
decimal places: precision 1.0, recall 0.6522916666666666, f1 0.7893846153846154,
tp 122000, fp 0.  Five different score functions cannot agree that exactly, so either
the scores are not actually different or the metric cannot see the difference.

The scores *are* different: top-20% by current risk coincides with top-20% by current
delay in only 23.6% of steps, and with utilization in 0.7%.  The metric is what fails,
and it fails structurally.  Ground truth is the OR of four future top-20% masks plus
future congestion, which always marks 15 or 16 of the 50 links.  Prediction is exactly
the top k=10 by one current metric, and every metric's top 10 lands inside that set in
100% of steps.  With precision pinned at 1.0 and k fixed, tp=k, fp=0, fn=|actual|-k and
tn=L-|actual| follow from |actual| alone -- the confusion matrix carries no information
about which links were chosen.  Any method scores 0.7894.

Underneath that is the environment.  In the dynamic scenario a link's congestion flag
is set at reset and never changes: zero state transitions in 70 steps across all 200
episodes, so no link is ever clean-then-congested.  "Is it congested right now" predicts
future congestion with f1 = 1.0, beating the proposed metric's 0.7894.  There is no
prediction task here to win.

This script reproduces all four diagnostics from the committed link log so the claim
can be checked rather than taken on trust.  It changes nothing in SDN_DQN_new_env or in
run_policy.py; it only re-reads results/dynamic_link_log.csv.
"""

import csv
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from review_stats import mean_std  # noqa: E402

LOG_PATH = os.path.join(BASE_DIR, "results", "dynamic_link_log.csv")
OUT_DIR = os.path.join(BASE_DIR, "review_results")
OUT_CSV = os.path.join(OUT_DIR, "risk_identification_diagnosis.csv")

# run_policy.py's constants, repeated here so this file stands alone.
TOP_RATIO = 0.20
FUTURE_WINDOW = 10

QUANTITIES = ("risk", "delay", "loss", "utilization", "congested")

# The published claim, for the reproduction check in diagnostic 3.  The committed
# summary carries two f1 columns: "f1" is the mean of the per-step values and
# "global_f1" pools the confusion matrix over all steps.  Diagnostic 3 pools, so it
# is global_f1 that it has to reproduce.
PUBLISHED_F1_PER_STEP_MEAN = 0.7893846153846154
PUBLISHED_F1_GLOBAL = 0.789110278161373
PUBLISHED_PRECISION = 1.0
TOLERANCE = 1e-9


def load_log(path=LOG_PATH):
    """-> array[episode, link, time, quantity], plus the axis labels.

    The log is one row per (episode, time, link).  A dense array makes the future
    window a slice instead of a join, which is what run_policy.py pays for with a
    groupby-apply over 710k rows.
    """
    if not os.path.exists(path):
        raise SystemExit("missing link log: {}\nrun run_policy.py's "
                         "save_dynamic_link_log() first".format(path))

    episodes, links, times, records = set(), set(), set(), []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            episode, link, time = int(row["episode"]), row["link"], int(row["time"])
            episodes.add(episode)
            links.add(link)
            times.add(time)
            records.append((episode, link, time,
                            [float(row[q]) for q in QUANTITIES]))

    episodes = sorted(episodes)
    links = sorted(links)
    max_time = max(times)
    ep_index = {e: i for i, e in enumerate(episodes)}
    link_index = {l: i for i, l in enumerate(links)}

    data = np.full((len(episodes), len(links), max_time + 1, len(QUANTITIES)), np.nan)
    for episode, link, time, values in records:
        data[ep_index[episode], link_index[link], time, :] = values

    return data, episodes, links, max_time


def top_k_mask(values, k):
    """run_policy.py's exact_top_k_mask, with k passed in rather than a ratio."""
    order = np.argsort(-np.asarray(values, dtype=float), kind="mergesort")
    mask = np.zeros(len(values), dtype=bool)
    mask[order[:k]] = True
    return mask


def minmax(values):
    low, high = np.nanmin(values), np.nanmax(values)
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def average_precision(score, labels):
    order = np.argsort(-score, kind="mergesort")
    hits = labels[order].astype(float)
    if hits.sum() == 0:
        return np.nan
    precision_at = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision_at[hits.astype(bool)].mean())


def roc_auc(score, labels):
    """Mann-Whitney U form, with midranks so ties cannot inflate the result."""
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum = ranks[labels.astype(bool)].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) /
                 (positives * negatives))


def confusion(predicted, actual):
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())
    tn = int(np.logical_and(~predicted, ~actual).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1)}


def current_scores(current):
    """The five candidate metrics, from the state at time t only."""
    risk, delay, loss, utilization = (current[:, 0], current[:, 1],
                                      current[:, 2], current[:, 3])
    return {
        "Proposed Risk": risk,
        "Delay Only": delay,
        "Loss Only": loss,
        "Utilization Only": utilization,
        "Simple QoS Score": (minmax(delay) + minmax(loss) + minmax(utilization)) / 3.0,
    }


# ---------------------------------------------------------------- diagnostic 1

def congestion_stability(data, max_time, rows):
    """Does a link's congestion flag ever change inside an episode?"""
    congested = data[:, :, :, QUANTITIES.index("congested")] > 0
    changes = (congested[:, :, 1:] != congested[:, :, :-1]).sum(axis=2).ravel()

    clean_to_congested = 0
    steps = 0
    for episode in range(congested.shape[0]):
        for t in range(0, max_time - FUTURE_WINDOW + 1):
            now = congested[episode, :, t]
            later = congested[episode, :, t + 1:t + 1 + FUTURE_WINDOW].any(axis=1)
            clean_to_congested += int((~now & later).sum())
            steps += 1

    print("=== 1. is the congestion set static? ===")
    print("  state changes per (episode, link) over {} steps: mean {:.3f}, "
          "max {}, never-changing {:.1f}%".format(
              max_time, changes.mean(), changes.max(),
              (changes == 0).mean() * 100))
    print("  links that are clean at t and congested within t+1..t+{}: {} "
          "in {} steps".format(FUTURE_WINDOW, clean_to_congested, steps))
    if clean_to_congested == 0:
        print("  -> no link ever becomes congested, so there is nothing to predict.")

    rows.append({"diagnostic": "congestion_stability", "subject": "state_changes",
                 "metric": "mean_per_link", "value": changes.mean()})
    rows.append({"diagnostic": "congestion_stability", "subject": "state_changes",
                 "metric": "pct_never_changing", "value": (changes == 0).mean() * 100})
    rows.append({"diagnostic": "congestion_stability",
                 "subject": "clean_to_congested_transitions",
                 "metric": "count", "value": float(clean_to_congested)})
    return congested


# ---------------------------------------------------------------- diagnostic 2

def variance_decomposition(data, rows):
    """How much of each quantity's spread is over time, and how much across links?

    A temporal risk window can only help if a link's own values move over time.  If
    the spread is between links instead, the window has nothing to integrate.
    """
    print("\n=== 2. temporal vs cross-link variation ===")
    print("  {:<13}{:>16}{:>16}{:>9}".format(
        "quantity", "SD within link", "SD across links", "ratio"))
    for index, name in enumerate(QUANTITIES):
        values = data[:, :, :, index]
        within = float(np.nanmean(np.nanstd(values, axis=2)))
        across = float(np.nanmean(np.nanstd(np.nanmean(values, axis=2), axis=1)))
        ratio = within / across if across else float("nan")
        print("  {:<13}{:>16.6f}{:>16.6f}{:>9.3f}".format(
            name, within, across, ratio))
        rows.append({"diagnostic": "variance_decomposition", "subject": name,
                     "metric": "sd_within_link_over_time", "value": within})
        rows.append({"diagnostic": "variance_decomposition", "subject": name,
                     "metric": "sd_across_links", "value": across})
        rows.append({"diagnostic": "variance_decomposition", "subject": name,
                     "metric": "ratio_within_over_across", "value": ratio})
    print("  -> a ratio well under 1 means the link's identity, not the clock,")
    print("     sets its state.  congested is exactly 0: it never moves at all.")


# ---------------------------------------------------------------- diagnostic 3

def published_metric_is_degenerate(data, congested, max_time, links, rows):
    """Reproduce the published numbers, then show they ignore the method.

    Also scored: a random top-k pick, which shows the metrics do beat chance, and
    "is it congested right now", which shows they lose to the trivial baseline.
    """
    n_links = len(links)
    k = max(1, int(np.ceil(n_links * TOP_RATIO)))
    rng = np.random.default_rng(0)

    totals = {}
    subset_steps = {}
    actual_sizes = []
    steps = 0

    for episode in range(data.shape[0]):
        for t in range(0, max_time - FUTURE_WINDOW + 1):
            current = data[episode, :, t, :]
            future = data[episode, :, t + 1:t + 1 + FUTURE_WINDOW, :]
            future_mean = {name: np.nanmean(future[:, :, i], axis=1)
                           for i, name in enumerate(QUANTITIES[:4])}
            future_congested = np.nanmax(
                future[:, :, QUANTITIES.index("congested")], axis=1) > 0

            actual = (top_k_mask(future_mean["delay"], k) |
                      top_k_mask(future_mean["loss"], k) |
                      top_k_mask(future_mean["utilization"], k) |
                      top_k_mask(future_mean["risk"], k) |
                      future_congested)
            actual_sizes.append(int(actual.sum()))
            steps += 1

            predictions = {name: top_k_mask(score, k)
                           for name, score in current_scores(current).items()}
            random_mask = np.zeros(n_links, dtype=bool)
            random_mask[rng.choice(n_links, k, replace=False)] = True
            predictions["Random top-k"] = random_mask
            predictions["Congested now"] = congested[episode, :, t]

            for name, predicted in predictions.items():
                counts = confusion(predicted, actual)
                bucket = totals.setdefault(name, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
                for key in bucket:
                    bucket[key] += counts[key]
                subset_steps[name] = subset_steps.get(name, 0) + int(counts["fp"] == 0)

    sizes = np.array(actual_sizes, dtype=float)
    print("\n=== 3. the published metric cannot rank the methods ===")
    print("  {} links, k = {} predicted per step, {} steps".format(
        n_links, k, steps))
    print("  ground-truth danger set: {:.2f}/{} links ({:.2f}%), "
          "min {:.0f}, max {:.0f}".format(
              sizes.mean(), n_links, sizes.mean() / n_links * 100,
              sizes.min(), sizes.max()))
    print("  {:<19}{:>11}{:>9}{:>9}{:>10}{:>15}".format(
        "method", "precision", "recall", "f1", "fp", "steps w/ fp=0"))

    for name, bucket in totals.items():
        tp, fp, fn = bucket["tp"], bucket["fp"], bucket["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        print("  {:<19}{:>11.4f}{:>9.4f}{:>9.4f}{:>10d}{:>14.1f}%".format(
            name, precision, recall, f1, fp, subset_steps[name] / steps * 100))
        rows.append({"diagnostic": "detection_metric", "subject": name,
                     "metric": "precision", "value": precision})
        rows.append({"diagnostic": "detection_metric", "subject": name,
                     "metric": "recall", "value": recall})
        rows.append({"diagnostic": "detection_metric", "subject": name,
                     "metric": "f1", "value": f1})
        rows.append({"diagnostic": "detection_metric", "subject": name,
                     "metric": "false_positives", "value": float(fp)})

    candidates = [name for name in totals
                  if name not in ("Random top-k", "Congested now")]
    f1s = []
    for name in candidates:
        tp, fp, fn = totals[name]["tp"], totals[name]["fp"], totals[name]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall))
    spread = max(f1s) - min(f1s)
    print("  spread across the {} candidate metrics: {:.2e}".format(
        len(candidates), spread))

    published = f1s[0]
    if abs(published - PUBLISHED_F1_GLOBAL) < TOLERANCE:
        print("  reproduces the committed summary global_f1 {:.15f} (its per-step mean"
              " f1 is {:.16f})".format(PUBLISHED_F1_GLOBAL,
                                       PUBLISHED_F1_PER_STEP_MEAN))
    else:
        print("  [warning] got f1 {:.16f}, committed global_f1 is {:.15f}".format(
            published, PUBLISHED_F1_GLOBAL))

    print("  -> precision is pinned at 1.0 for every candidate, so with k fixed the")
    print("     whole confusion matrix follows from |actual|, never from the ranking.")
    rows.append({"diagnostic": "detection_metric", "subject": "candidate_spread",
                 "metric": "f1_max_minus_min", "value": spread})


# ---------------------------------------------------------------- diagnostic 4

def alternative_ground_truths(data, max_time, links, rows):
    """Labels that do discriminate -- and why each still answers the wrong question.

    Tightening the label does separate the metrics, but then every metric wins on the
    label built from itself: current delay predicts future delay, current risk predicts
    future risk.  That is autocorrelation.  A label derived from the candidate metrics
    cannot adjudicate between them.
    """
    n_links = len(links)
    definitions = ("future_congested", "future_top10pct_delay", "future_top5pct_risk")
    scores = {name: {d: {"ap": [], "auc": []} for d in definitions}
              for name in ("Proposed Risk", "Delay Only", "Loss Only",
                           "Utilization Only", "Simple QoS Score")}
    positives = {d: [] for d in definitions}

    for episode in range(data.shape[0]):
        for t in range(0, max_time - FUTURE_WINDOW + 1):
            current = data[episode, :, t, :]
            future = data[episode, :, t + 1:t + 1 + FUTURE_WINDOW, :]
            future_mean = {name: np.nanmean(future[:, :, i], axis=1)
                           for i, name in enumerate(QUANTITIES[:4])}
            labels = {
                "future_congested": np.nanmax(
                    future[:, :, QUANTITIES.index("congested")], axis=1) > 0,
                "future_top10pct_delay": top_k_mask(
                    future_mean["delay"], max(1, int(np.ceil(n_links * 0.10)))),
                "future_top5pct_risk": top_k_mask(
                    future_mean["risk"], max(1, int(np.ceil(n_links * 0.05)))),
            }
            candidates = current_scores(current)
            for definition, label in labels.items():
                positives[definition].append(int(label.sum()))
                as_float = label.astype(float)
                for name, score in candidates.items():
                    scores[name][definition]["ap"].append(
                        average_precision(score, as_float))
                    scores[name][definition]["auc"].append(roc_auc(score, as_float))

    print("\n=== 4. alternative labels: discriminating, but circular ===")
    for definition in definitions:
        count = np.array(positives[definition], dtype=float)
        print("  label {} -- {:.2f}/{} positives ({:.1f}%)".format(
            definition, count.mean(), n_links, count.mean() / n_links * 100))
        print("    {:<19}{:>10}{:>10}".format("method", "AvgPrec", "AUC"))
        for name in scores:
            ap = float(np.nanmean(scores[name][definition]["ap"]))
            auc = float(np.nanmean(scores[name][definition]["auc"]))
            print("    {:<19}{:>10.4f}{:>10.4f}".format(name, ap, auc))
            rows.append({"diagnostic": "alternative_label",
                         "subject": "{} | {}".format(definition, name),
                         "metric": "average_precision", "value": ap})
            rows.append({"diagnostic": "alternative_label",
                         "subject": "{} | {}".format(definition, name),
                         "metric": "roc_auc", "value": auc})
    print("  -> future_congested is separated perfectly (AUC 1.0) by every metric,")
    print("     because the congested set never changes.  The tighter labels each")
    print("     reward the metric they were built from, which measures persistence,")
    print("     not whether the proposed formula is the better risk signal.")


def main():
    data, episodes, links, max_time = load_log()
    print("link log: {} episodes x {} links x {} time steps".format(
        len(episodes), len(links), max_time + 1))

    rows = []
    congested = congestion_stability(data, max_time, rows)
    variance_decomposition(data, rows)
    published_metric_is_degenerate(data, congested, max_time, links, rows)
    alternative_ground_truths(data, max_time, links, rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["diagnostic", "subject",
                                                "metric", "value"])
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["value"] = "{:.10f}".format(row["value"])
            writer.writerow(row)

    print("\nconclusion: R3-2 is not answered by the committed experiment.  The metric")
    print("cannot rank the five candidates, and the environment contains no clean-to-")
    print("congested transition for a risk signal to anticipate.")
    print("\nwrote " + OUT_CSV)


if __name__ == "__main__":
    main()
