"""Turn results/scores.jsonl into the robustness report.

    python analyze.py   # writes results/report.md and results/summary.json

Flip rate = how often a verdict on a noisy transcript disagrees with a verdict on the clean one.
The judge disagrees with itself too, so every number is reported against its clean-vs-clean
flip rate (the noise floor): excess = flip(noisy vs clean) - flip(clean vs clean).
"""
import json
import random
from itertools import combinations
from pathlib import Path

import qa

RESULTS = qa.RESULTS
NOISY = [w for w in qa.WERS if w > 0]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def index(rows):
    """{(id, wer, seed): row}"""
    return {(r["id"], r["wer"], r["seed"]): r for r in rows}


def pair_flips(by, cid, crit, wer):
    """Per-call list of 0/1 disagreements. wer=0: clean repeat pairs; else every noisy draw vs every clean repeat."""
    clean = [by[(cid, 0.0, s)]["verdict"][crit] for s in qa.SEEDS if (cid, 0.0, s) in by]
    if wer == 0:
        return [int(a != b) for a, b in combinations(clean, 2)]
    noisy = [by[(cid, wer, s)]["verdict"][crit] for s in qa.SEEDS if (cid, wer, s) in by]
    return [int(a != b) for a in noisy for b in clean]


def flip_rate(by, ids, crit, wer):
    return mean(f for cid in ids for f in pair_flips(by, cid, crit, wer))


def excess_ci(by, ids, crit, wer, n_boot=2000, seed=0):
    """Excess flip rate with a 95% bootstrap interval over calls (calls are the independent unit)."""
    rng = random.Random(seed)
    point = flip_rate(by, ids, crit, wer) - flip_rate(by, ids, crit, 0.0)
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(ids) for _ in ids]
        boots.append(flip_rate(by, sample, crit, wer) - flip_rate(by, sample, crit, 0.0))
    boots.sort()
    return point, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def accuracy(rows, calls, crit, wer):
    gold = {c["id"]: c["gold"][crit] for c in calls}
    return mean(int(r["verdict"][crit] == gold[r["id"]]) for r in rows if r["wer"] == wer)


def kappa(rows, calls, wer):
    """Cohen's kappa of judge vs gold over every (call, criterion, seed) at one noise level."""
    gold = {c["id"]: c["gold"] for c in calls}
    pairs = [(r["verdict"][n], gold[r["id"]][n]) for r in rows if r["wer"] == wer for n in qa.CRITERIA]
    po = mean(int(a == b) for a, b in pairs)
    pa, pb = mean(a for a, _ in pairs), mean(b for _, b in pairs)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe)


def direction(rows, calls, wer):
    """(false-fail rate on calls that pass, false-pass rate on calls that fail), pooled over criteria."""
    gold = {c["id"]: c["gold"] for c in calls}
    at = [(r["verdict"][n], gold[r["id"]][n]) for r in rows if r["wer"] == wer for n in qa.CRITERIA]
    return mean(v == 0 for v, g in at if g == 1), mean(v == 1 for v, g in at if g == 0)


def evidence_split(rows, calls):
    """Flip rate vs the clean majority, split by whether noise touched the turn that decides the criterion."""
    by_id = {c["id"]: c for c in calls}
    conds = {(c["id"], c["wer"], c["seed"]): c for c in qa.conditions(calls)}
    by = index(rows)
    hit, miss = [], []
    for r in rows:
        if r["wer"] == 0:
            continue
        ops = conds[(r["id"], r["wer"], r["seed"])]["ops"]
        touched = {o["turn"] for o in ops}
        for crit in qa.CRITERIA:
            ev = set(by_id[r["id"]]["evidence"][crit])
            clean = [by[(r["id"], 0.0, s)]["verdict"][crit] for s in qa.SEEDS if (r["id"], 0.0, s) in by]
            if not ev or not clean:
                continue
            majority = int(sum(clean) * 2 > len(clean))
            (hit if ev & touched else miss).append(int(r["verdict"][crit] != majority))
    return mean(hit), len(hit), mean(miss), len(miss)


def build(rows, calls):
    by, ids = index(rows), sorted({r["id"] for r in rows})
    out = {"n_rows": len(rows), "n_calls": len(ids), "criteria": {}}
    for crit in qa.CRITERIA:
        row = {"floor": flip_rate(by, ids, crit, 0.0), "clean_accuracy": accuracy(rows, calls, crit, 0.0)}
        for w in NOISY:
            point, lo, hi = excess_ci(by, ids, crit, w)
            row[f"excess@{w:.2f}"] = point
            row[f"ci@{w:.2f}"] = [lo, hi]
            row[f"accuracy@{w:.2f}"] = accuracy(rows, calls, crit, w)
        out["criteria"][crit] = row
    out["kappa"] = {f"{w:.2f}": kappa(rows, calls, w) for w in qa.WERS}
    out["direction"] = {f"{w:.2f}": dict(zip(("false_fail", "false_pass"), direction(rows, calls, w))) for w in qa.WERS}
    gold_r = {c["id"]: c["gold"]["resolution_accuracy"] for c in calls}
    out["resolution_false_fail"] = {f"{w:.2f}": mean(r["verdict"]["resolution_accuracy"] == 0 for r in rows
                                                     if r["wer"] == w and gold_r[r["id"]] == 1) for w in qa.WERS}
    out["grounded"] = {f"{w:.2f}": mean(v for r in rows if r["wer"] == w for v in r["grounded"].values())
                       for w in qa.WERS}
    out["measured_wer"] = {f"{w:.2f}": mean(r["measured_wer"] for r in rows if r["wer"] == w) for w in qa.WERS}
    h, nh, m, nm = evidence_split(rows, calls)
    out["evidence"] = {"flip_if_decisive_turn_hit": h, "n_hit": nh, "flip_if_not_hit": m, "n_not_hit": nm}
    return out


def pct(x):
    return f"{100 * x:+.0f}" if x == x else "n/a"


def report(s):
    head = "| criterion | clean accuracy | self-flip (floor) | " + " | ".join(
        f"excess flips @ {int(100 * w)}% WER" for w in NOISY) + " |"
    lines = [head, "|" + "---|" * (3 + len(NOISY))]
    for crit, r in sorted(s["criteria"].items(), key=lambda kv: -kv[1][f"excess@{NOISY[-1]:.2f}"]):
        cells = [f"{pct(r[f'excess@{w:.2f}'])} pts ({pct(r[f'ci@{w:.2f}'][0])} to {pct(r[f'ci@{w:.2f}'][1])})"
                 for w in NOISY]
        lines.append(f"| {crit} | {100 * r['clean_accuracy']:.0f}% | {100 * r['floor']:.0f}% | " + " | ".join(cells) + " |")
    acc = ["| criterion | " + " | ".join(f"{int(100 * w)}% WER" for w in qa.WERS) + " |", "|" + "---|" * (1 + len(qa.WERS))]
    for crit, r in s["criteria"].items():
        vals = [r["clean_accuracy"]] + [r[f"accuracy@{w:.2f}"] for w in NOISY]
        acc.append(f"| {crit} | " + " | ".join(f"{100 * v:.0f}%" for v in vals) + " |")
    ev = s["evidence"]
    return "\n".join([
        "# Results", "",
        f"{s['n_rows']} scored transcripts: {s['n_calls']} calls x {len(qa.WERS)} noise levels x {len(qa.SEEDS)} seeds.", "",
        "## Excess verdict flips per criterion (noisy vs clean, minus the judge's clean-vs-clean floor)",
        "Percentage points, 95% bootstrap interval over calls. Sorted by the 30% column.", "",
        *lines, "",
        "## Accuracy against the gold labels", "", *acc, "",
        "Cohen's kappa vs gold: " + ", ".join(f"{int(100 * float(w))}% WER {k:.2f}" for w, k in s["kappa"].items()), "",
        "## Which way the errors go (all criteria pooled)",
        "False fail = the agent did it right and the judge marked it wrong. False pass = the reverse.", "",
        "| WER | false fail | false pass | correct resolutions marked wrong |", "|---|---|---|---|",
        *[f"| {int(100 * float(w))}% | {100 * d['false_fail']:.0f}% | {100 * d['false_pass']:.0f}% | "
          f"{100 * s['resolution_false_fail'][w]:.0f}% |" for w, d in s["direction"].items()], "",
        "Quotes found verbatim in the transcript the judge saw: "
        + ", ".join(f"{int(100 * float(w))}% WER {100 * g:.0f}%" for w, g in s["grounded"].items()), "",
        "Measured WER of the noised transcripts: "
        + ", ".join(f"target {int(100 * float(w))}% -> {100 * m:.1f}%" for w, m in s["measured_wer"].items()), "",
        "## Where the flips come from",
        f"When noise touched the turn that decides a criterion, the verdict left the clean majority "
        f"{100 * ev['flip_if_decisive_turn_hit']:.0f}% of the time (n={ev['n_hit']}); when it did not, "
        f"{100 * ev['flip_if_not_hit']:.0f}% (n={ev['n_not_hit']}).", ""])


def main():
    rows = [json.loads(l) for l in (RESULTS / "scores.jsonl").read_text().splitlines() if l.strip()]
    s = build(rows, qa.load_transcripts())
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=2))
    (RESULTS / "report.md").write_text(report(s))
    print(report(s))


if __name__ == "__main__":
    main()
