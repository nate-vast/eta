"""Score issued-time step information and blind forecasts against successful finishes."""
import json
import sys
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/eta/scripts"))
from job_history import ci_samples, CIHistory, remaining
import math


def peer_forecast(history, elapsed, active, peers):
    """Select a forecast by leaving out each distinct past job at this checkpoint.

    Each historical peer map contains only jobs finished by that historical
    checkpoint. Target-job finish times are used as labels and never as features.
    """
    prior = ci_samples([(d, steps) for d, steps, _ in history], elapsed, None)
    if not active or len(history) < 6:
        return prior
    number, label, begin = active

    def candidates(train, step_start, observed):
        age = elapsed - step_start
        baseline = remaining([d for d, _, _ in train], elapsed)
        matched, neighbors, scaled = [], [], []
        for duration, steps, others in train:
            for n, name, start, end in steps:
                if (n, name) != (number, label) or end - start <= age:
                    continue
                left = duration - start - age
                matched.append(left)
                ratios = [observed[k] / others[k] for k in observed.keys() & others.keys()
                          if observed[k] > 0 and others[k] > 0]
                if len(ratios) >= 3:
                    distance = median([abs(math.log(r)) for r in ratios])
                    neighbors.append((distance, left))
                    scaled.append(max(0, (duration - start) * median(ratios) - age))
                break
        step = matched if len(matched) >= 3 else baseline
        return [baseline, step, [v for _, v in sorted(neighbors, key=lambda pair: pair[0])[:5]] if len(neighbors) >= 5 else step,
                scaled if len(scaled) >= 3 else step]

    losses = [0.] * 4
    count = 0
    for i, (duration, steps, others) in enumerate(history):
        start = next((start for n, name, start, end in steps
                      if (n, name) == (number, label) and start <= elapsed < end), None)
        if start is None:
            continue
        guesses = candidates(history[:i] + history[i+1:], start, others)
        if len(guesses[0]) < 3:
            continue
        count += 1
        for k, group in enumerate(guesses):
            losses[k] += abs(median(group) - (duration - elapsed))
    options = candidates(history, begin, peers)
    choice = min(range(4), key=losses.__getitem__) if count >= 5 else 0
    methods = ["job history", "CI step history", "similar workflow jobs", "workflow pace"]
    return CIHistory(sorted(options[choice]), elapsed, "running", label if choice else None, methods[choice])


def score(data):
    rows = []
    for q in data["cases"]:
        if q["status"] != "success":
            continue
        history = [(j["duration"], j["steps"]) for j in data["history"] if j["name"] == q["name"]]
        previous = ci_samples(history, q["elapsed"], None)
        candidate = ci_samples(history, q["elapsed"], q["active"])
        if "peers" in q:
            training = [(j["duration"], j["steps"], j["peers"]) for j in data["history"]
                        if j["name"] == q["name"]]
            candidate = peer_forecast(training, q["elapsed"], q["active"], q["peers"])
        rows.append(dict(actual=q["duration"] - q["elapsed"], old=median(previous.remaining),
                         skill=median(candidate.remaining), agent=q["agent_s"], step=candidate.step, method=candidate.method))
    return rows


if __name__ == "__main__":
    data = json.loads((Path(__file__).with_name(sys.argv[1] if len(sys.argv)>1 else "ci-steps.json")).read_text())
    rows = score(data)
    print(f"{len(rows)} successful jobs, {sum(bool(r['step']) for r in rows)} step matches")
    for method in ("old", "skill", "agent"):
        errors = [abs(r[method] - r["actual"]) for r in rows]
        print(f"{method}: mean miss {mean(errors):.2f}s, median {median(errors):.2f}s, worst {max(errors):.2f}s")
    for method in ("old", "agent"):
        gaps = [abs(r["skill"]-r["actual"]) - abs(r[method]-r["actual"]) for r in rows]
        print(f"skill vs {method}: {sum(g < -1e-8 for g in gaps)} closer, "
              f"{sum(abs(g)<=1e-8 for g in gaps)} ties, {sum(g > 1e-8 for g in gaps)} further")
