"""Chronological replay using the installed estimator's CLI and frozen prior CI timings."""
import contextlib
import io
import json
from pathlib import Path
from statistics import mean, median
import sys
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "skills/eta/scripts"))
import eta
import job_history


def main():
    data = json.loads(Path(__file__).with_name("ci-history.json").read_text())
    errors = []
    hits = 0
    for obs in data["outcomes"]:
        if obs["state"] != "success":
            continue
        duration = job_history.epoch(obs["end"]) - job_history.epoch(obs["start"])
        for c in data["forecasts"]:
            age = c["elapsed"]
            if c["job"] != obs["job"] or age >= duration:
                continue
            historical = job_history.CIHistory(job_history.remaining(c["prior_successful_durations_s"], age), age, "running")
            with patch.object(job_history, "ci_history", return_value=historical), \
                    patch.object(eta, "confirmed_pairs", return_value=[]), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                eta.main(["estimate", "--agent-min", str(c["agent_s"]/60), "--model", "blind",
                          "--reference", obs["job"], "--task-id", str(obs["run"]), "--ci-job",
                          obs["url"], "--tail-min", "0", "--no-record", "--json"])
            pred = json.loads(out.getvalue())
            actual = duration - age
            errors.append((abs(actual-pred["est_s"]), abs(actual-c["agent_s"])))
            hits += pred["low_s"] <= actual <= pred["high_s"]
    print(f"{len(errors)} checkpoints; {hits}/{len(errors)} inside range")
    for name, values in zip(("skill", "agent"), zip(*errors)):
        print(f"{name}: mean {mean(values)/60:.3f} min, median {median(values)/60:.3f} min, "
              f"worst {max(values)/60:.3f} min")


if __name__ == "__main__":
    main()
