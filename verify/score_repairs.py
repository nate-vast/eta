"""Replay the prospective first-failed-test trial against the pinned merge."""
import contextlib
import io
import json
import math
import os
import subprocess
import tempfile
import types
from pathlib import Path
from statistics import mean, median
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "08299d93159a0fd09b04e96a0d18d31ee33ad4b0"


def merged_cli():
    source = subprocess.check_output(
        ["git", "show", BASELINE + ":skills/eta/scripts/eta.py"], cwd=ROOT, text=True)
    module = types.ModuleType("merged_eta")
    exec(compile(source, "merged_eta.py", "exec"), module.__dict__)
    return module


def invoke(cli, timestamp, args):
    with patch.object(cli.time, "time", return_value=timestamp), contextlib.redirect_stdout(io.StringIO()) as out:
        cli.main(args)
    return out.getvalue()


def replay(rows):
    cli = merged_cli()
    durations = []
    with tempfile.TemporaryDirectory() as ledger, patch.dict(os.environ, ETA_HOME=ledger):
        for row in rows:
            assert row["failed"] <= row["issued"] <= row["finish"]
            args = ["estimate", "--agent-min", str(row["agent"] / 60), "--agent", "codex",
                    "--task-id", str(row["task"]), "--model", "gpt-6.2-astra",
                    "--reference", "repair small python function and verify", "--json"]
            old = json.loads(invoke(cli, row["issued"], args))
            assert math.isclose(old["est_s"], row["merged"])
            age = row["issued"] - row["failed"]
            remaining = [duration - age for duration in durations if duration > age]
            prediction = median(remaining) if len(remaining) >= 3 else old["est_s"]
            assert math.isclose(prediction, row["candidate"]) and len(remaining) == row["matches"]
            invoke(cli, row["finish"], ["done", "--id", old["id"]])
            durations.append(row["finish"] - row["failed"])


def report(rows):
    actual = [r["finish"] - r["issued"] for r in rows]
    errors = {k: [abs(r[k] - t) for r, t in zip(rows, actual)] for k in ("agent", "merged", "candidate")}
    for name, values in errors.items():
        print(f"{name:10} mean absolute error {mean(values):.2f}s")
    for name in ("agent", "merged"):
        delta = [s - b for s, b in zip(errors["candidate"], errors[name])]
        print(f"vs {name}: {sum(d < -1e-8 for d in delta)} closer, "
              f"{sum(abs(d) <= 1e-8 for d in delta)} ties, {sum(d > 1e-8 for d in delta)} further")
    print(f"matched history on {sum(r['matches'] >= 3 for r in rows)}/{len(rows)} tasks")


if __name__ == "__main__":
    rows = json.loads(Path(__file__).with_name("repair-trials.json").read_text())
    assert [r["task"] for r in rows] == list(range(1, 10))
    assert all(a["finish"] <= b["failed"] for a, b in zip(rows, rows[1:]))
    replay(rows)
    for title, subset in [("Held out", rows[3:]), ("All, including warmup", rows)]:
        print(title)
        report(subset)
