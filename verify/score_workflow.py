"""Replay the five-minute checkpoint of the latest 20 successful workflow runs."""
import json
from pathlib import Path
from statistics import mean, median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/eta/scripts"))
from job_history import epoch, remaining

runs = [(epoch(start), epoch(finish), attempt)
        for _, start, finish, attempt in json.loads(Path(__file__).with_name("workflow-history.json").read_text())["runs"]]
errors = []
actuals = []
for index in range(80, len(runs)):
    start, finish, attempt = runs[index]
    elapsed = 300
    actual = finish - start - elapsed
    if actual <= 0 or attempt != 1:
        continue
    past = [b - a for a, b, n in runs[:index] if n == 1 and b < start]
    samples = remaining(past, elapsed)
    if len(samples) >= 3:
        actuals.append(actual)
        errors.append(median(samples) - actual)
print(f"{len(errors)} forecasts, median actual remaining {median(actuals):.0f}s, "
      f"mean absolute error {mean(map(abs, errors)):.2f}s, {sum(e > 0 for e in errors)} too long")
