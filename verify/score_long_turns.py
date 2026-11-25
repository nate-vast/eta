"""Replay five-minute ETA check-ins on local Codex transcripts without exporting their contents."""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/eta/scripts"))
import job_history

def main():
    groups = defaultdict(dict)
    base = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    for path in base.glob("sessions/*/*/*/rollout-*.jsonl"):
        try:
            with path.open() as stream:
                cwd = job_history.directory(json.loads(stream.readline())["payload"]["cwd"])
            for record in job_history.completed_codex_turns(path, cwd):
                groups[cwd, record[1]][record[0]] = record
        except (ValueError, OSError, KeyError):
            continue
    age = 300
    errors = []
    actuals = []
    eligible = 0
    for (_, model), rows in groups.items():
        for _, _, start, end in rows.values():
            if end - start <= age:
                continue
            eligible += 1
            samples = job_history.turn_remaining(rows.values(), model, age, start + age)
            if len(samples) >= 5:
                actual = end - start - age
                errors.append(median(samples) - actual)
                actuals.append(actual)
    print(f"At five minutes: {len(errors)} forecasts from {eligible} eligible turns.")
    if errors:
        print(f"Median actual remaining {median(actuals):.0f}s; mean absolute error "
              f"{mean(map(abs, errors)):.0f}s; overestimates {sum(e > 0 for e in errors)}/{len(errors)}.")

if __name__ == "__main__":
    main()
