#!/usr/bin/env python3
"""Score the immutable forecasts from the accuracy trials, in seconds."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main():
    groups = defaultdict(list)
    source = Path(__file__).with_name("paired-trials.jsonl")
    with source.open() as stream:
        for line in stream:
            record = json.loads(line)
            actual = record["finished"] - record["issued"]
            groups[record["trial"]].append((abs(actual - record["skill_s"]),
                                            abs(actual - record["agent_s"])))
    print("trial                     n  skill MAE  agent MAE  wins/ties/losses")
    for trial, errors in groups.items():
        skill, agent = zip(*errors)
        wins = sum(a < b - 1e-9 for a, b in errors)
        losses = sum(b < a - 1e-9 for a, b in errors)
        ties = len(errors) - wins - losses
        print(f"{trial:<24} {len(errors):2}  {mean(skill):9.3f}  {mean(agent):9.3f}  "
              f"{wins}/{ties}/{losses}")


if __name__ == "__main__":
    main()
