"""Replay the release CLI with phase-isolated blind agent estimates."""
import contextlib
import io
import json
import sys
from pathlib import Path
from statistics import mean, median
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/eta/scripts"))
import eta
import job_history


def score(bundle, frozen_band=False):
    output = []
    for q in bundle["cases"]:
        if q["status"] != "success":
            continue
        age = q["elapsed"]
        training = [(j["duration"], j["steps"]) for j in bundle["history"] if j["name"] == q["name"]]
        base = job_history.ci_samples(training, age, None)
        stage = job_history.ci_samples(training, age, q["active"])
        for label in ("ordinary", "informed"):
            agent = q[label]
            with patch.object(job_history, "ci_history", return_value=stage), \
                    patch.object(eta, "confirmed_pairs", return_value=[]), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                eta.main(["estimate", "--agent-min", str(agent / 60), "--model", label,
                          "--reference", q["name"], "--task-id", str(q["id"]), "--ci-job", q["url"],
                          "--tail-min", "0", "--json", "--no-record"])
            pred = json.loads(out.getvalue())
            point = pred["est_s"]
            if frozen_band and stage.step:
                point = min(eta.quantile(stage.remaining, .75), max(eta.quantile(stage.remaining, .25), agent))
            output.append(dict(label=label, checkpoint=q["checkpoint"], actual=q["duration"]-age,
                               old=median(base.remaining) if len(base.remaining)>=3 else agent,
                               agent=agent, skill=point, url=q["url"]))
    return output


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "ci-gate.json"
    rows = score(json.loads(Path(__file__).with_name(filename).read_text()), filename == "ci-band.json")
    for label in ("ordinary","informed"):
        for subset in ("all","four minutes","first observed cleanup step"):
            cases = [r for r in rows if r["label"]==label and (subset=="all" or r["checkpoint"]==subset)]
            errors = {k: [abs(r[k]-r["actual"]) for r in cases] for k in ("old","agent","skill")}
            diff = [s-a for s,a in zip(errors["skill"], errors["agent"])]
            print(label,subset,len(cases), {k:round(mean(v),2) for k,v in errors.items()},
                  "wins/ties/losses",sum(d < -1e-8 for d in diff),sum(abs(d)<=1e-8 for d in diff),
                  sum(d > 1e-8 for d in diff))
