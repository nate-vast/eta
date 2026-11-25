import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from test_eta import Base, iso, run
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/eta/scripts"))
import job_history as jobs


class Workflow(Base):
    def test_discovers_whole_workflow_and_uses_only_prior_first_attempts(self):
        sha = "a" * 40
        run_row = dict(id=11, head_sha=sha, name="CI", head_branch="main", workflow_id=3,
                       event="push", created_at=iso(self.now-300), status="in_progress", run_attempt=1)
        history = [dict(id=i, created_at=iso(self.now-3000), updated_at=iso(self.now-3000+d),
                        run_attempt=1, conclusion="success") for i, d in enumerate([200, 400, 500, 600])]
        history.extend([dict(history[1], run_attempt=2), dict(history[1], updated_at=iso(self.now))])
        found = [run_row]
        def api(path):
            if "head_sha=" in path:
                self.assertIn(sha, path)
                return dict(workflow_runs=found, total_count=len(found))
            if "/workflows/3/" in path:
                self.assertIn("branch=main", path)
                return dict(workflow_runs=history)
            return run_row
        def git(args, **kwargs):
            output = "git@github.com:example/project.git" if "remote" in args else sha
            return subprocess.CompletedProcess(args, 0, output)
        opts = ["estimate", "--checkout-ci", "--measured", "--tail-min", "0", "--agent-min", "90",
                "--reference", "workflow", "--task-id", "test", "--model", "m", "--no-record", "--json"]
        with patch.object(jobs, "gh", side_effect=api), patch.object(jobs.subprocess, "run", side_effect=git):
            result = json.loads(run(opts))
            self.assertEqual((result["method"], result["job_samples"]), ("workflow history", 3))
            self.assertAlmostEqual(result["est_s"], 200, delta=1)
            found.append(dict(run_row, id=12, name="Other"))
            self.assertIsNone(json.loads(run(opts))["est_s"])
            run_row.update(status="completed", conclusion="failure")
            self.assertEqual(json.loads(run(opts + ["--checkout-ci", "CI"]))["state"], "failure")
