import json
from pathlib import Path
import sys
from unittest.mock import patch

from test_eta import Base, iso, run
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"skills/eta/scripts"))
import job_history as jobs


class History(Base):
    def test_local_forecast_without_prior_eta(self):
        import os
        cwd = os.path.realpath(os.getcwd())
        path = Path(self.tmp.name)/"rollout.jsonl"
        rows = []
        for i, seconds in enumerate([15, 100, 120, 140, 999]):
            rows.append({"type":"event_msg","timestamp":iso(self.now-10),
                         "payload":{"type":"item_completed","item":{
                             "type":"CommandExecution","id":str(i),"status":"completed",
                             "exit_code":int(i == 4),"command":["/bin/zsh","-lc","make test"],
                             "duration":{"secs":seconds,"nanos":0},"cwd":Path(cwd).as_uri()}}})
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        argv = ["estimate", "--agent-min", "9", "--model", "m", "--task-id", "a",
                "--agent", "codex", "--reference", "tests", "--command", "make test",
                "--elapsed-s", "20", "--tail-min", "1", "--wait-min", ".5", "--json"]
        with patch.object(jobs, "project_files", return_value=[path]):
            p = json.loads(run(argv))
            self.assertEqual((p["method"], p["est_s"]), ("job history", 190))
            q = json.loads(run(argv + ["--elapsed-s","130","--event","retry"]))
            self.assertEqual(q["method"], "agent fallback")
        from test_eta import eta
        self.assertEqual(eta.load_jsonl(eta.pred_path())[-1]["event"], "retry")

    def test_remaining_commands_include_tail_and_require_evidence_for_every_step(self):
        argv = ["estimate", "--measured", "--model", "m", "--reference", "verify", "--task-id", "k",
                "--agent", "codex", "--agent-min", "99", "--json", "--no-record",
                "--remaining-command", "build", "--remaining-command", "test", "--elapsed-s", "60"]
        history = {"build": [100, 110, 120], "test": [300, 310, 320], "package": []}
        with patch.object(jobs, "recent_project_files", return_value=[]), patch.object(
                jobs, "local_durations", side_effect=lambda files, cwd, cmd, before: history[cmd]):
            self.assertEqual(json.loads(run(argv))["est_s"], 360)
            self.assertIsNone(json.loads(run(argv + ["--remaining-command", "package"]))["est_s"])

    def test_reject_codex_start_and_foreign_pi_session(self):
        p = Path(self.tmp.name)/"case.jsonl"
        item = dict(type="CommandExecution", status="completed", exit_code=0, cwd="/p",
                    command=["sh","-c","make"], duration={"secs":30,"nanos":0}, id="x")
        rows = [dict(type="event_msg", timestamp=iso(100), payload=dict(type="item_started", item=item))]
        p.write_text(json.dumps(rows[0]))
        self.assertEqual(list(jobs.shell_samples(p, "codex", "/p", "make", 200)), [])
        rows = [dict(type="session", cwd="/other"),
                dict(type="message", timestamp=iso(1), message=dict(role="assistant",
                     content=[dict(type="toolCall", name="bash", id="x", arguments=dict(command="make"))])),
                dict(type="message", timestamp=iso(8), message=dict(role="toolResult", toolCallId="x", isError=False))]
        p.write_text("\n".join(map(json.dumps, rows)))
        self.assertEqual(list(jobs.shell_samples(p, "pi", "/p", "make", 200)), [])
        self.assertEqual([sec for _, sec in jobs.shell_samples(p, "pi", "/other", "make", 200)], [7])

    def test_step_forecast_includes_downstream_work_and_excludes_finished_steps(self):
        # Current step began at 200s and has run for 40s. Its three historical
        # starts differed; the rest of the job after each matched point matters.
        past = [(300, [(3, "tests", 100, 180)]), (360, [(3, "tests", 140, 210)]),
                (320, [(3, "tests", 120, 210)]), (900, [(3, "tests", 100, 139)])]
        past += past[:2]
        result = jobs.ci_samples(past, 240, (3, "tests", 200))
        self.assertEqual(result.remaining, [160, 160, 160, 180, 180])
        self.assertEqual(result.step, "tests")
        self.assertIsNone(jobs.ci_samples(past, 240, (4, "tests", 200)).step)
        self.assertIsNone(jobs.ci_samples(past[:2], 240, (3, "tests", 200)).step)

    def test_step_bound_preserves_in_band_guess_and_tail(self):
        history = jobs.CIHistory([10,20,30,40,50], 240, "running", "Publish", "CI step history", late=True)
        argv = ["estimate","--model","m","--reference","publish","--task-id","t","--ci-job","url",
                "--tail-min","1","--json","--no-record"]
        with patch.object(jobs, "ci_history", return_value=history):
            for guess, expected in [(1.5,90), (1,80), (2,100)]:
                self.assertEqual(json.loads(run(argv + ["--agent-min",str(guess)]))["est_s"],expected)

    def test_early_outliers_use_elapsed_median_or_step_median(self):
        argv = ["estimate", "--model", "m", "--reference", "tests", "--task-id", "t",
                "--ci-job", "url", "--tail-min", "1", "--wait-min", ".5", "--json", "--no-record"]
        for baseline, expected in [(80, 170), (None, 120)]:
            history = jobs.CIHistory([10, 20, 30, 40, 50], 240, "running", "Tests",
                                     "CI step history", elapsed_median=baseline)
            with patch.object(jobs, "ci_history", return_value=history):
                for guess in ["1.6", "4"]:
                    pred = json.loads(run(argv + ["--agent-min", guess]))
                    self.assertEqual(pred["est_s"], expected)
                    self.assertLessEqual(pred["est_s"], pred["high_s"])
                self.assertEqual(json.loads(run(argv + ["--agent-min", "2"]))["est_s"], 120)

    def test_ci_fetch_and_cache(self):
        job = dict(run_id=9, name="tests", status="in_progress", started_at=iso(1000))
        calls = []
        def api(endpoint):
            calls.append(endpoint)
            if endpoint.endswith("/jobs/42"): return job
            if endpoint.endswith("/runs/9"): return dict(head_branch="main",event="push",workflow_id=11)
            if "/workflows/" in endpoint:
                return {"workflow_runs":[dict(id=8,created_at=iso(100),run_attempt=2)]}
            return {"jobs":[dict(name="tests",conclusion="success",started_at=iso(100),
                                 completed_at=iso(160))]}
        url = "https://github.com/o/r/actions/runs/9/job/42"
        with patch.object(jobs, "gh", side_effect=api):
            for _ in range(2):
                result = jobs.ci_history(url,self.tmp.name,1010)
                self.assertEqual((result.remaining,result.elapsed,result.state,result.step), ([50],10,"running",None))
            self.assertEqual(sum("/attempts/2/jobs" in c for c in calls),1)
            job["status"] = "queued"
            self.assertEqual(jobs.ci_history(url,self.tmp.name,1010),jobs.CIHistory([],0,"queued"))
            job.update(status="completed",conclusion="failure")
            self.assertEqual(jobs.ci_history(url,self.tmp.name,1010),jobs.CIHistory([],0,"failure"))
