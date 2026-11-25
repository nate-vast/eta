import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

from test_eta import Base, iso, run
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/eta/scripts"))
import job_history


class DelegateTiming(Base):
    def test_partial_history_stays_separate_from_whole_task(self):
        path = Path(self.tmp.name) / "t3.sqlite"
        db = sqlite3.connect(path)
        db.execute("""CREATE TABLE orchestration_v2_projection_subagents
                   (subagent_id, thread_id, provider, status, payload_json, started_at, completed_at)""")
        def add(index, thread, model, start, end, prefix="command"):
            db.execute("INSERT INTO orchestration_v2_projection_subagents VALUES(?,?,?,?,?,?,?)",
                       (f"node:delegated-task:{prefix}:{index}", thread, "claudeAgent",
                        "running" if end is None else "completed", json.dumps(dict(model=model)),
                        iso(start), iso(end) if end is not None else None))
        for i in range(5):
            add(i, "before", "other", self.now-1000, self.now-800+i)
        add(8, "before", "other", self.now-1000, self.now-975)
        add(9, "before", "other", self.now-1000, self.now-5)
        add(10, "other-thread", "m", self.now-80, None)
        add(11, "target", "m", self.now-100, None)
        db.commit()
        db.close()
        result = job_history.delegate_timings("target", self.now, path)
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0]["n"],result[0]["median_s"],result[0]["same_model"]), (5,102,False))
        with patch.dict("os.environ", T3CODE_THREAD_ID="target"), patch.object(
                job_history, "delegate_timings", return_value=result):
            answer = json.loads(run(["estimate","--measured","--agent-min","90","--model","m",
                                     "--reference","finish integration","--task-id","task","--json","--no-record"]))
        self.assertIsNone(answer["finish"])
        self.assertEqual(answer["delegate_history"], result)
        self.assertIn("across models",answer["message"])
