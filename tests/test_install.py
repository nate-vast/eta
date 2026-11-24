"""Run the public installer and launcher against isolated home directories."""
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix", "the bash installer targets POSIX")
class Installation(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        self.state = self.home / ".eta"
        self.state.mkdir()
        self.env = dict(os.environ, HOME=temp.name, ETA_HOME=str(self.state))

    def install(self, *args, check=True):
        return subprocess.run([str(ROOT / "install.sh"), *args], env=self.env,
                              capture_output=True, check=check)

    def test_foreign_files_and_links_survive(self):
        other = self.home / "other"
        other.write_text("user data")
        for name in ("eta.py", "job_history.py"):
            target = self.state / name
            for linked in (False, True):
                if linked:
                    target.symlink_to(other)
                else:
                    target.write_text("user data")
                for flag in ([], ["--copy"]):
                    self.assertNotEqual(self.install(*flag, check=False).returncode, 0)
                self.install("--uninstall")
                self.assertEqual(target.read_text(), "user data")
                self.assertEqual(target.is_symlink(), linked)
                target.unlink()

    def test_upgrade_switch_modes_and_uninstall_preserve_ledger(self):
        history = self.state / "predictions.jsonl"
        history.write_text("")
        history.chmod(0o644)
        self.state.chmod(0o755)
        for flags in ([], ["--copy"], ["--copy"], [], ["--copy"]):
            self.install(*flags)
            launcher = self.state / "eta.py"
            self.assertEqual(launcher.is_symlink(), not flags)
            subprocess.run([sys.executable, str(launcher), "stats"], env=self.env,
                           check=True, capture_output=True)
            self.assertEqual(stat.S_IMODE(self.state.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(history.stat().st_mode), 0o600)
        self.install("--uninstall")
        self.assertEqual(list(self.state.iterdir()), [history])

    def test_custom_parent_permissions_preserved(self):
        custom = self.home / "custom"
        custom.mkdir(mode=0o755)
        custom.chmod(0o755)
        subprocess.run([sys.executable, str(ROOT / "skills/eta/scripts/eta.py"), "estimate",
                        "--agent-min", "1", "--reference", "test", "--task-id", "test",
                        "--model", "test"], check=True, capture_output=True,
                       env=dict(self.env, ETA_HOME=str(custom)))
        self.assertEqual(stat.S_IMODE(custom.stat().st_mode), 0o755)
        self.assertEqual(stat.S_IMODE((custom / "predictions.jsonl").stat().st_mode), 0o600)
