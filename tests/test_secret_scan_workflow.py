"""Check the PR scan boundary with temporary Git histories and synthetic keys."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/secret-scan.yml"


def command(name):
    match = re.search(r"(?m)^      - name: " + re.escape(name)
                      + r"\n        run: (\||>-)\n((?:          .*\n)+)", WORKFLOW.read_text())
    if match is None:
        raise AssertionError("missing workflow command: " + name)
    script = textwrap.dedent(match.group(2))
    return script if match.group(1) == "|" else script.replace("\n", " ")


class SecretScanTrigger(unittest.TestCase):
    def test_pr_head_uses_the_trusted_workflow_with_no_persisted_token(self):
        source = WORKFLOW.read_text()
        trigger = source.split("on:\n", 1)[1].split("\npermissions:", 1)[0]
        self.assertRegex(trigger, r"pull_request_target:.*\n    branches: \[deepseek-v41-tp2\]")
        self.assertIn("types: [opened, synchronize, reopened]", trigger)
        self.assertIn("group: secret-scan-${{ github.event.pull_request.number || github.ref }}", source)
        self.assertIn("ref: ${{ github.event.pull_request.head.sha || github.sha }}", source)
        self.assertIn("persist-credentials: false", source)
        self.assertIn("allow-unsafe-pr-checkout: true", source)
        self.assertIn("permissions:\n  contents: read\n", source)
        self.assertNotIn("${{ secrets.", source)


@unittest.skipUnless(shutil.which("gitleaks") and shutil.which("bash") and shutil.which("git"),
                     "needs Gitleaks, Bash and Git")
class SecretScanBoundary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scan = self.root / "scan"
        self.scan.mkdir()
        self.runner = self.root / "runner"
        self.runner.mkdir()
        self.env = {**os.environ, "RUNNER_TEMP": str(self.runner)}
        os.symlink(shutil.which("gitleaks"), self.runner / "gitleaks")
        install = command("Install gitleaks")
        self.shell('cd "$RUNNER_TEMP"\n' + install[install.index("printf "):])
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Scan Test")
        self.git("config", "user.email", "scan@example.invalid")
        self.write("readme.txt", "A clean test repository.\n")
        self.commit("Clean base")

    def shell(self, script, expected=0):
        result = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=self.root,
                                env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, expected, result.stderr)
        return result

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.scan), *args], check=True,
                              text=True, capture_output=True).stdout.strip()

    def write(self, name, value):
        (self.scan / name).write_text(value)

    def commit(self, message):
        self.git("add", "--all")
        self.git("commit", "-qm", message)

    def key(self):
        return "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"

    def scan_history(self, expected):
        self.shell(command("Refuse a .gitleaksignore in the checkout"))
        self.shell(command("Ignore the .gitattributes of the checkout"))
        script = command("gitleaks (all commits of this branch)")
        self.shell(script + ' --report-format json --report-path "$RUNNER_TEMP/report.json"', expected)
        findings = json.loads((self.runner / "report.json").read_text())
        if expected:
            self.assertTrue(any(item["RuleID"] == "github-pat" for item in findings))
            self.assertNotIn(self.key(), json.dumps(findings))
        else:
            self.assertEqual(findings, [])
        return findings

    def test_clean_history_passes(self):
        self.scan_history(0)

    def test_candidate_policy_attributes_and_allow_comments_cannot_hide_a_key(self):
        self.write("credentials.txt", "api_key = '" + self.key() + "' # gitleaks:allow\n")
        self.write(".gitleaks.toml", "[extend]\nuseDefault = true\n[[allowlists]]\nregexes = ['''.*''']\n")
        self.write(".gitattributes", "* -diff\n")
        self.write("gitleaks", '#!/bin/sh\ntouch "$RUNNER_TEMP/candidate-ran"\n')
        (self.scan / "gitleaks").chmod(0o755)
        self.commit("Candidate scan inputs")
        self.scan_history(1)
        self.assertFalse((self.runner / "candidate-ran").exists())

    def test_removing_a_key_from_the_head_does_not_hide_its_history(self):
        self.write("credentials.txt", "api_key = '" + self.key() + "'\n")
        self.commit("Synthetic key")
        (self.scan / "credentials.txt").unlink()
        self.commit("Remove synthetic key")
        self.scan_history(1)

    def test_a_key_added_only_in_a_merge_resolution_is_scanned(self):
        self.git("branch", "other")
        self.write("left.txt", "Left change.\n")
        self.commit("Left")
        self.git("checkout", "-q", "other")
        self.write("right.txt", "Right change.\n")
        self.commit("Right")
        self.git("checkout", "-q", "main")
        self.git("merge", "--no-commit", "--no-ff", "other")
        self.write("credentials.txt", "api_key = '" + self.key() + "'\n")
        self.commit("Merge resolution with synthetic key")
        self.scan_history(1)

    def test_repo_ignore_files_and_symlinks_are_refused(self):
        path = self.scan / ".gitleaksignore"
        for kind in ("file", "link", "dangling"):
            with self.subTest(kind=kind):
                if kind == "file":
                    path.write_text("ignored fingerprint\n")
                else:
                    target = self.root / "ignore-target"
                    if kind == "link":
                        target.write_text("ignored fingerprint\n")
                    elif target.exists():
                        target.unlink()
                    path.symlink_to(target)
                self.shell(command("Refuse a .gitleaksignore in the checkout"), 1)
                path.unlink()
