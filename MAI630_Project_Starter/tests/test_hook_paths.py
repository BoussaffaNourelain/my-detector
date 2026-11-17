"""The commit hook when the project is a subfolder of the Git repository, as in my-detector/MAI630_Project_Starter/.

Git names staged files relative to the repository root ("MAI630_Project_Starter/runs/..."), so the hook must apply
its rules to that name and find the file on disk from the repository root. Each test stages one file in a throwaway
repository under tmp_path and runs the real hook on a `git commit` event: nothing is committed.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".claude" / "hooks" / "pre_commit_check.py"
PROJECT = "MAI630_Project_Starter"


def nested_repo(tmp_path, filename, size=1):
    """tmp/repo (the Git root) with the project in tmp/repo/MAI630_Project_Starter; stage `filename` (repo-relative)."""
    repo = tmp_path / "repo"
    (repo / PROJECT).mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    path = repo / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)                                     # a file of `size` bytes (zeros), written quickly
    subprocess.run(["git", "add", "-f", filename], cwd=repo, check=True)
    return repo


def run_hook(repo, command='git commit -m "test"'):
    """Run the hook as Claude Code does, with the project folder (not the Git root) as CLAUDE_PROJECT_DIR."""
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(repo / PROJECT)}
    event = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    return subprocess.run([sys.executable, str(HOOK)], input=event, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("filename", [
    f"{PROJECT}/runs/all_e30_s416_colab/results.csv",          # runs/ inside the project
    f"{PROJECT}/data/plastic_bottles_yolo/images/test/a.jpg",  # data/ inside the project
    f"{PROJECT}/weights/notes.txt",                            # weights/ inside the project
    ".venv/Lib/site.py",                                       # .venv/ at the repository root
])
def test_forbidden_folders_are_blocked(tmp_path, filename):
    r = run_hook(nested_repo(tmp_path, filename))
    assert r.returncode == 2 and filename in r.stderr


def test_model_file_is_blocked(tmp_path):
    filename = f"{PROJECT}/models/best.pt"                     # not in a forbidden folder: blocked by its extension
    r = run_hook(nested_repo(tmp_path, filename))
    assert r.returncode == 2 and filename in r.stderr


def test_file_over_10_mb_is_blocked(tmp_path):
    filename = f"{PROJECT}/docs/big.bin"
    r = run_hook(nested_repo(tmp_path, filename, size=11 * 1024 * 1024))
    assert r.returncode == 2 and filename in r.stderr and "MB" in r.stderr


def test_ordinary_file_passes(tmp_path):
    r = run_hook(nested_repo(tmp_path, f"{PROJECT}/src/detect.py", size=2000))
    assert r.returncode == 0, r.stderr


def test_a_folder_named_data_deeper_in_the_project_is_not_blocked(tmp_path):
    r = run_hook(nested_repo(tmp_path, f"{PROJECT}/src/data/helpers.py"))   # only top-level data/ is forbidden
    assert r.returncode == 0, r.stderr
