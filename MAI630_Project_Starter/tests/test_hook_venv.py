"""The commit hook runs the tests with the project's .venv interpreter, never with the system Python.

Unit tests import the hook to check which interpreter it chooses; the end-to-end tests run the real hook on a
`git commit` event in a throwaway repository under tmp_path, where nothing is committed.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".claude" / "hooks" / "pre_commit_check.py"
PROJECT = "MAI630_Project_Starter"
EXE = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")


@pytest.fixture(scope="module")
def hook():
    spec = importlib.util.spec_from_file_location("pre_commit_check", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                          # main() only runs as a script, so this reads no stdin
    return module


def fake_interpreter(base):
    path = base / ".venv" / EXE
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    return path


def test_chooses_the_venv_at_the_repository_root(tmp_path, hook):
    repo, project = tmp_path, tmp_path / PROJECT
    project.mkdir()
    expected = fake_interpreter(repo)
    assert hook.venv_python(repo, project) == expected


def test_chooses_the_venv_in_the_project_folder(tmp_path, hook):
    repo, project = tmp_path, tmp_path / PROJECT
    project.mkdir()
    expected = fake_interpreter(project)
    assert hook.venv_python(repo, project) == expected


def test_no_venv_means_no_interpreter_not_the_system_python(tmp_path, hook):
    project = tmp_path / PROJECT
    project.mkdir()
    assert hook.venv_python(tmp_path, project) is None


def test_the_real_project_uses_its_venv(hook):
    repo = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip())
    python = hook.venv_python(repo, ROOT)
    assert python is not None and python.parent.parent.name == ".venv"


def project_with_tests(tmp_path):
    """tmp/repo (Git root) with tmp/repo/MAI630_Project_Starter/tests/; one ordinary staged file."""
    repo = tmp_path / "repo"
    (repo / PROJECT / "tests").mkdir(parents=True)
    (repo / PROJECT / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / PROJECT / "notes.txt").write_text("ok\n")
    subprocess.run(["git", "add", f"{PROJECT}/notes.txt"], cwd=repo, check=True)
    return repo


def run_hook(repo):
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(repo / PROJECT)}
    event = json.dumps({"tool_name": "Bash", "tool_input": {"command": 'git commit -m "test"'}})
    return subprocess.run([sys.executable, str(HOOK)], input=event, env=env, capture_output=True, text=True)


def test_blocks_clearly_without_a_venv(tmp_path):
    r = run_hook(project_with_tests(tmp_path))
    assert r.returncode == 2
    assert "no virtual environment interpreter" in r.stderr and ".venv" in r.stderr
    assert "system Python is not used" in r.stderr


def test_runs_the_tests_with_the_venv_interpreter(tmp_path):
    """A real, empty .venv (no pytest) at the repository root: the hook must run the tests with it, so they fail
    with 'No module named pytest' and the message names that interpreter (the system Python is never tried)."""
    repo = project_with_tests(tmp_path)
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(repo / ".venv")], check=True)
    r = run_hook(repo)
    assert r.returncode == 2
    assert f"run with {repo / '.venv' / EXE}" in r.stderr
    assert "No module named pytest" in r.stderr
