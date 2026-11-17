#!/usr/bin/env python3
"""Claude Code hook (PreToolUse, Bash): checks every `git commit` that Claude is about to run.

It blocks the commit when the staged files include datasets, training runs, model weights or files over 10 MB,
or when the tests fail (all of them except tests/test_readme.py: you can commit code before your README is ready). Exit code 2 blocks the command, and the message goes back to Claude.
Any other command passes untouched (exit code 0).

The tests run with the project's virtual environment (.venv), not with the `python` that started this hook (that one
can be a system Python without pytest). If no .venv interpreter exists, the commit is blocked with a clear message.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
FORBIDDEN_DIRS = ("data/", "runs/", "weights/", ".venv/")
FORBIDDEN_EXT = (".pt", ".pth", ".onnx")
MAX_MB = 10
TEST_ARGS = ["-m", "pytest", "-q", "-x", "tests", "--ignore=tests/test_readme.py"]


def block(reason: str) -> None:
    print(f"Commit blocked by .claude/hooks/pre_commit_check.py: {reason}", file=sys.stderr)
    sys.exit(2)


def git_root() -> Path:
    """The top of the Git repository. It can be above ROOT (here: my-detector/ above MAI630_Project_Starter/)."""
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True)
    return Path(out.stdout.strip()).resolve() if out.returncode == 0 and out.stdout.strip() else ROOT.resolve()


def git_names(*args: str) -> list[str]:
    """File names from a `git diff --name-only` command, relative to the repository root (NUL-separated, so a name
    with spaces stays one name)."""
    out = subprocess.run(["git", "diff", "--name-only", "-z", *args], cwd=ROOT, capture_output=True, text=True)
    return [n for n in out.stdout.split("\0") if n]


def forbidden_dir(name: str, repo: Path) -> bool:
    """True when the file is in data/, runs/, weights/ or .venv/ at the top of the repository or of the project."""
    rel_repo = name.replace("\\", "/")
    candidates = [rel_repo]
    try:
        candidates.append((repo / name).resolve().relative_to(ROOT.resolve()).as_posix())
    except ValueError:                                      # the file is outside the project folder
        pass
    return any(c.startswith(FORBIDDEN_DIRS) for c in candidates)


def venv_candidates(repo: Path, project: Path) -> list[Path]:
    """Where the .venv interpreter may be, in order: at the repository root, then in the project folder
    (Windows: .venv/Scripts/python.exe, Mac/Linux: .venv/bin/python)."""
    found = []
    for base in dict.fromkeys([Path(repo), Path(project)]):  # the same folder once when the project is the repo root
        found += [base / ".venv" / "Scripts" / "python.exe", base / ".venv" / "bin" / "python"]
    return found


def venv_python(repo: Path, project: Path) -> Path | None:
    """The first .venv interpreter that exists, or None. Never the system Python."""
    return next((p for p in venv_candidates(repo, project) if p.is_file()), None)


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)
    command = event.get("tool_input", {}).get("command", "")
    if "git commit" not in command:
        sys.exit(0)

    # Git gives every name relative to the repository root, which may be above the project folder (ROOT).
    repo = git_root()
    # `git commit -a` / `-am` also commits modified tracked files: check those as well.
    staged = git_names("--cached")
    if " -a" in command or " -am" in command or "--all" in command:
        staged += git_names()

    problems = []
    for name in sorted(set(staged)):
        path = repo / name
        if forbidden_dir(name, repo) or name.endswith(FORBIDDEN_EXT):
            problems.append(f"{name} (datasets, runs and model weights never go to GitHub)")
        elif path.exists() and path.stat().st_size > MAX_MB * 1024 * 1024:
            problems.append(f"{name} ({path.stat().st_size / 1e6:.0f} MB; the limit is {MAX_MB} MB)")
    if problems:
        block("remove these files from the commit with `git restore --staged <file>` and check .gitignore: " + "; ".join(problems))

    if (ROOT / "tests").is_dir():
        python = venv_python(repo, ROOT)
        if python is None:
            looked = "\n".join(f"  {p}" for p in venv_candidates(repo, ROOT))
            block("the tests cannot run: no virtual environment interpreter was found. Looked for:\n" + looked +
                  "\nCreate .venv with Python 3.12 and `pip install -r requirements.txt` (see the README), "
                  "then commit again. The system Python is not used.")
        tests = subprocess.run([str(python), *TEST_ARGS], cwd=ROOT, capture_output=True, text=True)
        if tests.returncode not in (0, 5):                  # 5 = no tests collected
            tail = "\n".join((tests.stdout.strip() or tests.stderr.strip()).splitlines()[-15:])
            block(f"the tests fail (run with {python}). Fix the code first, then commit again.\n" + tail)
    sys.exit(0)


if __name__ == "__main__":
    main()
