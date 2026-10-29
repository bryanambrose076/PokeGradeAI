"""Run the fix agent (Claude Code in Docker) on the tune misses, then score its candidates with the harness.

Run: python -m scripts.export_labels, then python -m scripts.fix_agent [--model opus] [--effort high]
Needs CLAUDE_CODE_OAUTH_TOKEN in .env (from `claude setup-token`). Writes experiments/<run id>/.
"""
import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

from scripts import harness

REPO = Path(__file__).resolve().parent.parent
# The know-how lives in the fix-matcher skill (.claude/skills/), which reaches the workspace through git
TASK = "/fix-matcher Diagnose why the matcher misses these scans and write 2-3 candidate fixes."


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


# A clean copy of the committed code as a fresh git repo on branch "base", plus the tune photos.
# Built from git, so .env, venv and untracked files never get in; .mcp.json is removed so no MCP server can start.
def make_workspace(work):
    work.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(work)], input=archive.stdout, check=True)
    (work / ".mcp.json").unlink(missing_ok=True)
    shutil.copytree(REPO / "eval" / "tune", work / "eval" / "tune")
    for cmd in (["init", "-q", "-b", "base"], ["add", "-A"],
                ["-c", "user.name=fix-agent", "-c", "user.email=agent@localhost", "commit", "-q", "-m", "base"]):
        run(["git", "-C", str(work), *cmd], check=True)
    # Ignore what the evals write, so it can't end up in a candidate's commit
    (work / ".git" / "info" / "exclude").write_text("eval/crops/\n__pycache__/\n.pytest_cache/\n")


# Each branch fix-N the agent made becomes candidates/fix-N.diff, with its commit message as the description
def collect_candidates(work, run_dir):
    (run_dir / "candidates").mkdir()
    branches = run(["git", "-C", str(work), "branch", "--list", "fix-*", "--format=%(refname:short)"]).stdout.split()
    for branch in branches:
        diff = run(["git", "-C", str(work), "diff", "base", branch]).stdout
        notes = run(["git", "-C", str(work), "log", "--format=%s", f"base..{branch}"]).stdout
        if diff:
            (run_dir / "candidates" / f"{branch}.diff").write_text(diff)
            (run_dir / "candidates" / f"{branch}.txt").write_text(notes)
    return branches


def main(model, effort, budget, minutes):
    token = dotenv_values(REPO / ".env").get("CLAUDE_CODE_OAUTH_TOKEN")
    if not token:
        raise SystemExit("CLAUDE_CODE_OAUTH_TOKEN isn't in .env; run `claude setup-token` and add it")
    if not (REPO / "eval" / "tune" / "labels.csv").exists():
        raise SystemExit("No tune labels; run python -m scripts.export_labels first")

    run_dir = REPO / "experiments" / datetime.now().strftime("%Y-%m-%d-%H%M")
    work = run_dir / "workspace"
    make_workspace(work)

    print("Building images...")
    for cmd in (["docker", "build", "-q", "-t", "pokegrade", str(REPO)],
                ["docker", "build", "-q", "-f", str(REPO / "Dockerfile.agent"), "-t", "pokegrade-agent", str(REPO)]):
        build = run(cmd)
        if build.returncode:
            raise SystemExit(f"Build failed:\n{build.stderr[-2000:]}")

    # The token is the container's only credential, passed by name so it never appears on the command line.
    # Reference images are mounted read-only for comparing; no MCP servers, no permission prompts inside the sandbox.
    print(f"Running the fix agent ({model}, {effort} effort, up to ${budget} of usage, {minutes} min)...")
    name = f"fix-agent-{run_dir.name}"
    try:
        agent = run(["docker", "run", "--rm", "--name", name, "-e", "CLAUDE_CODE_OAUTH_TOKEN",
                     "-v", f"{work}:/work", "-v", f"{REPO / 'data' / 'images'}:/work/data/images:ro",
                     "pokegrade-agent",
                     "claude", "-p", TASK, "--model", model, "--effort", effort,
                     "--permission-mode", "bypassPermissions", "--strict-mcp-config",
                     "--max-budget-usd", str(budget), "--output-format", "json"],
                    env={**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": token}, timeout=minutes * 60)
    except subprocess.TimeoutExpired:
        # Stopping the docker client doesn't stop the container, so kill it; branches made so far still get scored
        run(["docker", "kill", name])
        agent = subprocess.CompletedProcess([], 1, json.dumps({"result": f"Stopped after {minutes} minutes"}), "")
    (run_dir / "agent.json").write_text(agent.stdout or json.dumps({"error": agent.stderr[-2000:]}))
    try:
        summary = json.loads(agent.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"Agent run failed:\n{agent.stderr[-2000:]}")
    print(f"Agent finished: {summary.get('num_turns')} turns, about ${summary.get('total_cost_usd', 0):.2f} "
          "at API prices (on the subscription, so not billed)")

    branches = collect_candidates(work, run_dir)
    if not branches:
        raise SystemExit(f"The agent made no fix-N branches. Its summary:\n{summary.get('result', '')}")
    (run_dir / "diagnosis.md").write_text(summary.get("result", "") + "\n")
    print(f"Scoring {len(branches)} candidates with the harness...")
    harness.main(run_dir)
    print(f"Done: read {run_dir / 'diagnosis.md'} and {run_dir / 'report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # On the subscription these cost usage limits, not money; Opus at high fits a fresh Pro window better than xhigh
    parser.add_argument("--model", default="opus", help="opus or sonnet")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--budget", type=float, default=25, help="stop the agent at this much usage, in API dollars")
    parser.add_argument("--minutes", type=int, default=90, help="stop the agent after this many minutes")
    args = parser.parse_args()
    main(args.model, args.effort, args.budget, args.minutes)
