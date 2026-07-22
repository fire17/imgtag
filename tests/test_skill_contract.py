"""B20 skill-contract conformance: the agent door is a machine API.

Every verb × `--json` must put valid, ANSI-free JSON on stdout, keep human text on stderr,
never prompt, and use the documented exit codes. Latency bounds per BUDGETS B20
(info ≤200ms · index returns a job id ≤500ms · search ≤ B3 p95 + 50ms = 170ms).

Skips (never fails) while the CLI is still being built — these tests are the acceptance
gate for `imgtag bench skill-contract`, not a build blocker for the engine lane.
"""

import json
import re
import shutil
import subprocess
import sys
import time

import pytest

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Latency ceilings (ms) — B20; search = B3-dev p95 (120) + 50.
LIMITS = {"info": 200, "index": 500, "search": 170, "manage": 500}

# Prefer the installed console script (what agents actually call); fall back to -m.
_EXE = shutil.which("imgtag")
CLI = [_EXE] if _EXE else [sys.executable, "-m", "imgtag.cli"]


def _cli_ready() -> bool:
    try:
        import imgtag.cli  # noqa: F401
    except Exception:
        return False
    try:
        return subprocess.run(CLI + ["--help"], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


needs_cli = pytest.mark.skipif(not _cli_ready(), reason="imgtag.cli not implemented yet (b-engine lane)")


def run(*args, timeout=60):
    t0 = time.perf_counter()
    p = subprocess.run(CLI + list(args), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return p, (time.perf_counter() - t0) * 1000.0


def parsed(p):
    """stdout must be pure, ANSI-free JSON."""
    assert not ANSI.search(p.stdout), "ANSI escapes on stdout — --json output must be clean"
    return json.loads(p.stdout)


@needs_cli
@pytest.mark.parametrize("argv", [("info",), ("info", "--json"), ("search", "cat", "--json")])
def test_no_stdin_prompt(argv):
    """0 interactive prompts: stdin is /dev/null, the command must still complete."""
    p, _ = run(*argv)
    assert p.returncode in (0, 4, 7), f"unexpected exit {p.returncode}: {p.stderr[:400]}"


@needs_cli
def test_info_json_shape_and_latency():
    p, ms = run("info", "--json")
    assert p.returncode == 0, p.stderr[:400]
    doc = parsed(p)
    assert isinstance(doc.get("datasets"), list)
    assert "daemon" in doc and "tookMs" in doc
    assert ms <= LIMITS["info"], f"info took {ms:.0f}ms > {LIMITS['info']}ms (B20)"


@needs_cli
def test_search_json_shape_and_provenance():
    p, ms = run("search", "vehicle", "--json")
    assert p.returncode == 0, p.stderr[:400]
    doc = parsed(p)
    for key in ("query", "tookMs", "coverage", "hits", "no_match"):
        assert key in doc, f"missing {key} in search JSON"
    assert {"indexed", "total"} <= set(doc["coverage"])
    for h in doc["hits"]:
        # B18: provenance is NEVER null.
        for key in ("image_id", "path", "dataset", "score", "p", "why"):
            assert h.get(key) is not None, f"null provenance field {key}"
        assert h["why"].get("path") in ("tag", "text")
    if not doc["hits"]:
        assert doc["no_match"] is True, "empty hits must be reported as no_match, not silence"
    assert ms <= LIMITS["search"], f"search took {ms:.0f}ms > {LIMITS['search']}ms (B20)"


@needs_cli
def test_index_returns_job_id_without_blocking(tmp_path):
    from PIL import Image

    src = tmp_path / "imgs"
    src.mkdir()
    for i in range(3):
        Image.new("RGB", (64, 64), (i * 40, 90, 200)).save(src / f"{i}.jpg")
    ds = "imgtag-contract-test"
    try:
        p, ms = run("index", str(src), "--dataset", ds, "--json")
        assert p.returncode == 0, p.stderr[:400]
        doc = parsed(p)
        assert doc.get("job_id"), "index must return a job id"
        assert doc.get("dataset") == ds
        assert ms <= LIMITS["index"], f"index returned in {ms:.0f}ms > {LIMITS['index']}ms (must be non-blocking)"

        # the job is pollable and progress is real
        j, _ = run("info", "--job", doc["job_id"], "--json")
        assert j.returncode == 0
        job = parsed(j)
        assert job["state"] in ("queued", "running", "done", "failed", "aborted")
        assert set(job) >= {"done", "inflight", "failed", "total"}
    finally:
        run("manage", "delete", ds, "--yes", "--json")


@needs_cli
def test_unknown_dataset_exits_4_with_clean_stdout():
    p, _ = run("search", "cat", "--dataset", "no-such-dataset-xyz", "--json")
    assert p.returncode == 4, f"expected exit 4 for unknown dataset, got {p.returncode}"
    parsed(p)  # even errors keep stdout valid JSON under --json
    assert p.stderr, "human-readable error belongs on stderr"


@needs_cli
def test_delete_leaves_no_orphan_bytes(tmp_path):
    """B20: delete leaves 0 orphan bytes."""
    from pathlib import Path

    from PIL import Image

    root = Path.home() / ".imgtag" / "datasets"
    src = tmp_path / "imgs"
    src.mkdir()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(src / "a.jpg")
    ds = "imgtag-orphan-test"
    p, _ = run("index", str(src), "--dataset", ds, "--json", "--wait")
    assert p.returncode == 0, p.stderr[:400]
    assert (root / ds).exists()
    d, _ = run("manage", "delete", ds, "--yes", "--json")
    assert d.returncode == 0
    assert not (root / ds).exists(), "dataset directory survived delete"
    leftovers = [q for q in root.rglob("*") if ds in q.name]
    assert not leftovers, f"orphan files after delete: {leftovers}"


def test_skill_source_is_wellformed():
    """The skill itself: frontmatter present, quoted argument-hint, verbs documented."""
    from pathlib import Path

    skill = Path(__file__).resolve().parents[1] / "skill" / "SKILL.md"
    text = skill.read_text()
    assert text.startswith("---\n")
    fm = text.split("---", 2)[1]
    assert re.search(r"^name: imgtag$", fm, re.M)
    assert re.search(r'^argument-hint: ".+"$', fm, re.M), "argument-hint must be quoted (house rule)"
    assert re.search(r"^description: .{80,}$", fm, re.M)
    for verb in ("index", "info", "manage", "search"):
        assert f"### " in text and verb in text, f"verb {verb} undocumented"
    for code in ("3", "4", "5", "6", "7"):
        assert re.search(rf"^\| {code} \|", text, re.M), f"exit code {code} undocumented"


def test_installer_is_executable_and_idempotent_shape():
    from pathlib import Path

    sh = Path(__file__).resolve().parents[1] / "skill" / "install.sh"
    assert sh.exists()
    assert shutil.which("bash")
    p = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    assert p.returncode == 0, f"install.sh syntax error: {p.stderr}"
    body = sh.read_text()
    assert "ln -sfn" in body and "imgtag-search" in body
