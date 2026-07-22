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


def _imgtag_home():
    """The home the CLI will actually use — tests/conftest.py isolates it per test (E1)."""
    import os
    from pathlib import Path as _P

    return _P(os.environ.get("IMGTAG_HOME", str(_P.home() / ".imgtag")))


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
    """B20: info ≤200ms. Best-of-3 (first run pays page-cache), so this measures the
    steady-state agent call, not disk warm-up. info touches no model — the ceiling is
    interpreter start + manifest reads, and heavy imports must stay lazy to hold it."""
    runs = [run("info", "--json") for _ in range(3)]
    p, _ = runs[-1]
    assert p.returncode == 0, p.stderr[:400]
    doc = parsed(p)
    assert isinstance(doc.get("datasets"), list)
    assert "daemon" in doc and "tookMs" in doc
    ms = min(m for _, m in runs)
    print(f"info latency: {ms:.0f}ms (best of 3, limit {LIMITS['info']}ms)")
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


def _daemon_running() -> bool:
    p, _ = run("status", "--json")
    try:
        return bool(json.loads(p.stdout).get("daemon", {}).get("running"))
    except Exception:
        return False


@needs_cli
def test_search_latency(real_imgtag_home):
    """B20 search ceiling is a WARM-DAEMON number (B3 asserts the daemon resident).

    Without a daemon every invocation re-pays interpreter + ORT session creation — that is
    the cold path, budgeted by B13 (≤2s), not B3. Assert whichever contract actually applies
    and always print the measured value; never quote a cold number as the B20 result.

    Uses the REAL home (opt-in fixture) on purpose: the resident daemon lives there, and a
    per-test throwaway home never has one, so isolation would silently downgrade this to the
    cold path forever. Search is READ-ONLY — it creates no dataset, so E1 is not violated.
    """
    warm = _daemon_running()
    run("search", "vehicle", "--json")  # discard the first (may be the call that starts the daemon)
    # Best-of-3 distinct queries: the agent-relevant number is steady-state wall time
    # (interpreter start + client + serverMs), not a momentary spike while sibling lanes churn.
    trials = [run("search", q, "--json") for q in ("parked cars at night", "a dog on grass", "red bicycle")]
    ms = min(m for _, m in trials)
    served = parsed(trials[-1][0]).get("served_by")
    limit, label = (LIMITS["search"], "B20 warm-daemon") if warm else (2000, "B13 cold-process")
    print(f"search latency: {ms:.0f}ms served_by={served} (best of 3, limit {limit}ms {label})")
    if warm:
        # ADR-5's warm path is only real if the daemon actually answered — an in-process
        # fallback that happens to be fast is not the budgeted path.
        assert served == "daemon", f"daemon resident but query served_by={served!r} (ADR-13 client path)"
    assert ms <= limit, f"search took {ms:.0f}ms > {limit}ms ({label})"
    if not warm:
        pytest.skip(f"B20 ceiling untested: no resident daemon; cold path {ms:.0f}ms ≤2s (B13) only")


def _settle(job_id, timeout=60.0):
    """Block until a job leaves queued/running — deleting a dataset out from under a live
    writer is its own contract (see test_delete_during_active_job_is_safe); tests that only
    want cleanup must not race it."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        p, _ = run("info", "--job", job_id, "--json")
        try:
            state = json.loads(p.stdout).get("state")
        except Exception:
            return None
        if state not in ("queued", "running"):
            return state
        time.sleep(0.25)
    return "timeout"


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
        _settle(doc.get("job_id"))
        run("manage", "delete", ds, "--yes", "--json")


@needs_cli
def test_delete_during_active_job_is_safe(tmp_path):
    """A delete issued while an index job is in flight must be UNAMBIGUOUS.

    Either the CLI refuses it (exit 3, dataset-locked — the documented code) or it deletes
    and the dataset STAYS deleted. What it must never do is report success/unknown and then
    let the detached writer recreate the dataset: the user's delete would be silently undone
    and the recreated shards are orphan bytes by B12's definition.
    """
    from pathlib import Path

    from PIL import Image

    src = tmp_path / "imgs"
    src.mkdir()
    for i in range(60):
        Image.new("RGB", (128, 128), (i % 255, 80, 160)).save(src / f"{i}.jpg")
    ds = "imgtag-delete-race-test"
    root = _imgtag_home() / "datasets" / ds
    # The window is timing-dependent (queued vs running writer) — probe it a few times so a
    # lucky ordering cannot report the race as fixed.
    for attempt in range(3):
        p, _ = run("index", str(src), "--dataset", ds, "--json")
        job_id = parsed(p).get("job_id")
        try:
            d, _ = run("manage", "delete", ds, "--yes", "--json")
            assert d.returncode in (0, 3), (
                f"attempt {attempt}: delete during an in-flight job exited {d.returncode} "
                f"(expected 0, or 3 dataset-locked); stderr={d.stderr[:200]}"
            )
            _settle(job_id)
            if d.returncode == 0:
                assert not root.exists(), f"attempt {attempt}: deleted dataset resurrected by the in-flight job"
        finally:
            _settle(job_id)
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

    root = _imgtag_home() / "datasets"
    src = tmp_path / "imgs"
    src.mkdir()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(src / "a.jpg")
    ds = "imgtag-orphan-test"
    p, _ = run("index", str(src), "--dataset", ds, "--json", "--wait")
    assert p.returncode == 0, p.stderr[:400]
    assert (root / ds).exists(), (
        f"index --wait reported success but wrote nothing under {root} "
        f"(IMGTAG_HOME={_imgtag_home()}); stdout={p.stdout[:300]} stderr={p.stderr[:300]}"
    )
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
