"""CLI smoke: every verb, in a real subprocess, under --json and without it.

WHY THIS EXISTS (I6, root-cause-with-a-memory): a `--no-daemon` flag was read in
cmd_search before it was registered on the subparser, so `imgtag search ... --json`
died with an AttributeError traceback — which is *also* a B20 violation, since the
contract allows documented exit codes only, never a traceback. Unit tests could not
see it: they call the functions, not the parser. These tests invoke the installed
entry point exactly as an agent does.

Every case asserts: exit code in the documented set, no traceback on stderr, and
(under --json) parseable JSON on stdout.
"""

import json
import os
import re
import shutil
import subprocess
import sys

import pytest
from PIL import Image

DOCUMENTED_EXITS = {0, 2, 3, 4, 5, 6, 7}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_EXE = shutil.which("imgtag")
CLI = [_EXE] if _EXE else [sys.executable, "-m", "imgtag.cli"]


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    """An isolated IMGTAG_HOME with one tiny real dataset, so verbs have something to chew."""
    h = tmp_path_factory.mktemp("home")
    src = tmp_path_factory.mktemp("imgs")
    for i in range(2):
        Image.new("RGB", (48, 48), (i * 90, 40, 200)).save(src / f"{i}.jpg")
    env = {**os.environ, "IMGTAG_HOME": str(h)}
    subprocess.run(CLI + ["index", str(src), "--dataset", "smoke", "--wait", "--json"],
                   capture_output=True, text=True, timeout=600, env=env)
    return h


def run(home, *args, timeout=120):
    return subprocess.run(CLI + list(args), capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL, env={**os.environ, "IMGTAG_HOME": str(home)})


def check(p, argv, want_json: bool):
    assert "Traceback (most recent call last)" not in p.stderr, \
        f"{argv} crashed instead of exiting with a documented code:\n{p.stderr[-800:]}"
    assert p.returncode in DOCUMENTED_EXITS, f"{argv} -> undocumented exit {p.returncode}: {p.stderr[-300:]}"
    if want_json and p.returncode in (0, 4, 5, 6, 7) and p.stdout.strip():
        assert not ANSI.search(p.stdout), f"{argv} put ANSI escapes on stdout"
        json.loads(p.stdout)  # raises on malformed JSON


# Every verb an agent can call, happy path. --json and bare, because the parser
# differs between them and only a real invocation proves the attributes line up.
HAPPY = [
    ("info",),
    ("info", "--dataset", "smoke"),
    ("info", "smoke"),
    ("status",),
    ("manage", "list"),
    ("manage", "verify", "smoke"),
    ("search", "a blue square"),
    ("search", "a blue square", "--dataset", "smoke", "-k", "5"),
    ("search", "a blue square", "--no-daemon"),
    ("doctor", "--show"),
    ("job",),
]


@pytest.mark.parametrize("argv", HAPPY, ids=lambda a: "_".join(a[:3]))
@pytest.mark.parametrize("as_json", [True, False], ids=["json", "human"])
def test_verb_runs_without_crashing(home, argv, as_json):
    full = list(argv) + (["--json"] if as_json else [])
    check(run(home, *full), full, as_json)


# Error paths must ALSO be documented exits, never tracebacks (B20).
ERRORS = [
    (("search", "cat", "--dataset", "no-such-dataset-xyz", "--json"), 4),
    (("info", "--job", "nosuchjob", "--json"), 4),
    (("manage", "delete", "no-such-dataset-xyz", "--yes", "--json"), 4),
    (("manage", "verify", "no-such-dataset-xyz", "--json"), 4),
    (("index", "/nonexistent/path/xyz", "--dataset", "smoke2", "--json", "--wait"), 0),  # 0 files is not an error
]


@pytest.mark.parametrize("argv,want", ERRORS, ids=lambda a: "_".join(a[:3]) if isinstance(a, tuple) else str(a))
def test_error_paths_use_documented_exit_codes(home, argv, want):
    p = run(home, *argv)
    check(p, argv, "--json" in argv)
    assert p.returncode == want, f"{argv} -> {p.returncode}, expected {want}: {p.stderr[-300:]}"


def test_daemon_and_in_process_search_return_the_same_schema(home):
    """One owner (core.search.Searcher) produces one schema on both transports. The CLI
    used to carry its own scan + invented sigmoid; --no-daemon must choose the transport,
    never the semantics."""
    a = json.loads(run(home, "search", "a blue square", "--json").stdout)
    b = json.loads(run(home, "search", "a blue square", "--json", "--no-daemon").stdout)
    ignore = {"clientMs", "tookMs", "served_by", "text_tower_load_ms", "text_tower"}
    assert set(a) - ignore == set(b) - ignore, "daemon and in-process answers disagree on keys"
    if a["hits"] and b["hits"]:
        assert set(a["hits"][0]) == set(b["hits"][0]), "hit shape differs between transports"
        assert a["hits"][0]["image_id"] == b["hits"][0]["image_id"], "same query, different top hit"


def test_index_returns_job_id_and_search_answers(home):
    p = run(home, "search", "a blue square", "--dataset", "smoke", "--json")
    assert p.returncode == 0, p.stderr[-300:]
    doc = json.loads(p.stdout)
    assert doc["coverage"]["indexed"] == 2
    for h in doc["hits"]:  # B18: provenance never null, whoever served it
        assert h["image_id"] and h["path"] and h["dataset"] == "smoke"
        assert h["why"]["path"] in ("tag", "text")
    # served_by names the path that actually answered — the daemon adds its own labels
    # (e.g. "cold-load" when it had to load the tower), so assert it is stated, not which
    assert isinstance(doc.get("served_by", ""), str)


def test_no_undefined_names_in_engine_sources():
    """The static half of the same lesson: ruff's F821 caught a `del` that emptied a
    closure cell. Keep the engine's own files clean so that class cannot come back."""
    ruff = shutil.which("ruff") or shutil.which("uvx")
    if not ruff:
        pytest.skip("ruff not available")
    cmd = ([ruff, "check"] if ruff.endswith("ruff") else [ruff, "ruff", "check"]) + [
        "--select", "F821,F822,F811,E999",
        "src/imgtag/core/store.py", "src/imgtag/core/models.py", "src/imgtag/core/indexer.py",
        "src/imgtag/core/doctor.py", "src/imgtag/core/progress.py", "src/imgtag/cli.py",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                       cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    assert p.returncode == 0, p.stdout[-1500:]
