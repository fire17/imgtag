"""Test-home isolation (E1 law) — no test may touch the user's real ~/.imgtag.

Fixtures that forgot this polluted the user's fleet view with `daemontest`,
`imgtag-contract-test`, `imgtag-orphan-test` and `imgtag-delete-race-test`, and one of
those corpses then blocked a delete for 20s. The guard is structural, not a convention:

1. every test runs with ``IMGTAG_HOME`` pointed at its own tmp dir, inherited by any
   subprocess the test spawns (the CLI and the daemon both read that variable);
2. the real home is checked at session start AND end for datasets matching test patterns,
   so a test that bypasses the env var fails the run instead of quietly leaking.

A test that genuinely needs the real home can request the ``real_imgtag_home`` fixture,
which makes the intent explicit and visible in the report.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

TEST_DATASET_PATTERNS = re.compile(
    r"(^|[-_])(test|smoke|contract|race|orphan|daemontest|pytest|tmp|fixture)([-_]|$)|^ds$|^dup$", re.I
)


def _leaked_datasets() -> list[str]:
    root = Path.home() / ".imgtag" / "datasets"
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and TEST_DATASET_PATTERNS.search(d.name))


@pytest.fixture(autouse=True)
def isolated_imgtag_home(tmp_path, monkeypatch, request):
    """Point IMGTAG_HOME at this test's tmp dir unless the test opted out explicitly."""
    if "real_imgtag_home" in request.fixturenames:
        yield None
        return
    home = tmp_path / "imgtag-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("IMGTAG_HOME", str(home))
    yield home


@pytest.fixture
def real_imgtag_home():
    """Opt-in escape hatch: the test really means the user's home. Say so out loud."""
    return Path(os.environ.get("IMGTAG_HOME", str(Path.home() / ".imgtag")))


@pytest.fixture(scope="session", autouse=True)
def no_pollution_of_the_real_home():
    before = set(_leaked_datasets())
    if before:
        print(f"\n[conftest] pre-existing test-shaped datasets in the real home: {sorted(before)}")
    yield
    after = set(_leaked_datasets())
    new = after - before
    assert not new, (
        f"tests leaked datasets into the real ~/.imgtag: {sorted(new)} — a test bypassed "
        f"IMGTAG_HOME (E1 law: never touch the user's fleet)"
    )
