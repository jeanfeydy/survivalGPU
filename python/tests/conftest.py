"""Auto-skip logic for tests that require the optional pykeops dependency.

Tests marked `@pytest.mark.needs_keops` (the WCE-specific tests) are
automatically skipped when pykeops isn't installed, instead of erroring
out -- CoxPH doesn't need pykeops and must stay testable without it (e.g.
on Windows, where pykeops can't be installed at all). A single warning
banner is printed at the end of the run summarizing what was skipped.

We deliberately don't use `warnings.warn()` for that banner: pyproject.toml
sets `filterwarnings = ["error"]`, which would turn a real warning into a
test failure. `pytest.mark.skip` and direct `terminalreporter` writes are
both unaffected by that filter.
"""

import pytest
from survivalgpu.wce_features import KEOPS_AVAILABLE

_NEEDS_KEOPS_SKIPPED = pytest.StashKey[list]()


def pytest_configure(config):
    config.stash[_NEEDS_KEOPS_SKIPPED] = []


def pytest_collection_modifyitems(config, items):
    if KEOPS_AVAILABLE:
        return

    reason = (
        "pykeops is not installed. Install it with "
        "`pip install survivalgpu[wce]` (not available on Windows) to "
        "run WCE-dependent tests."
    )
    skip_marker = pytest.mark.skip(reason=reason)
    skipped_ids = config.stash[_NEEDS_KEOPS_SKIPPED]

    for item in items:
        if item.get_closest_marker("needs_keops") is not None:
            item.add_marker(skip_marker)
            skipped_ids.append(item.nodeid)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    skipped_ids = config.stash.get(_NEEDS_KEOPS_SKIPPED, [])
    if not skipped_ids:
        return

    terminalreporter.write_sep(
        "=", "pykeops not installed", yellow=True, bold=True
    )
    terminalreporter.write_line(
        f"WARNING: {len(skipped_ids)} test(s) requiring the optional "
        "'pykeops' dependency (WCE model) were skipped. Install it with "
        "`pip install survivalgpu[wce]` (not available on Windows) to run "
        "the full test suite:",
        yellow=True,
    )
    for nodeid in skipped_ids:
        terminalreporter.write_line(f"  - {nodeid}", yellow=True)
