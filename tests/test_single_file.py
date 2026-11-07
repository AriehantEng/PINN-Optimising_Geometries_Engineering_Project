"""The single-file build must not drift from the tested source.

`pinn_poiseuille_all_in_one.py` embeds every `src/` file verbatim, as the
notebook does (tests/test_notebook.py). It went stale once without anything
noticing: its copy of src/evaluation/accuracy.py predated the near-inlet
residual band. Comparing it with a fresh build catches that.
"""

from pathlib import Path

import pytest

SINGLE_FILE = Path("pinn_poiseuille_all_in_one.py")


def test_single_file_matches_a_fresh_build():
    import tools.make_single_file as builder

    if not SINGLE_FILE.is_file():
        pytest.skip(f"{SINGLE_FILE} not built")
    assert SINGLE_FILE.read_text(encoding="utf-8") == builder.build(), (
        f"{SINGLE_FILE} differs from a fresh build of the sources. "
        "Regenerate with `python tools/make_single_file.py`."
    )
