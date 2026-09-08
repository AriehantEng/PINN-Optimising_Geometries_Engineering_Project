"""The self-contained notebook must not drift from the tested source.

`PINN_Poiseuille.ipynb` embeds the project's source files verbatim so it can
run in Colab with no clone and no repo layout. That is only safe if the
embedded copy is identical to the sources the rest of this suite covers -- a
notebook carrying a stale copy of the physics would be exactly the kind of
plausible-but-wrong artefact this project exists to prevent.
"""

import json
from pathlib import Path

import pytest

NOTEBOOK = Path("PINN_Poiseuille.ipynb")


@pytest.fixture(scope="module")
def notebook():
    if not NOTEBOOK.is_file():
        pytest.skip(f"{NOTEBOOK} not built")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_valid_json_and_has_cells(notebook):
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) > 5
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("code", "markdown")


def test_embedded_sources_match_the_repository(notebook):
    """The whole point: byte-for-byte, or the notebook is lying."""
    import tools.make_notebook as builder

    payload = None
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "SOURCES = json.loads" in text:
            start = text.index("r'''") + 4
            end = text.index("'''", start)
            payload = json.loads(text[start:end])
            break
    assert payload is not None, "no embedded SOURCES cell found"

    current = builder.collect_sources()
    assert set(payload) == set(current), (
        "the notebook embeds a different set of files than the repository has. "
        "Regenerate with `python tools/make_notebook.py`."
    )
    for name in sorted(current):
        assert payload[name] == current[name], (
            f"{name} differs between the notebook and the repository. "
            "Regenerate with `python tools/make_notebook.py`."
        )


def test_notebook_embeds_the_whole_package(notebook):
    import tools.make_notebook as builder

    files = set(builder.collect_sources())
    for required in (
        "src/__init__.py", "src/config.py", "src/predict.py",
        "src/physics/residuals.py", "src/physics/conventions.py",
        "src/training/loop.py", "src/evaluation/accuracy.py",
    ):
        assert required in files, f"{required} missing from the notebook bundle"
