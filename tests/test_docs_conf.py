import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "working_directory", ["checkouts/latest", "checkouts/latest/docs", "unrelated"]
)
def test_shorten_titles_uses_sphinx_source_directory(
    tmp_path, monkeypatch, working_directory
):
    conf_path = Path(__file__).resolve().parents[1] / "docs" / "conf.py"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    config = runpy.run_path(str(conf_path))

    checkout = tmp_path / "checkouts" / "latest"
    source = checkout / "docs"
    autosummary = source / "content" / "api_reference" / "_autosummary"
    autosummary.mkdir(parents=True)
    page = autosummary / "xeofs.single.EOF.rst"
    page.write_text("xeofs.single.EOF\n================\n\nAPI content.\n")

    cwd = tmp_path / working_directory
    cwd.mkdir(exist_ok=True)
    monkeypatch.chdir(cwd)

    callbacks = {}
    app = SimpleNamespace(srcdir=str(source), connect=callbacks.__setitem__)
    config["setup"](app)
    callbacks["builder-inited"](app)

    assert page.read_text() == "EOF\n===\n\nAPI content.\n"
