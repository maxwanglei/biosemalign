"""Ensure the example ``biosemalign.plugins`` entry point is discoverable offline too."""

from __future__ import annotations

import importlib.metadata
import sys
import tomllib
from pathlib import Path

import pytest

from biosemalign.plugins import PLUGIN_ENTRY_POINT_GROUP, ensure_plugins_loaded


@pytest.fixture(scope="session", autouse=True)
def _register_pk_entry_point_for_source_checkouts(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    eps = importlib.metadata.entry_points().select(group=PLUGIN_ENTRY_POINT_GROUP)
    if not any(ep.name == "pk" for ep in eps):
        pyproject = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        )
        project = pyproject["project"]
        plugin_eps = project["entry-points"][PLUGIN_ENTRY_POINT_GROUP]
        dist_dir = tmp_path_factory.mktemp("pk_pack_meta")
        info = dist_dir / f"{project['name'].replace('-', '_')}-{project['version']}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {project['name']}\nVersion: {project['version']}\n",
            encoding="utf-8",
        )
        ep_lines = "\n".join(f"{name} = {target}" for name, target in plugin_eps.items())
        (info / "entry_points.txt").write_text(
            f"[{PLUGIN_ENTRY_POINT_GROUP}]\n{ep_lines}\n", encoding="utf-8"
        )
        sys.path.insert(0, str(dist_dir))

    ensure_plugins_loaded()
