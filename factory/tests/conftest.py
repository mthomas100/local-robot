"""Shared fixtures for the sorting-cell tests (2026-11-02). Run: envs/genesis/bin/python -m pytest -q factory/tests
(rl factory-test does). The assets must exist: scripts/fetch_ycb.sh, then envs/genesis/bin/python -I factory/cell/ycb.py."""
import os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "factory", "cell"))


@pytest.fixture(scope="session")
def index():
    import ycb
    p = os.path.join(ycb.ASSETS, "index.json")
    if not os.path.exists(p): pytest.fail("no YCB assets: run scripts/fetch_ycb.sh, then envs/genesis/bin/python -I factory/cell/ycb.py")
    return ycb.load_index()


@pytest.fixture(scope="session")
def built(index, tmp_path_factory):
    """The MJCF files built fresh from layout.py into factory/cell/build (what both engines load)."""
    import build_mjcf
    m, paths = build_mjcf.write()
    return m, paths


@pytest.fixture(scope="session")
def mj(built):
    from cell_mujoco import CellMJ
    return CellMJ()


@pytest.fixture(scope="session")
def gs_cell(built):
    from cell_genesis import CellGS
    return CellGS(backend="cpu", cameras=False, items=["cracker_box", "windex_bottle", "tomato_soup_can"])
