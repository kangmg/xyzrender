"""Cross-cutting orientation invariants.

These tests pin observable behaviour that any future refactor of the
orientation system (e.g. promoting ``Molecule`` to a class that owns
``.orient()``, ``.copy()``, ``.set_frame()``) must preserve.  Feature-level
orientation tests live alongside their feature (``test_gif.py``,
``test_crystal.py``, ``test_overlay.py``, ``test_api.py``); this file is for
the invariants that span the whole system.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from xyzrender.types import CellData, RenderConfig
from xyzrender.utils import pca_matrix, pca_orient, resolve_orientation

_STRUCTURES = Path(__file__).parent.parent / "examples" / "structures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_periodic_graph() -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """A small 3-atom graph with a triclinic lattice and non-zero origin.

    Returns (graph, lattice, origin) so tests can recompute fractional coords.
    """
    g = nx.Graph()
    lattice = np.array([[5.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, 0.3, 4.5]], dtype=float)
    origin = np.array([0.2, -0.1, 0.05], dtype=float)
    frac = np.array(
        [[0.10, 0.25, 0.40], [0.60, 0.55, 0.30], [0.30, 0.80, 0.70]],
        dtype=float,
    )
    cart = frac @ lattice + origin
    for i, (s, p) in enumerate(zip(["C", "O", "N"], cart, strict=True)):
        g.add_node(i, symbol=s, position=tuple(p.tolist()))
    g.add_edges_from([(0, 1), (1, 2)])
    g.graph["lattice"] = lattice.copy()
    g.graph["lattice_origin"] = origin.copy()
    return g, lattice, origin


def _positions(graph: nx.Graph) -> np.ndarray:
    return np.array([graph.nodes[n]["position"] for n in graph.nodes()], dtype=float)


# ---------------------------------------------------------------------------
# Lattice + atom co-rotation invariant
# ---------------------------------------------------------------------------


def test_resolve_orientation_preserves_centroid_relative_fractional_coords():
    """resolve_orientation centers atoms (mean = 0) and rotates the lattice.
    The genuine invariant under pure rotation: fractional coords expressed
    *relative to the molecular centroid* must be preserved, since rotation
    cannot drift atoms relative to the cell.

    (We do NOT test invariance against the lattice's stored cell_origin,
    because resolve_orientation centers the atoms but keeps the origin
    in world frame — atoms and origin live in different frames after orient.
    That's an internal convention of resolve_orientation; downstream
    surface code knows to compensate.)
    """
    graph, lat0, _ = _build_periodic_graph()
    pos_before = _positions(graph)
    centroid_before = pos_before.mean(axis=0)
    frac_before = (pos_before - centroid_before) @ np.linalg.inv(lat0)

    cfg = RenderConfig(auto_orient=True, cell_data=CellData(lattice=lat0.copy()))
    resolve_orientation(graph, None, cfg)
    assert cfg.cell_data is not None  # narrow for type checker

    pos_after = _positions(graph)
    centroid_after = pos_after.mean(axis=0)
    rotated_lat = np.asarray(cfg.cell_data.lattice, dtype=float)
    frac_after = (pos_after - centroid_after) @ np.linalg.inv(rotated_lat)

    assert np.allclose(frac_after, frac_before, atol=1e-9), (
        "Atom fractional coords (relative to centroid) drifted after orient — "
        "atoms and lattice were not rotated by the same matrix.\n"
        f"before={frac_before.tolist()}\nafter={frac_after.tolist()}"
    )


@pytest.mark.xfail(
    reason="Bug #1 from the orientation audit: resolve_orientation rotates "
    "cfg.cell_data.lattice but does NOT update graph.graph['lattice']. "
    "The two storage locations diverge until Phase 1d (Molecule class) makes "
    "lattice a single property that updates both backing fields atomically. "
    "Existing render path reads from cfg.cell_data so the user-visible "
    "behaviour is correct; this test will start passing post-1d."
)
def test_resolve_orientation_syncs_lattice_storage_locations():
    """After resolve_orientation rotates a periodic system, the lattice stored
    on cfg.cell_data and the lattice stored on graph.graph['lattice'] must
    describe the same physical cell.  If they diverge, downstream code that
    reads one but not the other (crystal images, viewer, supercell expansion)
    will place ghosts in a stale frame."""
    graph, lat0, _ = _build_periodic_graph()
    cfg = RenderConfig(auto_orient=True, cell_data=CellData(lattice=lat0.copy()))

    resolve_orientation(graph, None, cfg)
    assert cfg.cell_data is not None

    lat_cfg = np.asarray(cfg.cell_data.lattice, dtype=float)
    lat_graph = np.asarray(graph.graph["lattice"], dtype=float)
    assert np.allclose(lat_cfg, lat_graph, atol=1e-9), (
        f"lattice diverged.\ncfg={lat_cfg.tolist()}\ngraph={lat_graph.tolist()}"
    )


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_resolve_orientation_idempotent_via_auto_orient_flag():
    """resolve_orientation clears cfg.auto_orient after applying PCA; calling
    it a second time on the same cfg must be a no-op (positions and lattice
    unchanged)."""
    graph, lat0, _ = _build_periodic_graph()
    cfg = RenderConfig(auto_orient=True, cell_data=CellData(lattice=lat0.copy()))

    resolve_orientation(graph, None, cfg)
    assert cfg.cell_data is not None
    pos_after_first = _positions(graph)
    lat_after_first = np.asarray(cfg.cell_data.lattice, dtype=float).copy()
    assert cfg.auto_orient is False, "auto_orient flag must clear after PCA"

    # Second call: cfg.auto_orient is False, so should be a no-op
    resolve_orientation(graph, None, cfg)
    pos_after_second = _positions(graph)
    lat_after_second = np.asarray(cfg.cell_data.lattice, dtype=float)

    assert np.allclose(pos_after_first, pos_after_second, atol=1e-12), (
        "second resolve_orientation rotated positions again"
    )
    assert np.allclose(lat_after_first, lat_after_second, atol=1e-12), (
        "second resolve_orientation rotated lattice again"
    )


def test_pca_orient_double_application_stable():
    """After PCA-orient, the variance is already aligned with x>y>z.  Applying
    PCA a second time to the result must give back essentially the same
    positions (no further rotation needed)."""
    rng = np.random.default_rng(0)
    pos = rng.standard_normal((20, 3)) * np.array([4.0, 2.0, 0.5])
    once = pca_orient(pos)
    twice = pca_orient(once)
    # Allow sign flips on individual axes — PCA result is unique up to sign
    # per axis — by comparing absolute coordinates per atom.
    assert np.allclose(np.abs(twice), np.abs(once), atol=1e-9)


# ---------------------------------------------------------------------------
# Molecule.oriented flag contract
# ---------------------------------------------------------------------------


def test_molecule_oriented_flag_suppresses_pca_in_render():
    """When mol.oriented is True, render() must not re-rotate the source
    mol.graph positions — even when the caller did not pass orient=False.
    This protects callers that orient via the interactive viewer first."""
    from xyzrender.api import load, render

    mol = load(str(_STRUCTURES / "caffeine.xyz"))
    pos_before = _positions(mol.graph)

    mol.oriented = True
    render(mol)
    pos_after = _positions(mol.graph)

    assert np.allclose(pos_before, pos_after, atol=1e-12), "render() mutated mol.graph positions when mol.oriented=True"


# ---------------------------------------------------------------------------
# render() must never mutate the caller's mol (CLAUDE.md design rule)
# ---------------------------------------------------------------------------


def test_render_does_not_mutate_caller_molecule_graph():
    """The CLAUDE.md critical design rule: render() deep-copies mol.graph
    so that orientation/PCA applied during render does not leak back to the
    caller.  Verifies end-to-end: a render with auto_orient on must leave
    mol.graph node positions unchanged."""
    from xyzrender.api import load, render

    mol = load(str(_STRUCTURES / "caffeine.xyz"))
    assert mol.oriented is False, "test precondition: caffeine should load un-oriented"
    pos_before = _positions(mol.graph).copy()

    render(mol)  # default settings include auto_orient=True

    pos_after = _positions(mol.graph)
    assert np.allclose(pos_before, pos_after, atol=1e-12), (
        "render() mutated mol.graph positions in place (violates the deep-copy-at-start-of-render contract)"
    )


# Note: an end-to-end "render twice gives identical SVG output" test belongs
# alongside Phase 1b (thread-safe _render_counter), not here — the current
# global counter increments per call and SVG ids drift (`x0g0`, `x1g0`, …),
# which is orthogonal to orientation correctness.  The orientation invariant
# (mol.graph positions stable across renders) is already covered by
# test_render_does_not_mutate_caller_molecule_graph above.


# ---------------------------------------------------------------------------
# Priority pairs (--align-atoms) reach the orientation path
# ---------------------------------------------------------------------------


def test_priority_pairs_influence_orientation():
    """pca_orient with priority_pairs must produce a different orientation
    than without — confirms the priority-weight machinery is wired through."""
    # A 4-atom system where atoms 0 and 3 are far apart along z; without
    # priority pairs PCA picks z as least-variance.  With priority pairs
    # weighting the (0,3) pair, that bond is pulled into the xy plane.
    pos = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]],
        dtype=float,
    )
    out_no_pp = pca_orient(pos)
    out_with_pp = pca_orient(pos, priority_pairs=[(0, 3)])
    assert not np.allclose(out_no_pp, out_with_pp, atol=1e-6), (
        "priority_pairs had no effect on pca_orient output — alignment flag is not influencing the rotation"
    )


# ---------------------------------------------------------------------------
# Tilt handling (MO surface uses -30° tilt)
# ---------------------------------------------------------------------------


def test_resolve_orientation_with_tilt_changes_z():
    """resolve_orientation(tilt_degrees=-30) applies an additional rotation
    around x AFTER PCA — z coordinates must differ from the no-tilt case
    (unless the molecule happens to be invariant under that rotation)."""
    graph_a, _, _ = _build_periodic_graph()
    graph_b, _, _ = _build_periodic_graph()
    cfg_a = RenderConfig(auto_orient=True)
    cfg_b = RenderConfig(auto_orient=True)

    resolve_orientation(graph_a, None, cfg_a, tilt_degrees=None)
    resolve_orientation(graph_b, None, cfg_b, tilt_degrees=-30.0)

    pos_a = _positions(graph_a)
    pos_b = _positions(graph_b)
    assert np.allclose(pos_a[:, 0], pos_b[:, 0], atol=1e-9), "tilt rotates around x; x coords must be preserved"
    assert not np.allclose(pos_a[:, 2], pos_b[:, 2], atol=1e-6), "tilt_degrees=-30 had no effect on z coordinates"


# ---------------------------------------------------------------------------
# pca_matrix vs pca_orient consistency
# ---------------------------------------------------------------------------


def test_pca_matrix_and_pca_orient_agree():
    """pca_matrix(pos) must return the same rotation matrix that
    pca_orient(pos, return_matrix=True) returns, so callers that want to
    rotate auxiliary data (cube atoms, lattice vectors) by the same matrix
    can use either primitive interchangeably."""
    rng = np.random.default_rng(1)
    pos = rng.standard_normal((10, 3)) * np.array([3.0, 1.5, 0.5])
    vt_only = pca_matrix(pos)
    _, vt_with_orient = pca_orient(pos, return_matrix=True)
    assert np.allclose(vt_only, vt_with_orient, atol=1e-9), (
        "pca_matrix and pca_orient produced different rotation matrices on "
        "the same input — callers mixing the two will get inconsistent frames"
    )
