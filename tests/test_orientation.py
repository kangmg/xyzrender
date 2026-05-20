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

from xyzrender.types import CellData
from xyzrender.utils import pca_matrix, pca_orient

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


def test_molecule_orient_preserves_centroid_relative_fractional_coords():
    """The genuine invariant under pure rotation: fractional coords expressed
    *relative to the molecular centroid* must be preserved, since rotation
    cannot drift atoms relative to the cell.
    """
    from xyzrender.api import Molecule

    graph, lat0, _ = _build_periodic_graph()
    pos_before = _positions(graph)
    centroid_before = pos_before.mean(axis=0)
    frac_before = (pos_before - centroid_before) @ np.linalg.inv(lat0)

    mol = Molecule(graph=graph, cell_data=CellData(lattice=lat0.copy()))
    mol.orient()
    assert mol.cell_data is not None

    pos_after = _positions(mol.graph)
    centroid_after = pos_after.mean(axis=0)
    rotated_lat = np.asarray(mol.cell_data.lattice, dtype=float)
    frac_after = (pos_after - centroid_after) @ np.linalg.inv(rotated_lat)

    assert np.allclose(frac_after, frac_before, atol=1e-9), (
        "Atom fractional coords (relative to centroid) drifted after orient — "
        "atoms and lattice were not rotated by the same matrix.\n"
        f"before={frac_before.tolist()}\nafter={frac_after.tolist()}"
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


def test_molecule_orient_with_tilt_changes_z():
    """``mol.orient(tilt_degrees=-30)`` applies an additional rotation around
    x AFTER PCA — z coordinates must differ from the no-tilt case (unless the
    molecule happens to be invariant under that rotation)."""
    from xyzrender.api import Molecule

    graph_a, _, _ = _build_periodic_graph()
    graph_b, _, _ = _build_periodic_graph()
    mol_a = Molecule(graph=graph_a)
    mol_b = Molecule(graph=graph_b)

    mol_a.orient(tilt_degrees=None)
    mol_b.orient(tilt_degrees=-30.0)

    pos_a = _positions(mol_a.graph)
    pos_b = _positions(mol_b.graph)
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


# ---------------------------------------------------------------------------
# Molecule.orient() — the Phase 1d canonical orientation method
# ---------------------------------------------------------------------------


def test_molecule_orient_syncs_lattice_storage_locations():
    """``mol.orient()`` must rotate cfg.cell_data.lattice AND
    graph.graph['lattice'] together — the Phase 1d invariant that the old
    ``resolve_orientation`` violated (Bug #1 from the audit)."""
    from xyzrender.api import Molecule

    graph, lat0, _ = _build_periodic_graph()
    cd = CellData(lattice=lat0.copy())
    mol = Molecule(graph=graph, cell_data=cd)

    mol.orient()
    assert mol.cell_data is not None

    lat_cd = np.asarray(mol.cell_data.lattice, dtype=float)
    lat_graph = np.asarray(mol.graph.graph["lattice"], dtype=float)
    assert np.allclose(lat_cd, lat_graph, atol=1e-12)


def test_molecule_orient_centers_atoms_at_origin():
    """Option A convention: ``mol.orient()`` writes back atoms centred at
    the origin (mean = 0).  Matches the existing surface-builder convention
    so downstream cube-grid math stays byte-identical."""
    from xyzrender.api import Molecule

    graph, _, _ = _build_periodic_graph()
    mol = Molecule(graph=graph)
    mol.orient()

    centroid_after = _positions(mol.graph).mean(axis=0)
    assert np.allclose(centroid_after, np.zeros(3), atol=1e-9)


def test_molecule_orient_is_idempotent():
    """Second call to ``mol.orient()`` is a no-op while ``mol.oriented`` is True."""
    from xyzrender.api import Molecule

    graph, lat0, _ = _build_periodic_graph()
    mol = Molecule(graph=graph, cell_data=CellData(lattice=lat0.copy()))

    mol.orient()
    assert mol.lattice is not None
    pos_after_first = _positions(mol.graph)
    lat_after_first = mol.lattice.copy()

    mol.orient()
    assert np.allclose(_positions(mol.graph), pos_after_first, atol=1e-12)
    assert np.allclose(mol.lattice, lat_after_first, atol=1e-12)


def test_molecule_orient_force_reorients():
    """``force=True`` overrides the idempotence check."""
    from xyzrender.api import Molecule

    graph, _, _ = _build_periodic_graph()
    mol = Molecule(graph=graph, oriented=True)  # pretend already oriented
    pos_before = _positions(mol.graph)

    mol.orient()  # no-op due to oriented=True
    assert np.allclose(_positions(mol.graph), pos_before, atol=1e-12)

    mol.orient(force=True)  # now rotates
    # PCA should change the positions for a triclinic 3-atom system
    assert not np.allclose(_positions(mol.graph), pos_before, atol=1e-3)


def test_molecule_copy_isolates_mutations():
    """``mol.copy()`` must produce an independent graph + cell_data so that
    orienting the copy does not mutate the original."""
    from xyzrender.api import Molecule

    graph, lat0, _ = _build_periodic_graph()
    mol = Molecule(graph=graph, cell_data=CellData(lattice=lat0.copy()))
    assert mol.lattice is not None
    pos_orig = _positions(mol.graph)
    lat_orig = mol.lattice.copy()

    rmol = mol.copy()
    rmol.orient()

    assert np.allclose(_positions(mol.graph), pos_orig, atol=1e-12), "mol.graph was mutated through the copy"
    assert np.allclose(mol.lattice, lat_orig, atol=1e-12), "mol.lattice was mutated through the copy"
