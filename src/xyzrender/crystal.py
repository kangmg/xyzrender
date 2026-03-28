"""Crystal structure support.

Loading periodic crystal structures (VASP, QE, SIESTA, ABINIT) and generating
periodic image atoms for rendering.

Public API
----------
load_crystal
    Load a VASP/QE/... crystal structure file and return a molecular graph
    together with its ``CellData`` (lattice matrix + cell origin).
build_supercell
    Expand a unit cell into a supercell by integer repetition.
add_crystal_images
    Populate a crystal graph with ghost atoms from the 26 neighbouring unit
    cells so that bonds crossing cell boundaries are visible.
"""

from __future__ import annotations

import itertools
import logging
from itertools import product as _product
from typing import TYPE_CHECKING

import numpy as np
from xyzgraph import DATA, build_graph
from xyzgraph.parameters import BondThresholds

from xyzrender.types import CellData

_bond_thresholds = BondThresholds()

if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx

logger = logging.getLogger(__name__)

__all__ = ["add_crystal_images", "build_supercell", "load_crystal"]


def _build_threshold_matrix(syms: list[str]) -> np.ndarray:
    """Return an (n, n) matrix of bond-distance cutoffs for a list of element symbols.

    Groups atoms by unique element so the inner work is O(E²) where E is the
    number of distinct elements, not O(n²) in Python.
    """
    unique = sorted(set(syms))
    elem_to_idx = {s: i for i, s in enumerate(unique)}

    # Build a small (E, E) cutoff matrix for unique elements only, then
    # expand to (n, n) by mapping each atom to its element row/col.
    vdw = np.array([DATA.vdw.get(s, 2.0) for s in unique])
    tf = np.array([[_bond_threshold_factor(si, sj) for sj in unique] for si in unique])
    elem_thresh = tf * (vdw[:, None] + vdw[None, :])  # (E, E)

    idx = np.array([elem_to_idx[s] for s in syms])  # (n,)
    return elem_thresh[idx[:, None], idx[None, :]]  # (n, n)


def _bond_threshold_factor(sym_i: str, sym_j: str) -> float:
    """Return the bond-distance threshold multiplier for a pair of element symbols."""
    metals = DATA.metals
    hi, hj = sym_i == "H", sym_j == "H"
    mi, mj = sym_i in metals, sym_j in metals
    if hi and hj:
        return _bond_thresholds.threshold_h_h
    if hi or hj:
        return _bond_thresholds.threshold_h_metal if (mi or mj) else _bond_thresholds.threshold_h_nonmetal
    if mi and mj:
        return _bond_thresholds.threshold_metal_metal_self
    if mi or mj:
        metal_sym = sym_i if sym_i in metals else sym_j
        if metal_sym in DATA.sblock_metals:
            return _bond_thresholds.threshold_sblock_ligand
        return _bond_thresholds.threshold_metal_ligand
    return _bond_thresholds.threshold_nonmetal_nonmetal


def _is_bonded(sym_i: str, sym_j: str, dist: float) -> bool:
    """Return True if two atoms at *dist* Å apart are likely bonded.

    Uses xyzgraph's VDW radii (DATA.vdw) and the same type-specific distance
    thresholds as xyzgraph's BondThresholds defaults, so ghost-bond detection
    is consistent with main-cell bond detection.  Note: xyzgraph also applies
    geometric pruning (bond angles, valence) which is not replicated here.
    """
    ri = DATA.vdw.get(sym_i, 2.0)
    rj = DATA.vdw.get(sym_j, 2.0)
    t = _bond_threshold_factor(sym_i, sym_j)
    return dist < t * (ri + rj)


def load_crystal(
    path: str | Path,
    interface_mode: str,
) -> tuple[nx.Graph, CellData]:
    """Load a periodic crystal structure.

    Uses built-in parsers for VASP, QE, SIESTA, and ABINIT.

    Parameters
    ----------
    path:
        Path to the crystal structure input file (POSCAR/CONTCAR for VASP,
        ``*.in`` / ``pw.in`` for Quantum ESPRESSO, ``.fdf`` for SIESTA, etc.).
    interface_mode:
        Interface identifier: ``"vasp"``, ``"qe"``, ``"siesta"``, ``"abinit"``.

    Returns
    -------
    tuple[nx.Graph, CellData]
        Molecular graph with atoms as nodes and ``CellData`` containing the
        3x3 lattice matrix (rows = a, b, c in Å).
    """
    logger.info("Loading %s", path)

    if interface_mode == "vasp":
        from xyzrender.inputs import parse_poscar

        atoms, lattice = parse_poscar(str(path))
    elif interface_mode == "qe":
        from xyzrender.inputs import parse_qe_input

        atoms, lattice, _charge = parse_qe_input(str(path))
    elif interface_mode == "siesta":
        from xyzrender.inputs import parse_siesta_fdf

        atoms, lattice = parse_siesta_fdf(str(path))
    elif interface_mode == "abinit":
        from xyzrender.inputs import parse_abinit_input

        atoms, lattice = parse_abinit_input(str(path))
    else:
        msg = f"Unsupported crystal interface mode: {interface_mode!r}. Supported: vasp, qe, siesta, abinit."
        raise ValueError(msg)

    graph = build_graph(atoms, charge=0, multiplicity=None, kekule=False, quick=True)
    logger.info(
        "Crystal graph: %d atoms, %d bonds, lattice=%s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        lattice.diagonal().round(3),
    )
    graph.graph["lattice"] = lattice
    graph.graph["lattice_origin"] = np.zeros(3)
    return graph, CellData(lattice=lattice)


def build_supercell(graph: "nx.Graph", cell_data: CellData, repeats: tuple[int, int, int]) -> "nx.Graph":
    """Return a new graph representing a repeated supercell.

    The unit-cell graph is replicated *m x n x l* times.  Intra-replica edges
    are copied verbatim (preserving bond orders and all edge attributes).
    Cross-boundary bonds between adjacent replicas are detected with the same
    ``_is_bonded`` distance logic used by :func:`add_crystal_images`.

    Parameters
    ----------
    graph:
        Base-cell graph. Must not already contain periodic image atoms
        (nodes with ``image=True``).
    cell_data:
        Cell lattice/origin describing the base cell.
    repeats:
        Integer repetition counts ``(m, n, l)`` along lattice vectors
        ``a, b, c``. Each must be >= 1.

    Returns
    -------
    nx.Graph
        Supercell graph.  Graph-level metadata (including ``lattice``) is
        copied from the input — the lattice remains the **unit-cell** lattice
        so that the cell-box overlay shows the original unit cell.
    """
    import networkx as nx

    m, n, l_rep = repeats
    if m < 1 or n < 1 or l_rep < 1:
        raise ValueError(f"supercell repeats must be >= 1, got {repeats!r}")

    if any(graph.nodes[nid].get("image", False) for nid in graph.nodes()):
        raise ValueError("build_supercell: graph already contains image atoms (apply before add_crystal_images)")

    a = np.array(cell_data.lattice[0], dtype=float)
    b = np.array(cell_data.lattice[1], dtype=float)
    c = np.array(cell_data.lattice[2], dtype=float)

    base_nodes = list(graph.nodes())
    n_base = len(base_nodes)

    if n_base == 0:
        empty = nx.Graph()
        empty.graph.update(dict(graph.graph))
        return empty

    nid_to_idx = {nid: idx for idx, nid in enumerate(base_nodes)}

    # -- 1. Replicate nodes ------------------------------------------------
    # Deterministic mapping: replica (ii,jj,kk) atom idx →
    #   (ii * n * l_rep + jj * l_rep + kk) * n_base + idx
    new_g = nx.Graph()
    for ii, jj, kk in _product(range(m), range(n), range(l_rep)):
        offset = ii * a + jj * b + kk * c
        base = (ii * n * l_rep + jj * l_rep + kk) * n_base
        for idx, nid in enumerate(base_nodes):
            attrs = dict(graph.nodes[nid])
            pos = np.array(attrs["position"], dtype=float) + offset
            attrs["position"] = (float(pos[0]), float(pos[1]), float(pos[2]))
            attrs.pop("image", None)
            attrs.pop("source", None)
            new_g.add_node(base + idx, **attrs)

    # -- 2. Copy intra-replica edges (preserves bond_order etc.) -----------
    edges = [(nid_to_idx[u], nid_to_idx[v], dict(d)) for u, v, d in graph.edges(data=True)]
    for ii, jj, kk in _product(range(m), range(n), range(l_rep)):
        base = (ii * n * l_rep + jj * l_rep + kk) * n_base
        for ui, vi, data in edges:
            new_g.add_edge(base + ui, base + vi, **data)

    # -- 3. Stitch cross-boundary bonds (same logic as add_crystal_images) -
    base_syms = [graph.nodes[nid]["symbol"] for nid in base_nodes]
    thresh = _build_threshold_matrix(base_syms)  # (n_base, n_base)

    # Only the 13 forward shifts (half of 26) so each replica pair is checked once.
    forward_shifts = [(dx, dy, dz) for dx, dy, dz in _product((-1, 0, 1), repeat=3) if (dx, dy, dz) > (0, 0, 0)]
    for dx, dy, dz in forward_shifts:
        for ii, jj, kk in _product(range(m), range(n), range(l_rep)):
            ni, nj, nk = ii + dx, jj + dy, kk + dz
            if not (0 <= ni < m and 0 <= nj < n and 0 <= nk < l_rep):
                continue
            src_base = (ii * n * l_rep + jj * l_rep + kk) * n_base
            tgt_base = (ni * n * l_rep + nj * l_rep + nk) * n_base
            # Vectorized all-pairs distance between the two replicas.
            src_pos = np.array([new_g.nodes[src_base + u]["position"] for u in range(n_base)])
            tgt_pos = np.array([new_g.nodes[tgt_base + v]["position"] for v in range(n_base)])
            dists = np.linalg.norm(src_pos[:, None, :] - tgt_pos[None, :, :], axis=2)
            bonded_mask = dists < thresh
            for u, v in zip(*np.where(bonded_mask), strict=False):
                new_g.add_edge(src_base + int(u), tgt_base + int(v), bond_order=1.0)

    # -- 4. Graph-level metadata (lattice stays as unit cell for cell box) --
    new_g.graph.update(dict(graph.graph))
    return new_g


def add_crystal_images(graph: nx.Graph, crystal_data: CellData) -> int:
    """Add periodic image atoms that are bonded to cell atoms.

    For each of the 26 neighbouring unit cells, adds image copies of cell
    atoms that form at least one bond with an atom inside the cell.  Image
    nodes carry ``image=True`` and ``source=<cell_atom_id>`` attributes;
    image bonds carry ``image_bond=True``.

    Returns the number of image atoms added.
    """
    lattice = crystal_data.lattice  # (3, 3)
    a, b, c = lattice[0], lattice[1], lattice[2]

    cell_ids = list(graph.nodes())
    if not cell_ids:
        return 0

    n_cell = len(cell_ids)
    cell_syms_list = [graph.nodes[i]["symbol"] for i in cell_ids]
    cell_pos_arr = np.array([graph.nodes[i]["position"] for i in cell_ids])  # (n, 3)

    thresh = _build_threshold_matrix(cell_syms_list)  # (n, n)

    # Precompute H and C masks for ghost-H filtering
    is_h = [s == "H" for s in cell_syms_list]
    is_c = [s == "C" for s in cell_syms_list]

    next_id = max(cell_ids) + 1
    n_added = 0

    shifts = [(dx, dy, dz) for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3) if (dx, dy, dz) != (0, 0, 0)]

    for dx, dy, dz in shifts:
        offset = dx * a + dy * b + dz * c
        img_pos_arr = cell_pos_arr + offset  # (n, 3)

        # All-pairs distances in one numpy call — this is the hot path.
        dists = np.linalg.norm(img_pos_arr[:, None, :] - cell_pos_arr[None, :, :], axis=2)
        bonded_mask = dists < thresh  # (n, n) bool

        # Walk the mask to add ghost nodes + edges (serial: graph mutation).
        for src_idx in range(n_cell):
            bonded_cols = np.where(bonded_mask[src_idx])[0]
            if len(bonded_cols) == 0:
                continue

            # Ghost H that only bonds to C across boundary is not interesting.
            if is_h[src_idx]:
                bonded_cols = [j for j in bonded_cols if not is_c[j]]
                if not bonded_cols:
                    continue

            src_id = cell_ids[src_idx]
            img_pos = img_pos_arr[src_idx]
            img_id = next_id
            next_id += 1
            n_added += 1
            graph.add_node(
                img_id,
                symbol=cell_syms_list[src_idx],
                position=(float(img_pos[0]), float(img_pos[1]), float(img_pos[2])),
                image=True,
                source=src_id,
            )
            for j in bonded_cols:
                graph.add_edge(img_id, cell_ids[j], bond_order=1.0, image_bond=True)

    logger.debug("Added %d image atoms", n_added)
    return n_added
