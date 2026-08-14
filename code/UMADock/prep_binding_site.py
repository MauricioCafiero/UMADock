"""Prepare a UMA-Dock binding-site cluster from any PDB + ligand residue name.

The rest of UMADock consumes a *pre-pruned* active site: a single XYZ file of the
binding-site atoms plus a ``bs_object`` dict (``file_location``, ``name``,
``charge``, ``spin``, ``constraints``, ``size``). Those sites used to be cut and
capped by hand. This module automates it, reusing the OpenMM / PDBFixer machinery
from the sibling ``openmm`` repo:

  1. Repair the raw PDB with PDBFixer (patch missing heavy atoms; add H at pH).
     Missing *residues* (loops) are not invented -- same default as
     ``openmm_md.prepare_protein`` -- so a structured binding site isn't padded
     with modeled loops.
  2. Find the ligand by residue name and select every protein residue with a
     heavy atom within ``cutoff`` (default 4 A) of any ligand heavy atom.
  3. Cut chain ends exposed by the pruning are capped with ACE (N-terminal cap)
     and NME (C-terminal cap), placed off the existing backbone geometry. This
     closes the dangling peptide bonds so the isolated cluster is chemically
     sensible for the MLIP.
  4. Re-protonate the cluster (residues + caps) with the AMBER ff14SB force field
     at pH, and read the net formal charge off the partial-charge sum (ff14SB
     charges sum to the integer formal charge of each residue). ``spin`` defaults
     to 1 (closed shell).
  5. Write the cluster XYZ (element symbols + A coords) and return a ``bs_object``
     ready to hand to ``UMA_Dock``. ``constraints`` are the alpha carbons (Cα) of
     the selected residues -- one backbone anchor per residue, fixed during UMA's
     constrained optimization, mirroring the hand-built ``*_QM_site.xyz`` sites
     (which pin exactly the Cα of every residue; the ACE/NME caps are left free).

Ligand atoms are NOT written into the binding-site XYZ (the ligand is docked
separately by UMADock); they are only used to define the site.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from openmm import app, unit
from pdbfixer import PDBFixer

# Force field file that carries every amino acid, the ACE/NME caps, and the
# protonation variants (HID/HIE/HIP, ASH, GLH, LYN, CYM) addHydrogens picks at pH.
PROTEIN_FF = "amber14/protein.ff14SB.xml"

# Standard amino acids (so we only cluster protein residues, never waters/ions/
# cofactors/ligands). Matches openmm_md.dynamics.PROTEIN_RES.
PROTEIN_RES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH", "LYN",
}

WATER_RESNAMES = {"HOH", "WAT", "TIP3", "SOL"}

# AMBER capping-group atom names (ff14SB templates key on these).
#   ACE (acetyl, N-terminal cap): CH3-C(=O)-  bonded to the next residue's N
#   NME (N-methylamide, C-terminal cap): -N(H)-CH3 bonded to the prev residue's C
ACE_HEAVY = ["CH3", "C", "O"]            # methyl C, carbonyl C, carbonyl O
NME_HEAVY = ["N", "CH3"]                 # amide N, methyl C  (H added by ff)


# ---------------------------------------------------------------------------
# geometry helpers for placing cap heavy atoms off the backbone
# ---------------------------------------------------------------------------

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def _place_ace(n_pos, ca_pos):
    """Heavy-atom coords (CH3, C, O) for an ACE cap bonded to ``n_pos``.

    The cap carbonyl C sits on the N-CA line extended past N (C-N peptide bond
    ~= 1.33 A); O and the methyl C fill the two other sp2 sites of the carbonyl
    in the backbone plane. Approximate geometry is plenty for an MLIP cluster
    model -- ff14SB types the cap by residue/atom name, not by exact coords.
    """
    cap_c = n_pos + 1.33 * _unit(n_pos - ca_pos)
    u = _unit(n_pos - cap_c)                 # capC -> N
    plane = ca_pos - cap_c                   # capC -> CA, defines the plane
    n = _unit(np.cross(u, plane))
    # +120 deg from N around the plane normal -> O; -120 deg -> CH3
    o_dir = -0.5 * u + math.sin(math.radians(120)) * np.cross(n, u)
    ch3_dir = -0.5 * u - math.sin(math.radians(120)) * np.cross(n, u)
    cap_o = cap_c + 1.23 * _unit(o_dir)
    cap_ch3 = cap_c + 1.52 * _unit(ch3_dir)
    # return in atom order [CH3, C, O]
    return {"CH3": cap_ch3, "C": cap_c, "O": cap_o}


def _place_nme(c_pos, ca_pos):
    """Heavy-atom coords (N, CH3) for an NME cap bonded to ``c_pos``."""
    cap_n = c_pos + 1.33 * _unit(c_pos - ca_pos)
    u = _unit(c_pos - cap_n)                 # capN -> C(residue)
    plane = ca_pos - cap_n
    n = _unit(np.cross(u, plane))
    ch3_dir = -0.5 * u + math.sin(math.radians(120)) * np.cross(n, u)
    cap_ch3 = cap_n + 1.47 * _unit(ch3_dir)
    return {"N": cap_n, "CH3": cap_ch3}


def _tetra_h(parent, partner, bond_len=1.09, n=3):
    """``n`` H positions tetrahedrally around ``parent`` (away from ``partner``).

    The parent's bond to ``partner`` defines one vertex of the tetrahedron; the H
    atoms fill the other three (~109.5 deg from the partner bond, 120 deg apart).
    """
    v = _unit(partner - parent)
    # any two vectors perpendicular to v
    ref = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = _unit(np.cross(v, ref))
    b2 = _unit(np.cross(v, b1))
    c, s = math.cos(math.radians(109.5)), math.sin(math.radians(109.5))  # ~ -0.334, 0.943
    pts = []
    for i in range(n):
        th = math.radians(120 * i)
        d = c * v + s * (b1 * math.cos(th) + b2 * math.sin(th))
        pts.append(parent + bond_len * _unit(d))
    return pts


def _amide_h(n_pos, p1, p2, bond_len=1.01):
    """One H on an sp2 amide N, in the plane, opposite the bisector of bonds p1,p2."""
    v = _unit(_unit(p1 - n_pos) + _unit(p2 - n_pos))
    if not np.any(v):
        v = _unit(p1 - n_pos)
    return n_pos + bond_len * (-v)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def _heavy_pos_ang(atom, pos_ang):
    if atom.element is None or atom.element.symbol == "H":
        return None
    p = pos_ang[atom.index]
    return np.array([float(p[0]), float(p[1]), float(p[2])])


def _select_residues(topology, pos_ang, ligand_resname, cutoff, chain=None):
    """Protein residues with a heavy atom within ``cutoff`` A of a ligand heavy atom.

    Selection is restricted to ONE ligand copy (one chain) so that an oligomer
    with multiple ligand copies does not get fused into a single nonsense site.
    ``chain`` selects the chain id of the ligand copy to build around; if None it
    defaults to the chain of the first ligand residue found in the topology.
    Protein residues are taken from that same chain only (a coherent pocket).

    Returns the list of selected Residue objects (OpenMM topology residues), the
    ligand heavy-atom coords (for reporting), and the resolved chain id.
    """
    lig_residues = [r for r in topology.residues()
                    if r.name.strip().upper() == ligand_resname.strip().upper()]
    if not lig_residues:
        raise ValueError(
            f"no ligand residue named {ligand_resname!r} found (with heavy atoms) "
            f"in the PDB. Check the residue name / chain."
        )
    # resolve the chain: explicit, else the chain of the first ligand copy
    if chain is None:
        chain = lig_residues[0].chain.id
    lig_residues = [r for r in lig_residues if r.chain.id == chain]
    if not lig_residues:
        raise ValueError(
            f"no ligand residue {ligand_resname!r} on chain {chain!r}. "
            f"Found copies on chains: {sorted({r.chain.id for r in topology.residues() if r.name.strip().upper()==ligand_resname.strip().upper()})}."
        )
    lig_pts = []
    for res in lig_residues:
        for a in res.atoms():
            p = _heavy_pos_ang(a, pos_ang)
            if p is not None:
                lig_pts.append(p)
    lig_pts = np.array(lig_pts)

    selected = []
    for res in topology.residues():
        if res.name not in PROTEIN_RES:
            continue
        if res.chain.id != chain:
            continue
        pts = [p for a in res.atoms() if (p := _heavy_pos_ang(a, pos_ang)) is not None]
        if not pts:
            continue
        pts = np.array(pts)
        d = np.linalg.norm(pts[:, None, :] - lig_pts[None, :, :], axis=-1)
        if d.min() <= cutoff:
            selected.append(res)
    return selected, lig_pts, chain


# ---------------------------------------------------------------------------
# build the capped cluster topology
# ---------------------------------------------------------------------------

def _atom_pos_ang(atom, pos_ang):
    p = pos_ang[atom.index]
    return np.array([float(p[0]), float(p[1]), float(p[2])])


def _backbone(res, pos_ang):
    """Return (N, CA, C) positions of a residue by atom name, or None if absent."""
    by_name = {a.name: _atom_pos_ang(a, pos_ang) for a in res.atoms()
               if a.name in ("N", "CA", "C")}
    if "N" in by_name and "CA" in by_name and "C" in by_name:
        return by_name["N"], by_name["CA"], by_name["C"]
    return None


def _adjacent(a, b):
    """Two residues are sequential in the original chain (resSeq differ by 1)."""
    if a is None or b is None:
        return False
    try:
        return abs(int(a.id) - int(b.id)) == 1
    except ValueError:
        return False


def _kept_residue_atoms(res, pos_ang, n_frag_term, c_frag_term):
    """Atoms to copy from a PDBFixer residue into the cluster (heavy + H).

    Residue H comes from PDBFixer (already protonated at pH). Two normalizations
    make a cut residue chemically consistent with its cap, so createSystem types
    it as an *internal* (capped) residue rather than a (charged) terminal one:

      * N-fragment terminus (ACE-capped): the backbone N must carry exactly one
        amide H (or zero for PRO, whose N is a secondary amine with no N-H when
        peptide-bonded). A real protein N-terminus has 3 N-H; we keep one, named
        ``H``. An internal cut already has one -- a no-op.
      * C-fragment terminus (NME-capped): drop the terminal extra oxygen ``OXT``
        (only present at a real C-terminus); keep the carbonyl ``O``.

    Sidechain H is always kept as PDBFixer set it.
    """
    keep = [a for a in res.atoms() if a.element is not None
            and not (c_frag_term and a.name == "OXT")]   # drop terminal OXT
    if n_frag_term:
        # backbone N-H names: "H" (internal) or H1/H2/H3 (real N-terminus).
        # Sidechain H like HA/HB is excluded by the exact-name match.
        n_h = [a for a in keep if a.element.symbol == "H"
               and a.name.upper() in ("H", "H1", "H2", "H3")]
        if res.name == "PRO":
            drop = set(n_h)          # capped PRO N has no H (secondary amine)
        else:
            drop = set(n_h[1:])       # keep exactly one amide H (renamed "H" later)
        keep = [a for a in keep if a not in drop]
    return keep


def build_cluster(topology, pos_ang, selected):
    """Build a capped, hydrogenated cluster OpenMM Topology + positions (nm).

    Copies each selected residue's atoms (heavy + H, as protonated by PDBFixer)
    into one new chain, re-attaches peptide bonds between adjacent selected
    residues, and caps every cut terminus with ACE (N-side) / NME (C-side) --
    heavy atoms placed off the backbone plus manually placed H (OpenMM's
    addHydrogens can't protonate NME: hydrogens.xml names the NME methyl "C" but
    ff14SB names it "CH3", so we add the cap H ourselves with the ff14SB names).

    Returns (topology, positions_nm, cap_atom_indices, n_res). ``cap_atom_indices``
    are local atom indices of every ACE/NME atom (informational only -- the UMA
    constraints are the residue Cα, assigned later in ``prepare_binding_site``).
    """
    # group selected residues by chain, ordered by resSeq
    chains = {}
    for res in selected:
        chains.setdefault(res.chain.id, []).append(res)
    for c in chains:
        chains[c].sort(key=lambda r: r.id)

    new_top = app.Topology()
    new_positions = []           # angstrom
    cap_atom_indices = []        # local indices of cap atoms (informational; NOT constraints)

    new_chain = new_top.addChain()  # one chain for the whole cluster
    local_i = 0

    def _add(name, element, residue, pos):
        nonlocal local_i
        a = new_top.addAtom(name, element, residue)
        new_positions.append(pos)
        local_i += 1
        return a

    for residues in chains.values():
        for idx, res in enumerate(residues):
            prev_res = residues[idx - 1] if idx > 0 else None
            next_res = residues[idx + 1] if idx < len(residues) - 1 else None
            n_frag_term = not _adjacent(prev_res, res)    # N needs ACE cap
            c_frag_term = not _adjacent(res, next_res)    # C needs NME cap
            bb = _backbone(res, pos_ang)

            # --- ACE cap on the N-terminus of this fragment ---
            ace_c_atom = None
            if n_frag_term:
                if bb is not None:
                    cap = _place_ace(bb[0], bb[1])
                else:
                    n_pos = _atom_pos_ang(next(a for a in res.atoms()
                                              if a.element and a.element.symbol != "H"), pos_ang)
                    cap = _place_ace(n_pos, n_pos + np.array([1.0, 0.0, 0.0]))
                ace_res = new_top.addResidue("ACE", new_chain)
                ace_atoms = {}
                for nm in ("CH3", "C", "O"):
                    el = app.element.oxygen if nm == "O" else app.element.carbon
                    ace_atoms[nm] = _add(nm, el, ace_res, cap[nm])
                    cap_atom_indices.append(local_i - 1)
                hh = _tetra_h(cap["CH3"], cap["C"])     # 3 methyl H
                for i, p in zip(("HH31", "HH32", "HH33"), hh):
                    ace_atoms[i] = _add(i, app.element.hydrogen, ace_res, p)
                    cap_atom_indices.append(local_i - 1)
                new_top.addBond(ace_atoms["CH3"], ace_atoms["C"])
                new_top.addBond(ace_atoms["C"], ace_atoms["O"])
                for i in ("HH31", "HH32", "HH33"):
                    new_top.addBond(ace_atoms["CH3"], ace_atoms[i])
                ace_c_atom = ace_atoms["C"]

            # --- the residue itself (heavy + H from PDBFixer, normalized) ---
            new_res = new_top.addResidue(res.name, new_chain)
            res_new_atoms = {}     # old_atom -> new_atom
            kept = _kept_residue_atoms(res, pos_ang, n_frag_term, c_frag_term)
            # PRO N-terminus fix: if we dropped all N-H, the kept single-H case
            # (non-PRO real terminus) still needs its one H renamed to "H"
            rename = {}
            if n_frag_term and res.name != "PRO":
                n_h = [a for a in kept if a.element.symbol == "H"
                       and a.name.upper() in ("H1", "H2", "H3", "HN1", "HN2", "HN3")]
                if n_h:
                    rename = {n_h[0]: "H"}   # standardize the surviving amide H name
            for old_atom in kept:
                nm = rename.get(old_atom, old_atom.name)
                na = _add(nm, old_atom.element, new_res, _atom_pos_ang(old_atom, pos_ang))
                res_new_atoms[old_atom] = na

            # carry over bonds among the residue's kept atoms from the original
            kept_set = set(res_new_atoms)
            for b in topology.bonds():
                a1, a2 = b[0], b[1]
                if a1 in kept_set and a2 in kept_set:
                    new_top.addBond(res_new_atoms[a1], res_new_atoms[a2])

            # bond the ACE cap (if any) to this residue's N
            if ace_c_atom is not None:
                n_new = next((na for oa, na in res_new_atoms.items()
                              if oa.name == "N"), None)
                if n_new is not None:
                    new_top.addBond(ace_c_atom, n_new)

            # --- peptide bond to the previously-added residue if adjacent ---
            if _adjacent(prev_res, res):
                new_residues = list(new_top.residues())
                prev_new_res = new_residues[-2]
                this_n = next((a for a in new_res.atoms() if a.name == "N"), None)
                prev_c = next((a for a in prev_new_res.atoms() if a.name == "C"), None)
                if this_n is not None and prev_c is not None:
                    new_top.addBond(prev_c, this_n)

            # --- NME cap on the C-terminus of this fragment ---
            if c_frag_term:
                if bb is not None:
                    cap = _place_nme(bb[2], bb[1])
                else:
                    c_pos = _atom_pos_ang(next(a for a in res.atoms()
                                              if a.element and a.element.symbol != "H"), pos_ang)
                    cap = _place_nme(c_pos, c_pos + np.array([1.0, 0.0, 0.0]))
                nme_res = new_top.addResidue("NME", new_chain)
                nme_atoms = {}
                for nm in ("N", "CH3"):
                    el = app.element.nitrogen if nm == "N" else app.element.carbon
                    nme_atoms[nm] = _add(nm, el, nme_res, cap[nm])
                    cap_atom_indices.append(local_i - 1)
                # amide H on N (in plane, opposite the C(res) and CH3 bonds)
                c_new = next((a for a in new_res.atoms() if a.name == "C"), None)
                h_pos = _amide_h(cap["N"], bb[2] if bb is not None else cap["CH3"],
                                 cap["CH3"])
                nme_atoms["H"] = _add("H", app.element.hydrogen, nme_res, h_pos)
                cap_atom_indices.append(local_i - 1)
                hh = _tetra_h(cap["CH3"], cap["N"])     # 3 methyl H
                for i, p in zip(("HH31", "HH32", "HH33"), hh):
                    nme_atoms[i] = _add(i, app.element.hydrogen, nme_res, p)
                    cap_atom_indices.append(local_i - 1)
                new_top.addBond(nme_atoms["N"], nme_atoms["H"])
                new_top.addBond(nme_atoms["N"], nme_atoms["CH3"])
                for i in ("HH31", "HH32", "HH33"):
                    new_top.addBond(nme_atoms["CH3"], nme_atoms[i])
                if c_new is not None:
                    new_top.addBond(c_new, nme_atoms["N"])

    positions_nm = unit.Quantity(np.array(new_positions, dtype=float) / 10.0,
                                 unit.nanometer)
    return new_top, positions_nm, cap_atom_indices, len(list(new_top.residues()))


# ---------------------------------------------------------------------------
# charge
# ---------------------------------------------------------------------------

def _net_charge(topology, positions_nm):
    """Net formal charge of the cluster: sum ff14SB partial charges, round to int.

    ff14SB partial charges sum to the integer formal charge of each residue
    (the neutral caps contribute 0), so the rounded sum is the chemically
    correct net charge to hand to the MLIP. The cluster is built with caps, so
    cut residues are typed as internal (neutral peptide) rather than as charged
    termini.
    """
    modeller = app.Modeller(topology, positions_nm)
    ff = app.ForceField(PROTEIN_FF)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
    )
    nb = next(f for f in system.getForces()
              if f.__class__.__name__ == "NonbondedForce")
    total = 0.0
    for i in range(nb.getNumParticles()):
        q, _, _ = nb.getParticleParameters(i)
        total += float(q.value_in_unit(unit.elementary_charge))
    return int(round(total)), modeller


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def prepare_binding_site(
    pdb_path,
    ligand_resname,
    cutoff: float = 4.0,
    ph: float = 7.0,
    name: str | None = None,
    spin: int = 1,
    output_dir: str | Path = ".",
    output_xyz: str | Path | None = None,
    write_pdb: bool = True,
    chain: str | None = None,
):
    """Build a UMA-Dock binding-site cluster from any PDB + ligand residue name.

    Args:
        pdb_path: path to the input PDB (any structure; need not be pruned).
        ligand_resname: 3-letter residue name of the ligand that defines the site
            (matched case-insensitively). Ligand atoms are used only to locate the
            site; they are not written into the binding-site XYZ.
        cutoff: distance (A) -- protein residues with a heavy atom within this of
            any ligand heavy atom are kept. Default 4.
        ph: pH for protonation / charge assignment. Default 7.
        name: binding-site name (used in filenames + bs_object). Defaults to the
            ligand residue name + "_site".
        spin: total spin (2S+1 multiplicity) for the bs_object. Default 1
            (closed-shell protein cluster).
        output_dir: directory for the written XYZ (and optional PDB).
        output_xyz: explicit output XYZ path; defaults to ``<name>.xyz`` in
            ``output_dir``.
        write_pdb: also write a capped-cluster PDB for inspection.
        chain: chain id of the ligand copy to build the site around. ``None`` (the
            default) uses the first ligand copy found. This matters for oligomeric
            structures (e.g. 2A3R is a homodimer with one dopamine per monomer):
            without it, the pockets of every ligand copy get fused into one
            nonsense site spanning the whole oligomer. Pass ``chain="A"`` (etc.) to
            pick a specific copy.

    Returns:
        bs_object dict (file_location, name, charge, spin, constraints, size)
        ready for ``UMA_Dock(frags, tries, calculator, bs_object)``.
    """
    pdb_path = Path(pdb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"{ligand_resname.strip().upper()}_site"

    # 1. repair the full structure (patch missing atoms + H at pH; no loop modeling)
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=ph)
    topology = fixer.topology
    pos_ang = fixer.positions.value_in_unit(unit.angstrom)

    # 2. select residues near the ligand (one chain / one ligand copy)
    selected, lig_pts, resolved_chain = _select_residues(
        topology, pos_ang, ligand_resname, cutoff, chain=chain)
    print(f"[prep-site] {len(selected)} protein residues within {cutoff} A of "
          f"ligand {ligand_resname!r} on chain {resolved_chain} "
          f"({len(lig_pts)} ligand heavy atoms)")

    # 3. build the capped, hydrogenated cluster (residue H from PDBFixer +
    #    manually placed cap H -- see build_cluster for why addHydrogens can't
    #    do the caps)
    cluster_top, cluster_pos_nm, cap_idx, n_res = build_cluster(
        topology, pos_ang, selected
    )
    print(f"[prep-site] cluster: {n_res} residues (incl. caps), "
          f"{cluster_top.getNumAtoms()} atoms, {len(cap_idx)} cap atoms")

    # 4. net formal charge from the ff14SB partial-charge sum
    charge, modeller = _net_charge(cluster_top, cluster_pos_nm)
    print(f"[prep-site] net formal charge at pH {ph}: {charge}")

    # 5. write the cluster XYZ (element symbols + A coords) and the capped PDB
    final_top = modeller.topology
    final_pos = modeller.positions.value_in_unit(unit.angstrom)

    if output_xyz is None:
        output_xyz = output_dir / f"{name}.xyz"
    output_xyz = Path(output_xyz)
    symbols, coords = [], []
    for a in final_top.atoms():
        symbols.append(a.element.symbol)
        p = final_pos[a.index]
        coords.append([float(p[0]), float(p[1]), float(p[2])])

    with open(output_xyz, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"{name} | charge={charge} spin={spin} | "
                f"{len(selected)} residues + ACE/NME caps, {cutoff} A cutoff\n")
        for s, c in zip(symbols, coords):
            f.write(f"{s:<2s} {c[0]: .6f} {c[1]: .6f} {c[2]: .6f}\n")

    if write_pdb:
        with open(output_dir / f"{name}_capped.pdb", "w") as f:
            app.PDBFile.writeFile(final_top, modeller.positions, f, keepIds=True)

    # constraints: the alpha carbon (Cα) of every selected protein residue -- the
    # backbone anchors held fixed during UMA's constrained optimization, matching
    # the hand-built *_QM_site.xyz sites (one Cα per residue). The ACE/NME caps are
    # left free to relax into the cut termini (the hand-built sites simply had no
    # caps, so there was nothing else to pin).
    constraints = [a.index for a in final_top.atoms()
                  if a.name == "CA" and a.residue.name in PROTEIN_RES]

    size = final_top.getNumAtoms()
    print(f"[prep-site] wrote {output_xyz} ({size} atoms, "
          f"{len(constraints)} Cα constraints)")
    return {
        "file_location": str(output_xyz),
        "name": name,
        "charge": charge,
        "spin": spin,
        "constraints": constraints,
        "size": size,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Prepare a UMA-Dock binding-site XYZ from any PDB + ligand name.")
    p.add_argument("pdb", help="input PDB (any structure)")
    p.add_argument("ligand", help="ligand residue name (3-letter code, e.g. LDP)")
    p.add_argument("-c", "--cutoff", type=float, default=4.0, help="distance cutoff (A)")
    p.add_argument("--ph", type=float, default=7.0, help="protonation pH")
    p.add_argument("-n", "--name", default=None, help="binding-site name")
    p.add_argument("--spin", type=int, default=1, help="total spin (multiplicity)")
    p.add_argument("-o", "--output-dir", default=".", help="output directory")
    p.add_argument("--chain", default=None,
                   help="chain id of the ligand copy to build the site around "
                        "(default: first ligand copy). For oligomers with one ligand "
                        "per monomer, pick one to avoid fusing the pockets.")
    args = p.parse_args()

    bs = prepare_binding_site(
        args.pdb, args.ligand, cutoff=args.cutoff, ph=args.ph,
        name=args.name, spin=args.spin, output_dir=args.output_dir,
        chain=args.chain,
    )
    print("\nbs_object:")
    for k, v in bs.items():
        print(f"  {k}: {v}")