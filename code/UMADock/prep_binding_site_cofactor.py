"""Extension of prep_binding_site.py for receptors whose binding site includes
a non-standard cofactor (a heme, an Fe-S cluster, F430, FAD/NAD, etc.) that
must remain a permanent part of the pocket cluster -- not be discarded like a
reference ligand, and not reducible to a bare structural-metal ion.

Why the base prepare_binding_site() can't do this directly: it only ever
copies two kinds of residue into the cluster -- standard amino acids
(PROTEIN_RES) and bare single-atom structural metals (SUPPORTED_METAL_RESNAMES).
Everything else, including whatever ligand_resname is handed to it, is used
only to locate the pocket and is then dropped (see that module's docstring:
"we only cluster protein residues, never waters/ions/cofactors/ligands"). A
multi-atom cofactor -- especially one with its metal embedded as a regular
atom inside it, like F430's nickel, rather than as its own residue -- fits
neither bucket.

This module auto-discovers any such residue (anything that is not a standard
amino acid, not water, and not an already-handled bare structural metal) and
keeps its atoms verbatim in the cluster: real coordinates and protonation from
the input structure, not re-derived (PDBFixer never sees these residues --
see "Why PDBFixer must not see the cofactor" below). Net formal charge for
each kept cofactor is taken from a caller-supplied map (e.g. a RESP partial-
charge sum from a real parameterization, such as this project's
f430_mcpb/site/*.mol2 for F430); a cofactor with no supplied charge defaults
to 0.0 with a loud warning, since there is no general, reliable way to derive
a cofactor's formal charge from bare coordinates alone (ff14SB has no
template for it, and PDB files don't carry bond orders).

Why PDBFixer must not see the cofactor: feeding it an unrecognized residue
name makes addMissingHydrogens() try an RCSB Chemical Component Dictionary
lookup for that name, which can crash outright -- observed directly against
this project's receptor: PDBFixer fetched a malformed CCD entry for residue
name 'F31' and raised ``ValueError: could not convert string to float: '?'``.
Every discovered/explicit cofactor residue is stripped out before PDBFixer
ever runs and spliced back in afterward from the original, untouched
coordinates.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from openmm import app, unit
from pdbfixer import PDBFixer

from UMADock.prep_binding_site import (
    _heavy_pos_ang, _select_metals, build_cluster, _net_charge,
    PROTEIN_RES, SUPPORTED_METAL_RESNAMES, METAL_CHARGE, WATER_RESNAMES,
)


def _select_residues_near_points(topology, pos_ang, lig_pts, cutoff):
    """Protein residues (any chain) with a heavy atom within cutoff of lig_pts.

    No chain restriction. The oligomer safeguard in the base module's
    _select_residues exists so that one ligand copy's pocket in a
    homo-oligomer doesn't get fused with another copy's. A cofactor-defined
    site is typically a single copy, and may sit at a subunit interface (as
    F430 does in MCR), so restricting to one chain risks dropping real
    protein contacts rather than preventing a fusion that isn't happening.
    """
    selected = []
    for res in topology.residues():
        if res.name not in PROTEIN_RES:
            continue
        pts = [p for a in res.atoms() if (p := _heavy_pos_ang(a, pos_ang)) is not None]
        if not pts:
            continue
        pts = np.array(pts)
        d = np.linalg.norm(pts[:, None, :] - lig_pts[None, :, :], axis=-1)
        if d.min() <= cutoff:
            selected.append(res)
    return selected


def discover_cofactor_residues(topology, exclude_resnames=()):
    """Residues that are neither a standard amino acid, water, nor an
    already-handled bare structural metal (SUPPORTED_METAL_RESNAMES) --
    hemes, Fe-S clusters, flavins, nicotinamides, F430, or any other
    multi-atom cofactor the input structure happens to carry.

    ``exclude_resnames`` drops specific residue names after inspection (e.g. a
    crystallization buffer molecule -- glycerol, PEG, sulfate -- that turned
    up but isn't a real cofactor). Matched case-insensitively.
    """
    exclude_upper = {r.strip().upper() for r in exclude_resnames}
    seen = {}
    for res in topology.residues():
        nm = res.name.strip().upper()
        if (res.name in PROTEIN_RES or nm in WATER_RESNAMES
                or nm in SUPPORTED_METAL_RESNAMES or nm in exclude_upper):
            continue
        seen.setdefault(nm, []).append(res)
    return seen  # {resname: [Residue, ...]}


def prepare_binding_site_with_cofactor(
    pdb_path,
    cofactor_charges: dict[str, float] | None = None,
    cofactor_resnames: list[str] | None = None,
    exclude_resnames: list[str] = (),
    ligand_resname: str | None = None,
    cutoff: float = 4.0,
    ph: float = 7.0,
    name: str | None = None,
    spin: int = 1,
    output_dir: str | Path = ".",
    write_pdb: bool = True,
    chain: str | None = None,
    keep_metals: bool = True,
    constrain_metals: bool = False,
    constrain_cofactor: bool = True,
):
    """Build a UMA-Dock binding-site cluster that keeps any cofactor(s) intact.

    Args:
        pdb_path: input PDB. Any cofactor(s) must already be fully built and
            protonated (this function does not repair or protonate them).
        cofactor_charges: ``{resname: net formal charge}`` for every cofactor
            residue to keep, e.g. from a RESP partial-charge sum. A discovered
            or explicitly-named cofactor with no entry here defaults to
            charge 0.0 **with a printed warning** -- there is no general way
            to derive a formal charge from bare coordinates, so an unlisted
            cofactor should be treated as "probably wrong, go find the real
            number" rather than silently trusted.
        cofactor_resnames: restrict to exactly these residue names instead of
            auto-discovering. Use this when you know precisely what the
            cofactor is (as this project does for F430: ``["F31", "NI1"]``).
            Default ``None`` auto-discovers every non-protein, non-water,
            non-bare-metal residue in the structure (see
            ``discover_cofactor_residues``).
        exclude_resnames: resnames to drop from auto-discovery (crystallization
            additives etc. that aren't real cofactors). Ignored if
            ``cofactor_resnames`` is given explicitly.
        ligand_resname: if the pocket should be defined by a separate
            crystallographic ligand (as in the base module) rather than by the
            cofactor(s) themselves, pass its residue name here; it is used
            only to locate the pocket and is not written into the cluster
            (same semantics as the base module). Default ``None`` uses the
            union of all kept cofactor atoms to define the pocket -- the
            right choice when the cofactor *is* the site, as for F430/MCR.
        cutoff, ph, name, spin, output_dir, write_pdb, chain, keep_metals,
            constrain_metals: as in ``prep_binding_site.prepare_binding_site``.
            ``chain`` only matters when ``ligand_resname`` is given.
        constrain_cofactor: freeze each kept cofactor's atoms at the input
            coordinates during UMA's pose optimization (default True), rather
            than leaving them free to relax. Default True because these
            coordinates are expected to come from a validated build (e.g.
            Seminario force constants holding F430's geometry close to its
            QM-optimized shape); re-deriving that geometry from an MLIP's
            unconstrained optimization during a short docking pose evaluation
            risks introducing cofactor-geometry noise unrelated to the new
            ligand's placement, which is the thing actually being tested. Set
            False if you specifically want the cofactor free to relax (e.g.
            you don't trust the input geometry, or you're testing how much a
            pose's energy depends on cofactor movement vs. ligand movement).

    Returns:
        bs_object dict (file_location, name, charge, spin, constraints, size).
    """
    pdb_path = Path(pdb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = name or "cofactor_site"
    cofactor_charges = {k.strip().upper(): v for k, v in (cofactor_charges or {}).items()}

    # 1. parse the ORIGINAL file (no PDBFixer) to find the cofactor(s), pull
    #    their atoms out verbatim, and get their coordinates for the
    #    pocket-residue cutoff -- independent of whatever PDBFixer does to the
    #    rest of the structure.
    raw = app.PDBFile(str(pdb_path))
    raw_top = raw.topology
    raw_pos_ang = raw.positions.value_in_unit(unit.angstrom)

    if cofactor_resnames is not None:
        want = {c.strip().upper() for c in cofactor_resnames}
        found = {nm: [r for r in raw_top.residues() if r.name.strip().upper() == nm]
                for nm in want}
        missing = [nm for nm, rs in found.items() if not rs]
        if missing:
            raise ValueError(f"no residues named {missing} found in {pdb_path}")
    else:
        found = discover_cofactor_residues(raw_top, exclude_resnames=exclude_resnames)
        if not found:
            raise ValueError(
                f"no cofactor-like residues auto-discovered in {pdb_path} (nothing "
                f"outside standard amino acids / water / {sorted(SUPPORTED_METAL_RESNAMES)}). "
                f"Pass cofactor_resnames explicitly if the site has none, or check the PDB."
            )
        print(f"[prep-site+cofactor] auto-discovered cofactor residue(s): "
              f"{[f'{nm} x{len(rs)}' for nm, rs in found.items()]}")

    cofactor_residues = [r for rs in found.values() for r in rs]
    cofactor_resnames_upper = set(found)

    for nm in cofactor_resnames_upper:
        if nm not in cofactor_charges:
            print(f"[prep-site+cofactor] WARNING: no charge supplied for cofactor "
                  f"residue {nm!r} -- defaulting to 0.0. If this is a real charged "
                  f"cofactor (most hemes, Fe-S clusters, nucleotide cofactors are), "
                  f"the site's net charge will be wrong until you supply the real "
                  f"value via cofactor_charges.")
            cofactor_charges[nm] = 0.0

    cof_symbols = []
    cof_coords_ang = []
    for res in cofactor_residues:
        for a in res.atoms():
            if a.element is None:
                continue
            cof_symbols.append(a.element.symbol)
            p = raw_pos_ang[a.index]
            cof_coords_ang.append([float(p[0]), float(p[1]), float(p[2])])
    cof_coords_ang = np.array(cof_coords_ang)
    total_cofactor_charge = sum(cofactor_charges[nm] * len(found[nm]) for nm in found)
    print(f"[prep-site+cofactor] cofactor residues "
          f"{[f'{r.name}{r.id}' for r in cofactor_residues]}: {len(cof_symbols)} atoms "
          f"kept verbatim (net charge {total_cofactor_charge:+.2f})")

    # 2. locate the ligand (if any) BEFORE stripping the cofactor, since both
    #    live in the same original, unrepaired topology/coordinate frame.
    ligand_pts = None
    if ligand_resname is not None:
        lig_residues = [r for r in raw_top.residues()
                        if r.name.strip().upper() == ligand_resname.strip().upper()]
        if not lig_residues:
            raise ValueError(f"no ligand residue named {ligand_resname!r} found in {pdb_path}")
        if chain is None:
            chain = lig_residues[0].chain.id
        lig_residues = [r for r in lig_residues if r.chain.id == chain]
        pts = []
        for res in lig_residues:
            for a in res.atoms():
                p = _heavy_pos_ang(a, raw_pos_ang)
                if p is not None:
                    pts.append(p)
        ligand_pts = np.array(pts)

    # 3. a cofactor-free (and ligand-free, if any) copy of the structure for
    #    PDBFixer -- see module docstring for why it must never see the
    #    cofactor; the ligand is dropped for the same reason the base module
    #    drops it (it's not part of the receptor to keep).
    to_delete = list(cofactor_residues)
    if ligand_resname is not None:
        to_delete += [r for r in raw_top.residues()
                     if r.name.strip().upper() == ligand_resname.strip().upper()]
    modeller = app.Modeller(raw_top, raw.positions)
    modeller.delete(a for r in to_delete for a in r.atoms())
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, dir=output_dir)
    tmp.close()
    with open(tmp.name, "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

    # 4. repair exactly as the base pipeline does (patch missing heavy atoms;
    #    add H at pH; no loop modeling)
    fixer = PDBFixer(filename=tmp.name)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=ph)
    topology = fixer.topology
    pos_ang = fixer.positions.value_in_unit(unit.angstrom)
    Path(tmp.name).unlink(missing_ok=True)

    # 5. pocket residues: around the ligand if one was given, else around the
    #    union of all kept cofactor atoms (the cofactor IS the site).
    pocket_pts = ligand_pts if ligand_pts is not None else cof_coords_ang
    selected = _select_residues_near_points(topology, pos_ang, pocket_pts, cutoff)
    print(f"[prep-site+cofactor] {len(selected)} protein residues within {cutoff} A "
          f"of the {'ligand' if ligand_pts is not None else 'cofactor'} "
          f"({len(pocket_pts)} reference heavy atoms)")

    selected_metals = (_select_metals(topology, pos_ang, pocket_pts, cutoff)
                       if keep_metals else [])
    if selected_metals:
        names = ", ".join(f"{r.name}/{r.id}" for r in selected_metals)
        print(f"[prep-site+cofactor] {len(selected_metals)} additional bare structural "
              f"metal ion(s) within {cutoff} A: {names}")

    # 6. capped, hydrogenated protein cluster (unchanged logic from the base module)
    cluster_top, cluster_pos_nm, cap_idx, n_res = build_cluster(topology, pos_ang, selected)
    print(f"[prep-site+cofactor] cluster: {n_res} residues (incl. caps), "
          f"{cluster_top.getNumAtoms()} atoms, {len(cap_idx)} cap atoms")

    charge, final_modeller = _net_charge(cluster_top, cluster_pos_nm)
    protein_charge = charge
    metal_charge = sum(METAL_CHARGE[r.name] for r in selected_metals)
    charge += metal_charge + total_cofactor_charge
    print(f"[prep-site+cofactor] net formal charge at pH {ph}: {charge:.2f} "
          f"({protein_charge} protein/caps + {metal_charge} metal "
          f"+ {total_cofactor_charge:+.2f} cofactor)")

    # 6b. append any OTHER bare structural metals (unchanged from base module)
    if selected_metals:
        metal_top = app.Topology()
        metal_chain = metal_top.addChain()
        metal_pos_nm = []
        for res in selected_metals:
            for a in res.atoms():
                if a.element is None:
                    continue
                m_res = metal_top.addResidue(res.name, metal_chain)
                metal_top.addAtom(a.name, a.element, m_res)
                p = pos_ang[a.index]
                metal_pos_nm.append([float(p[0]) / 10.0, float(p[1]) / 10.0, float(p[2]) / 10.0])
        final_modeller.add(metal_top, unit.Quantity(np.array(metal_pos_nm), unit.nanometer))

    # 6c. append every kept cofactor residue, verbatim -- element + input
    #     coordinates, no re-protonation, no ff14SB typing (there is no
    #     template for it).
    cof_top = app.Topology()
    cof_chain = cof_top.addChain()
    cof_pos_nm = []
    for res in cofactor_residues:
        c_res = cof_top.addResidue(res.name, cof_chain)
        for a in res.atoms():
            if a.element is None:
                continue
            cof_top.addAtom(a.name, a.element, c_res)
            p = raw_pos_ang[a.index]
            cof_pos_nm.append([float(p[0]) / 10.0, float(p[1]) / 10.0, float(p[2]) / 10.0])
    final_modeller.add(cof_top, unit.Quantity(np.array(cof_pos_nm), unit.nanometer))

    # 7. write cluster XYZ + inspection PDB
    final_top = final_modeller.topology
    final_pos = final_modeller.positions.value_in_unit(unit.angstrom)
    output_xyz = output_dir / f"{name}.xyz"
    symbols, coords = [], []
    for a in final_top.atoms():
        symbols.append(a.element.symbol)
        p = final_pos[a.index]
        coords.append([float(p[0]), float(p[1]), float(p[2])])
    with open(output_xyz, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"{name} | charge={charge:.0f} spin={spin} | {len(selected)} residues + "
                f"ACE/NME caps + cofactor(s) ({len(cof_symbols)} atoms), {cutoff} A cutoff\n")
        for s, c in zip(symbols, coords):
            f.write(f"{s:<2s} {c[0]: .6f} {c[1]: .6f} {c[2]: .6f}\n")

    if write_pdb:
        with open(output_dir / f"{name}_capped.pdb", "w") as f:
            app.PDBFile.writeFile(final_top, final_modeller.positions, f, keepIds=True)

    # 8. constraints: protein Calpha (as in the base module) + optionally the
    #    cofactor atoms (frozen -- see constrain_cofactor docstring above)
    constraints = [a.index for a in final_top.atoms()
                  if a.name == "CA" and a.residue.name in PROTEIN_RES]
    n_ca_constraints = len(constraints)
    if constrain_metals and selected_metals:
        constraints += [a.index for a in final_top.atoms()
                       if a.residue.name in SUPPORTED_METAL_RESNAMES]
    if constrain_cofactor:
        constraints += [a.index for a in final_top.atoms()
                       if a.residue.name.strip().upper() in cofactor_resnames_upper]

    size = final_top.getNumAtoms()
    print(f"[prep-site+cofactor] wrote {output_xyz} ({size} atoms, "
          f"{n_ca_constraints} C-alpha constraints"
          f"{' + cofactor(s) frozen' if constrain_cofactor else ' + cofactor(s) free'})")
    return {
        "file_location": str(output_xyz),
        "name": name,
        "charge": int(round(charge)),
        "spin": spin,
        "constraints": constraints,
        "size": size,
    }


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(
        description="Prepare a UMA-Dock binding-site XYZ from a PDB whose "
                    "binding site includes a cofactor that must stay in the "
                    "cluster (a heme, an Fe-S cluster, F430, etc.). By default "
                    "auto-discovers every non-protein, non-water, non-bare-metal "
                    "residue and treats it as a cofactor to keep.")
    p.add_argument("pdb", help="input PDB, cofactor(s) already built + protonated")
    p.add_argument("--cofactor-resnames", nargs="+", default=None,
                   help="restrict to exactly these residue names instead of "
                        "auto-discovering (e.g. F31 NI1)")
    p.add_argument("--cofactor-charge", action="append", default=[],
                   metavar="RESNAME:CHARGE",
                   help="net formal charge for one cofactor residue name, "
                        "e.g. HEM:-2 or F31:-4.282. Repeatable. Unlisted "
                        "discovered cofactors default to 0.0 with a warning.")
    p.add_argument("--exclude-resnames", nargs="+", default=[],
                   help="resnames to drop from auto-discovery (crystallization "
                        "additives etc. that showed up but aren't real cofactors)")
    p.add_argument("--ligand-resname", default=None,
                   help="if the pocket is defined by a separate crystallographic "
                        "ligand rather than by the cofactor(s) themselves")
    p.add_argument("-c", "--cutoff", type=float, default=4.0)
    p.add_argument("--ph", type=float, default=7.0)
    p.add_argument("-n", "--name", default=None)
    p.add_argument("--spin", type=int, default=1)
    p.add_argument("-o", "--output-dir", default=".")
    p.add_argument("--chain", default=None)
    p.add_argument("--no-keep-metals", action="store_true")
    p.add_argument("--constrain-metals", action="store_true")
    p.add_argument("--no-constrain-cofactor", action="store_true")
    args = p.parse_args()

    charges = {}
    for item in args.cofactor_charge:
        resname, _, val = item.partition(":")
        charges[resname] = float(val)

    bs = prepare_binding_site_with_cofactor(
        args.pdb, cofactor_charges=charges, cofactor_resnames=args.cofactor_resnames,
        exclude_resnames=args.exclude_resnames, ligand_resname=args.ligand_resname,
        cutoff=args.cutoff, ph=args.ph, name=args.name, spin=args.spin,
        output_dir=args.output_dir, chain=args.chain,
        keep_metals=not args.no_keep_metals, constrain_metals=args.constrain_metals,
        constrain_cofactor=not args.no_constrain_cofactor,
    )
    print("\nbs_object:")
    for k, v in bs.items():
        print(f"  {k}: {v}")
