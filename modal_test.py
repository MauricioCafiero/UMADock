"""End-to-end UMA-Dock on Modal -- general runner.

Dock any small molecule (SMILES) into any protein binding site defined by a
crystal ligand in a PDB. The binding site is built ON THE FLY from the PDB via
``prep_binding_site.prepare_binding_site`` (repair + 4 A residue selection +
ACE/NME capping + ff14SB charges), NOT from the hand-cut ``CafChem/data/*_QM_site.xyz``
cached structures (those still exist in UMADock.py and remain usable for the
original DUDE targets).

Run it with (defaults reproduce the validated paracetamol / SULT1A3 test):

    modal run modal_test.py                                       # paracetamol into SULT1A3 (2A3R)
    modal run modal_test.py --smiles "CC(=O)Nc1ccc(O)cc1" --pdb-id 2A3R --ligand-resname LDP --name SULT1A3
    modal run modal_test.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --pdb-path ./my_protein.pdb --ligand-resname LIG --name MYT

The PDB comes from EITHER:
  --pdb-id <rcsbId>   fetched inside the container from RCSB (self-contained), or
  --pdb-path <file>   a local PDB whose contents are shipped to the container
                      (no image rebuild needed; the file text is passed as an arg).

GPU is chosen by the UMADOCK_GPU env var (read when Modal builds the app), not a
flag -- the flag would be set after the @app.function gpu= is already bound:
    modal run modal_test.py                         # T4 (default)
    UMADOCK_GPU=A10G modal run modal_test.py        # ~2x price, ~2x faster

GPU cost (modal.com/pricing, per hour):
    T4   ~ $0.59  (16 GB) -- cheapest; UMA-s fits easily. Recommended for tests.
    L4   ~ $0.80  (24 GB)
    A10G ~ $1.10  (24 GB)
    A100 ~ $2.10-2.50  (40/80 GB)
    H100 ~ $3.95

UMA model id defaults to ``uma-s-1p1`` (current fairchem-core renamed the bare
``uma-s-1`` -> ``uma-s-1p1`` / ``uma-s-1p2``). Override with UMADOCK_UMA_MODEL.

SAMPLING NOTE: placement tries the ligand center on a Gaussian whose sigma is
the binding site's spatial spread, and keeps a pose only if the ligand lands
within 5 A of the site center. A ~550-atom site has a large sigma, so the old
10-try default gives ~0 accepted poses -- use >= 150-200 tries for real sites.
The dock phase (single-point UMA evals) is cheap; only the per-pose optimization
+ desolvation is costly, so scaling --number-tries is nearly free.

Prereqs (one time):
    1. pip install modal  &&  modal token new
    2. Create a Modal secret (here named "huggingface-secret") holding your HF
       read token, e.g.  modal secret put huggingface-secret HF_TOKEN
       You must also have been granted access to Meta's FAIR-Chem UMA model repo
       on HuggingFace (see requirements-mlip.txt). The UMA weights are cached in
       a Modal Volume so later runs skip the (multi-GB) download.
"""
from __future__ import annotations

import os

import modal

GPU = os.environ.get("UMADOCK_GPU", "T4")
UMA_MODEL = os.environ.get("UMADOCK_UMA_MODEL", "uma-s-1p1")

app = modal.App("umadock-dock")

# Cache the (multi-GB) UMA weights across runs so we don't re-download every time.
hf_cache = modal.Volume.from_name("umadock-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "wget", "ca-certificates")  # libgomp1 for openmm/rdkit
    .env({
        "MPLBACKEND": "Agg",                       # headless plotting
        "HF_HOME": "/root/.cache/huggingface",     # on the cached volume
    })
    # core (prep + conformers + analysis)
    .pip_install(
        "openmm>=8.2", "pdbfixer>=1.9", "rdkit>=2024.3", "ase>=3.23",
        "numpy", "scipy", "pandas", "matplotlib", "py3Dmol",
    )
    # MLIP tier -- required to import UMADock.py and to score poses
    .pip_install("torch")
    .pip_install("fairchem-core")
    # ship the repo modules as an importable package `UMADock`
    .add_local_file("UMADock.py", "/root/UMADock/UMADock.py")
    .add_local_file("prep_binding_site.py", "/root/UMADock/prep_binding_site.py")
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_test(smiles: str, ligand_resname: str, name: str,
             num_confs: int = 5, number_tries: int = 200,
             criteria: str = "distance", cutoff: float = 4.0, ph: float = 7.0,
             pdb_id: str | None = None, pdb_text: str | None = None) -> dict:
    """Dock `smiles` into the binding site defined by `ligand_resname` in a PDB.

    Provide exactly one of `pdb_id` (fetch from RCSB) or `pdb_text` (raw PDB
    contents, e.g. read from a local file in the entrypoint).
    """
    import os
    import urllib.request

    assert pdb_id or pdb_text, "provide --pdb-id or --pdb-path"

    os.chdir("/root")
    for d in ("out", "temp_files", "frag_files", "opt_files"):
        os.makedirs(d, exist_ok=True)

    import numpy as np
    import torch
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    import UMADock.UMADock as ud
    from UMADock import prep_binding_site as prep

    # 1. obtain the PDB
    pdb_path = "/root/input.pdb"
    if pdb_text:
        print(f"[dock] using supplied PDB text ({len(pdb_text)} bytes)")
        with open(pdb_path, "w") as f:
            f.write(pdb_text)
    else:
        print(f"[dock] downloading RCSB {pdb_id} -> {pdb_path}")
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb_id}.pdb", pdb_path)

    # 2. build the binding site FROM THE PDB (ligand atoms define the pocket but
    #    are not written into the site XYZ -- the query ligand is docked into it)
    bs = prep.prepare_binding_site(
        pdb_path, ligand_resname, cutoff=cutoff, ph=ph,
        name=name, output_dir="/root/out",
    )
    print(f"[dock] binding site '{name}': {bs['size']} atoms, charge={bs['charge']}, "
          f"spin={bs['spin']}, {len(bs['constraints'])} cap constraints")

    # 3. UMA calculator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dock] loading {UMA_MODEL} on {device} (first run downloads weights)")
    predictor = pretrained_mlip.get_predict_unit(UMA_MODEL, device=device)
    calculator = FAIRChemCalculator(predictor, task_name="omol")

    # 4. query-ligand conformers
    print(f"[dock] generating {num_confs} conformers for {smiles}")
    confs_obj = ud.conformers(smiles, num_confs)
    confs_obj.get_confs(use_random=True)
    confs_obj.get_XYZ_strings()
    confs = confs_obj.prep_XYZ_docking()           # neutral, singlet defaults

    # 5. dock + post-process (opt best pose, desolvation, strain -> binding energy)
    print(f"[dock] docking ({number_tries} placement tries/conformer)")
    dock = ud.UMA_Dock(confs, number_tries, calculator, bs)
    dock.dock()
    opt_ies, ebes = dock.post_process(criteria=criteria)

    valid = [i for i, e in enumerate(ebes) if e != -1]
    if not valid:
        print("[dock] WARNING: no valid poses -- increase --number-tries or --num-confs")
        hf_cache.commit()
        return {"ok": False, "smiles": smiles, "name": name,
                "opt_ies": list(map(float, opt_ies)), "ebes": list(map(float, ebes))}

    best_conf = int(np.argmin(ebes))
    best_pose = int(np.argmin(dock.distances[best_conf]))
    best_energy = float(ebes[best_conf])
    print(f"\n[dock] ===== RESULT =====")
    print(f"[dock] {name} / {smiles}")
    print(f"[dock] best conformer={best_conf} pose={best_pose} "
          f"electronic binding energy = {best_energy:.3f} kcal/mol")
    print(f"[dock] best pose: /root/opt_files/{name}_w_conf_{best_conf}{best_pose}_OPTIMIZED.xyz")

    hf_cache.commit()
    return {
        "ok": True,
        "smiles": smiles,
        "name": name,
        "binding_site": {k: v for k, v in bs.items() if k != "constraints"},
        "num_confs": num_confs,
        "number_tries": number_tries,
        "best_conformer": best_conf,
        "best_pose": best_pose,
        "electronic_binding_energy_kcal_mol": best_energy,
        "all_ebes": [float(e) for e in ebes],
    }


@app.local_entrypoint()
def main(smiles: str = "CC(=O)Nc1ccc(O)cc1",   # paracetamol
        pdb_id: str = "2A3R",                  # SULT1A3 + dopamine (LDP) + PAP
        pdb_path: str = None,                  # local PDB file (overrides --pdb-id)
        ligand_resname: str = "LDP",           # crystal ligand defining the pocket
        name: str = "SULT1A3",
        num_confs: int = 5, number_tries: int = 200,
        criteria: str = "distance", cutoff: float = 4.0, ph: float = 7.0):
    """Dock `smiles` into the binding site of `ligand_resname` in a PDB.
    Defaults reproduce the validated paracetamol / SULT1A3 test. GPU is selected
    by the UMADOCK_GPU env var (default T4); see the module docstring."""
    pdb_text = None
    if pdb_path:
        with open(pdb_path) as f:
            pdb_text = f.read()
        print(f"[main] using local PDB: {pdb_path}")
    result = run_test.remote(
        smiles=smiles, ligand_resname=ligand_resname, name=name,
        num_confs=num_confs, number_tries=number_tries,
        criteria=criteria, cutoff=cutoff, ph=ph,
        pdb_id=None if pdb_path else pdb_id, pdb_text=pdb_text,
    )
    print("\n===== SUMMARY =====")
    for k, v in result.items():
        print(f"  {k}: {v}")