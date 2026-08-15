"""End-to-end UMA-Dock on Modal -- general runner.

Dock any small molecule (SMILES) into any protein binding site defined by a
crystal ligand in a PDB. The binding site is built ON THE FLY from the PDB via
``prep_binding_site.prepare_binding_site`` (repair + 4 A residue selection +
ACE/NME capping + ff14SB charges), NOT from the hand-cut ``CafChem/data/*_QM_site.xyz``
cached structures (those still exist in UMADock.py and remain usable for the
original DUDE targets).

Run it with (defaults reproduce the validated paracetamol / SULT1A3 test):

    modal run code/modal_test.py                                       # paracetamol into SULT1A3 (2A3R)
    modal run code/modal_test.py --smiles "CC(=O)Nc1ccc(O)cc1" --pdb-id 2A3R --ligand-resname LDP --name SULT1A3
    modal run code/modal_test.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --pdb-path ./my_protein.pdb --ligand-resname LIG --name MYT
    modal run code/modal_test.py --model mace-omol --smiles "CC(=O)Nc1ccc(O)cc1" --pdb-id 2A3R --ligand-resname LDP --name SULT1A3

The PDB comes from EITHER:
  --pdb-id <rcsbId>   fetched inside the container from RCSB (self-contained), or
  --pdb-path <file>   a local PDB whose contents are shipped to the container
                      (no image rebuild needed; the file text is passed as an arg).

GPU is chosen by the UMADOCK_GPU env var (read when Modal builds the app), not a
flag -- the flag would be set after the @app.function gpu= is already bound:
    modal run code/modal_test.py                         # T4 (default)
    UMADOCK_GPU=A10G modal run code/modal_test.py        # ~2x price, ~2x faster

GPU cost (modal.com/pricing, per hour):
    T4   ~ $0.59  (16 GB) -- cheapest; UMA-s fits easily. Recommended for tests.
    L4   ~ $0.80  (24 GB)
    A10G ~ $1.10  (24 GB)
    A100 ~ $2.10-2.50  (40/80 GB)
    H100 ~ $3.95

Scoring model (--model):
    uma         (default) Meta's UMA via FAIRChem (omol task). The UMA model id
                defaults to ``uma-s-1p1`` (current fairchem-core renamed the bare
                ``uma-s-1`` -> ``uma-s-1p1``/``uma-s-1p2``); override with
                UMADOCK_UMA_MODEL. Needs the HF secret + Meta FAIR-Chem access.
    mace-omol   MACE-OMOL-0 (the MACE analog of UMA's omol task; 89 elements). The
                MACE variant to try for protein clusters (broader training than
                MACE-OFF23). Override the checkpoint URL with UMADOCK_MACE_OMOL_URL.
    ('mace-off23' is blocked: retested against the corrected single-chain SULT1A3
    site on 2026-08-15 -- still diverges (fmax -> ~1e8, oscillating) in float64 on
    A100, the same failure mode as on the original buggy dimer-fused site. Out of
    distribution for capped protein clusters regardless of site correctness. Use
    `uma`, `mace-omol`, or `aimnet2`.)
    aimnet2     AIMNet2 (isayevlab/aimnetcentral), model 'aimnet2-2025' by default
                (override via UMADOCK_AIMNET2_MODEL). Covers H, B, C, N, O, F, Si,
                P, S, Cl, As, Se, Br, I; self-validates element coverage. Weights
                download from Hugging Face on first use.

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

Caching & output (Modal Volumes):
    - umadock-data-cache (/root/data): reusable cache -- data/pdbs/<id>.pdb (raw
      PDBs, fetched once from RCSB) and data/sites/<name>.xyz + <name>.meta.json
      (prepared binding sites; deterministic for a given PDB+resname+cutoff+ph, so
      a repeat target skips the PDBFixer repair/select/cap/charge prep).
    - umadock-runs (/root/output): per-run output/<name>_<model>_<UTC>/ with
      opt_files/<...>_OPTIMIZED.xyz (the final bound complex) + result.json. These
      persist so the final pose is NOT lost with the ephemeral container.
    Retrieving the final pose: an attached `modal run` writes the best pose +
      result.json into the LOCAL repo `output/` tree. For a --detach run that
      outlives the local client, pull them from the volume instead, e.g.
      `modal volume get umadock-runs /<name>_<model>_<UTC>/result.json .`
    The binding energy itself is always printed to `modal app logs <app-id>`.
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

# Resolve sibling repo modules relative to THIS file so `modal run code/modal_test.py`
# works from any CWD. The library lives in the importable package `code/UMADock/`; the
# in-container destination `/root/UMADock/` is unchanged so `import UMADock.UMADock`
# still resolves exactly as before.
_HERE = Path(__file__).resolve().parent
_PKG = _HERE / "UMADock"

GPU = os.environ.get("UMADOCK_GPU", "T4")

app = modal.App("umadock-dock")

# Cache model weights across runs so we don't re-download every time.
# UMA (fairchem) -> HF cache; MACE models -> ~/.cache/mace (its default path).
hf_cache = modal.Volume.from_name("umadock-hf-cache", create_if_missing=True)
mace_cache = modal.Volume.from_name("umadock-mace-cache", create_if_missing=True)
# Reusable cache: raw PDBs (data/pdbs/) + prepared binding sites (data/sites/).
# Persists across runs so a repeat target skips the RCSB fetch + the PDBFixer
# repair/select/cap/charge prep (deterministic for a given PDB+resname+cutoff+ph).
data_cache = modal.Volume.from_name("umadock-data-cache", create_if_missing=True)
# Per-run outputs (optimized poses, result.json) on a persistent volume so the
# final bound complex is NOT lost with the ephemeral container -- retrievable via
# `modal volume get umadock-runs ...` even from a detached run (lid closed).
runs_cache = modal.Volume.from_name("umadock-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "wget", "ca-certificates")  # libgomp1 for openmm/rdkit
    .env({
        "MPLBACKEND": "Agg",                       # headless plotting
        "HF_HOME": "/root/.cache/huggingface",     # on the cached volume
        # PyTorch 2.6 made torch.load default to weights_only=True, which rejects
        # the fairchem UMA checkpoint (it pickles a `slice` global). Force the
        # pre-2.6 behavior so both UMA (fairchem) and MACE load cleanly. (mace
        # sets this itself; fairchem does not, so the UMA path needs it here.)
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    })
    # core (prep + conformers + analysis)
    .pip_install(
        "openmm>=8.2", "pdbfixer>=1.9", "rdkit>=2024.3", "ase>=3.23",
        "numpy", "scipy", "pandas", "matplotlib", "py3Dmol",
    )
    # MLIP tier -- torch + fairchem (UMA) is required to import UMADock.py;
    # mace-torch enables the mace-omol scoring alternative.
    .pip_install("torch")
    .pip_install("fairchem-core")
    .pip_install("mace-torch>=0.3.14")
    .pip_install("aimnet[ase]")
    # ship the repo modules as an importable package `UMADock` in the container.
    # Source paths resolve relative to this file (code/), so the run is CWD-independent.
    .add_local_file(str(_PKG / "__init__.py"), "/root/UMADock/__init__.py")
    .add_local_file(str(_PKG / "UMADock.py"), "/root/UMADock/UMADock.py")
    .add_local_file(str(_PKG / "prep_binding_site.py"), "/root/UMADock/prep_binding_site.py")
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/mace": mace_cache,
        "/root/data": data_cache,
        "/root/output": runs_cache,
    },
)
def run_test(smiles: str, ligand_resname: str, name: str,
             num_confs: int = 5, number_tries: int = 200,
             criteria: str = "distance", cutoff: float = 4.0, ph: float = 7.0,
             model: str = "uma", mace_size: str = "medium", mace_dtype: str = "float64",
             pdb_id: str | None = None, pdb_text: str | None = None,
             chain: str | None = None) -> dict:
    """Dock `smiles` into the binding site defined by `ligand_resname` in a PDB.

    Provide exactly one of `pdb_id` (fetch from RCSB) or `pdb_text` (raw PDB
    contents, e.g. read from a local file in the entrypoint).
    `model` selects the scorer: 'uma' (default) | 'mace-off23' | 'mace-omol' | 'aimnet2'.
    """
    import os
    import json
    import urllib.request
    from datetime import datetime, timezone

    assert pdb_id or pdb_text, "provide --pdb-id or --pdb-path"

    os.chdir("/root")
    os.makedirs("/root/data/pdbs", exist_ok=True)
    os.makedirs("/root/data/sites", exist_ok=True)

    import numpy as np
    import torch
    import UMADock.UMADock as ud
    from UMADock import prep_binding_site as prep

    # per-run output dir on the persistent runs volume: output/<name>_<model>_<UTC>/
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = f"/root/output/{name}_{model}_{stamp}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"[dock] run output dir: {run_dir}")

    # 1. obtain the PDB -- cache raw PDBs on the data volume so a repeat target
    #    skips the RCSB fetch. (pdb_text mode bypasses the cache -- no pdb_id.)
    if pdb_text:
        pdb_path = "/root/input.pdb"
        print(f"[dock] using supplied PDB text ({len(pdb_text)} bytes)")
        with open(pdb_path, "w") as f:
            f.write(pdb_text)
    else:
        pdb_path = f"/root/data/pdbs/{pdb_id}.pdb"
        if os.path.exists(pdb_path):
            print(f"[dock] using cached PDB {pdb_id} (data/pdbs/)")
        else:
            print(f"[dock] downloading RCSB {pdb_id} -> {pdb_path} (caching for reuse)")
            urllib.request.urlretrieve(
                f"https://files.rcsb.org/download/{pdb_id}.pdb", pdb_path)

    # 2. build (or reuse) the binding site. The prepared site is deterministic for a
    #    given (pdb_id, ligand_resname, cutoff, ph, chain), so cache it under
    #    data/sites/ with a .meta.json -- reuse on a param-matching hit, re-prep on
    #    miss/mismatch. ``chain`` picks one ligand copy (one chain) so an oligomer
    #    with one ligand per monomer does NOT get its pockets fused into one site.
    site_xyz = f"/root/data/sites/{name}.xyz"
    site_meta = f"/root/data/sites/{name}.meta.json"
    bs = None
    if os.path.exists(site_xyz) and os.path.exists(site_meta):
        with open(site_meta) as f:
            meta = json.load(f)
        if (meta.get("pdb_id") == pdb_id
                and meta.get("ligand_resname") == ligand_resname
                and meta.get("cutoff") == cutoff
                and meta.get("ph") == ph
                and meta.get("chain") == chain):
            print(f"[dock] using cached prepared site '{name}' (data/sites/) "
                  f"chain={meta.get('chain')}")
            bs = {"file_location": site_xyz, "name": name,
                  "charge": meta["charge"], "spin": meta["spin"],
                  "constraints": meta["constraints"], "size": meta["size"]}
    if bs is None:
        print(f"[dock] preparing binding site '{name}' from {pdb_id or 'supplied PDB'}"
              f" (chain={chain or 'first ligand copy'})")
        bs = prep.prepare_binding_site(
            pdb_path, ligand_resname, cutoff=cutoff, ph=ph,
            name=name, output_dir="/root/data/sites", chain=chain)
        with open(site_meta, "w") as f:
            json.dump({"pdb_id": pdb_id, "ligand_resname": ligand_resname,
                       "cutoff": cutoff, "ph": ph, "chain": chain,
                       "charge": bs["charge"], "spin": bs["spin"],
                       "constraints": bs["constraints"], "size": bs["size"]}, f)
    print(f"[dock] binding site '{name}': {bs['size']} atoms, charge={bs['charge']}, "
          f"spin={bs['spin']}, {len(bs['constraints'])} Cα constraints")

    # 3. calculator (UMA by default; or MACE-OFF23 / MACE-OMOL-0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dock] loading scorer '{model}' on {device} (first run downloads weights)")
    calculator = ud.build_calculator(model, device=device, mace_size=mace_size, mace_dtype=mace_dtype)

    # 4. query-ligand conformers
    print(f"[dock] generating {num_confs} conformers for {smiles}")
    confs_obj = ud.conformers(smiles, num_confs)
    confs_obj.get_confs(use_random=True)
    confs_obj.get_XYZ_strings()
    confs = confs_obj.prep_XYZ_docking()           # neutral, singlet defaults

    # 5. dock + post-process; run artifacts write under the per-run output dir
    print(f"[dock] docking ({number_tries} placement tries/conformer)")
    dock = ud.UMA_Dock(confs, number_tries, calculator, bs, output_dir=run_dir)
    dock.dock()
    opt_ies, ebes = dock.post_process(criteria=criteria)

    # persist the reusable cache (PDBs + prepared sites) for next time
    data_cache.commit()

    valid = [i for i, e in enumerate(ebes) if e != -1]
    if not valid:
        print("[dock] WARNING: no valid poses -- increase --number-tries or --num-confs")
        runs_cache.commit()
        hf_cache.commit()
        mace_cache.commit()
        return {"ok": False, "smiles": smiles, "name": name, "model": model,
                "run_dir": run_dir,
                "opt_ies": list(map(float, opt_ies)), "ebes": list(map(float, ebes))}

    best_conf = int(np.argmin(ebes))
    best_pose = int(np.argmin(dock.distances[best_conf]))
    best_energy = float(ebes[best_conf])
    best_frag_name = dock.frags[best_conf]["name"]

    # 6. retrieve the final optimized bound complex so it is NOT lost with the
    #    ephemeral container -- read its XYZ text into the result dict (also lives
    #    on the persistent runs volume under <run_dir>/opt_files/).
    best_xyz_path = os.path.join(
        dock.opt_dir, f"{name}_w_{best_frag_name}{best_pose}_OPTIMIZED.xyz")
    best_xyz_text = ""
    if os.path.exists(best_xyz_path):
        with open(best_xyz_path) as f:
            best_xyz_text = f.read()

    result = {
        "ok": True,
        "smiles": smiles,
        "name": name,
        "model": model,
        "run_dir": run_dir,
        "binding_site": {k: v for k, v in bs.items() if k != "constraints"},
        "num_confs": num_confs,
        "number_tries": number_tries,
        "best_conformer": best_conf,
        "best_pose": best_pose,
        "best_optimized_xyz": best_xyz_text,
        "electronic_binding_energy_kcal_mol": best_energy,
        "interaction_energies": [float(e) for e in opt_ies],
        "strain_energies": [float(e) for e in dock.strain_energies],
        "desolvation_energies": [float(e) for e in dock.desolvation_energies],
        "all_ebes": [float(e) for e in ebes],
    }
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump({k: v for k, v in result.items() if k != "best_optimized_xyz"},
                  f, indent=2)

    print(f"\n[dock] ===== RESULT =====")
    print(f"[dock] {name} / {smiles}")
    print(f"[dock] best conformer={best_conf} pose={best_pose} "
          f"electronic binding energy = {best_energy:.3f} kcal/mol")
    print(f"[dock] best pose: {best_xyz_path}")
    print(f"[dock] breakdown: interaction={float(opt_ies[best_conf]):.3f} "
          f"strain={float(dock.strain_energies[best_conf]):.3f} "
          f"desolvation={float(dock.desolvation_energies[best_conf]):.3f} kcal/mol")

    runs_cache.commit()
    hf_cache.commit()
    mace_cache.commit()
    return result


@app.local_entrypoint()
def main(smiles: str = "CC(=O)Nc1ccc(O)cc1",   # paracetamol
        pdb_id: str = "2A3R",                  # SULT1A3 + dopamine (LDP) + PAP
        pdb_path: str = None,                  # local PDB file (overrides --pdb-id)
        ligand_resname: str = "LDP",           # crystal ligand defining the pocket
        name: str = "SULT1A3",
        model: str = "uma",                    # uma | mace-off23 | mace-omol | aimnet2
        mace_size: str = "medium",             # small|medium|large (mace-off23)
        mace_dtype: str = "float64",           # float64 (precise) | float32 (fast); slow on T4 -- use A100/H100
        num_confs: int = 5, number_tries: int = 200,
        criteria: str = "distance", cutoff: float = 4.0, ph: float = 7.0,
        chain: str = None):                    # ligand copy's chain (None = first copy; set for oligomers)
    """Dock `smiles` into the binding site of `ligand_resname` in a PDB.
    Defaults reproduce the validated paracetamol / SULT1A3 test. GPU is selected
    by the UMADOCK_GPU env var (default T4); see the module docstring. ``--chain``
    picks one ligand copy (one chain) so an oligomer's pockets are not fused."""
    pdb_text = None
    if pdb_path:
        with open(pdb_path) as f:
            pdb_text = f.read()
        print(f"[main] using local PDB: {pdb_path}")
    result = run_test.remote(
        smiles=smiles, ligand_resname=ligand_resname, name=name,
        num_confs=num_confs, number_tries=number_tries,
        criteria=criteria, cutoff=cutoff, ph=ph,
        model=model, mace_size=mace_size, mace_dtype=mace_dtype,
        pdb_id=None if pdb_path else pdb_id, pdb_text=pdb_text, chain=chain,
    )
    print("\n===== SUMMARY =====")
    for k, v in result.items():
        if k == "best_optimized_xyz":
            print(f"  {k}: <{len(v)} chars>")
        else:
            print(f"  {k}: {v}")

    # materialize the retrieved best pose + result.json into the local output/ tree.
    # (For a --detach run that outlives the local client, retrieve the same files
    # from the persistent runs volume: `modal volume get umadock-runs <path> <local>`.)
    if result.get("ok") and result.get("best_optimized_xyz"):
        import json as _json
        from pathlib import Path
        run_name = Path(result.get("run_dir", f"/{name}_{model}")).name
        local_run = Path("output") / run_name
        local_run.mkdir(parents=True, exist_ok=True)
        bc, bp = result["best_conformer"], result["best_pose"]
        (local_run / f"{name}_w_conf_{bc}{bp}_OPTIMIZED.xyz").write_text(
            result["best_optimized_xyz"])
        (local_run / "result.json").write_text(_json.dumps(
            {k: v for k, v in result.items() if k != "best_optimized_xyz"}, indent=2))
        print(f"\n[main] wrote best pose + result.json -> {local_run}/")


@app.local_entrypoint()
def spawn_main(smiles: str = "CC(=O)Nc1ccc(O)cc1",   # paracetamol
               pdb_id: str = "2A3R",                  # SULT1A3 + dopamine (LDP) + PAP
               pdb_path: str = None,                  # local PDB file (overrides --pdb-id)
               ligand_resname: str = "LDP",           # crystal ligand defining the pocket
               name: str = "SULT1A3",
               model: str = "uma",                    # uma | mace-off23 | mace-omol | aimnet2
               mace_size: str = "medium",             # small|medium|large (mace-off23)
               mace_dtype: str = "float64",           # float64 (precise) | float32 (fast)
               num_confs: int = 5, number_tries: int = 200,
               criteria: str = "distance", cutoff: float = 4.0, ph: float = 7.0,
               chain: str = None):                    # ligand copy's chain (None = first copy; set for oligomers)
    """Fire-and-forget: spawn ``run_test`` on Modal's servers and EXIT immediately.

    Unlike ``main`` (which calls ``.remote()`` and keeps a local client alive
    streaming logs for the whole run), this calls ``.spawn()`` and returns, so the
    local ``modal run`` process exits within seconds. The spawned call runs
    server-side, independent of this process -- a lid-close / laptop sleep / lost
    WiFi CANNOT cancel it (the earlier ``--detach`` failures were the long-lived
    streaming client getting frozen by sleep and hard-killed; spawning removes that
    client entirely). Use this for any run you walk away from.

    Launch detached so the ephemeral app persists after this process exits:

        UMADOCK_GPU=A100 modal run --detach code/modal_test.py::spawn_main \\
            --model mace-omol --mace-dtype float64 --num-confs 3 --number-tries 200 \\
            --pdb-id 2A3R --ligand-resname LDP --name SULT1A3 \\
            --smiles "CC(=O)Nc1ccc(O)cc1"

    Retrieve the result later (the run also persists result.json + the optimized
    pose to the umadock-runs volume, and prints the binding energy to its logs):

        modal app list                 # find the app id (ephemeral, detached, 1 task)
        modal app logs <app-id>        # binding energy + breakdown printed here
        modal volume get umadock-runs /<name>_<model>_<UTC>/result.json .
        modal volume get umadock-runs /<name>_<model>_<UTC>/opt_files/<...>_OPTIMIZED.xyz .
    """
    pdb_text = None
    if pdb_path:
        with open(pdb_path) as f:
            pdb_text = f.read()
        print(f"[spawn] using local PDB: {pdb_path}")
    fc = run_test.spawn(
        smiles=smiles, ligand_resname=ligand_resname, name=name,
        num_confs=num_confs, number_tries=number_tries,
        criteria=criteria, cutoff=cutoff, ph=ph,
        model=model, mace_size=mace_size, mace_dtype=mace_dtype,
        pdb_id=None if pdb_path else pdb_id, pdb_text=pdb_text, chain=chain,
    )
    print(f"[spawn] SPAWNED call_id={fc.object_id}")
    print(f"[spawn] local process exiting now; call runs server-side (lid-close safe).")
    print(f"[spawn] monitor: modal app list   |   logs: modal app logs <app-id>")
    print(f"[spawn] result:   modal volume get umadock-runs /<run_dir>/result.json .")