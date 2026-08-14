# UMA-Dock
Docking molecules in protein binding sites using Meta's UMA MLIP as the energy scoring function. Also runs with an AI Agent.
- creates conformations of the input molecule with RDKit
- **Prepares a binding site from any PDB + ligand residue name** (no hand-pruning): finds residues within 4 A of the ligand, caps cut termini with ACE/NME, and assigns ff14SB charges/protonation
- Docks the conformers into the prepared binding site
-  Evaluates pose energy with UMA
-  Optimizes best pose from each conformer
-  Calculates an explcit desolvation energy and a ligand strain energy; combines these with the interaction energy for an electronic binding energy.
-  Chooses best overall, and performs rudimentary dynamics to examine stability.
-  See notebooks folder for Colab examples
-  Needs HuggingFace token and access to the Meta repo.

## Prepare a binding site from any PDB (`prep_binding_site.py`)
UMADock used to require a pre-pruned active-site XYZ (the `*_QM_site.xyz` files in `CafChem/data`, cut and capped by hand). `prep_binding_site.py` now automates that, reusing the OpenMM / PDBFixer machinery from the sibling `openmm` repo:

1. **Repair** the raw PDB with PDBFixer (patch missing heavy atoms; add H at pH; no loop modeling).
2. **Select** the ligand by residue name and keep every protein residue with a heavy atom within a cutoff (default 4 A) of any ligand heavy atom.
3. **Cap** each cut chain end with ACE (N-side) / NME (C-side), placed off the backbone geometry, so the isolated cluster is chemically closed.
4. **Protonate + charge**: the cluster is typed with AMBER ff14SB at the target pH, and the net formal charge is read off the partial-charge sum. `spin` defaults to 1 (closed shell).
5. **Write** the cluster XYZ and return a `bs_object` (`file_location`, `name`, `charge`, `spin`, `constraints`, `size`) ready for `UMA_Dock`. `constraints` are the cap atoms (anchors held fixed during UMA's optimization, mirroring the hand-built sites).

Ligand atoms define the site only; they are not written into the binding-site XYZ (the ligand is docked separately).

### From the command line
```sh
python prep_binding_site.py protein.pdb LIG -c 4 --ph 7 -n MYTARGET -o out/
# -> out/MYTARGET.xyz  (+ out/MYTARGET_capped.pdb for inspection) and a printed bs_object
```

### From Python
```python
import prep_binding_site as prep
bs = prep.prepare_binding_site("protein.pdb", "LIG", cutoff=4.0, ph=7.0, name="MYTARGET", output_dir="out")
# bs is the dict you'd otherwise have to write by hand:
# {'file_location': 'out/MYTARGET.xyz', 'name': 'MYTARGET', 'charge': -2, 'spin': 1, 'constraints': [...], 'size': 550}
```

> Note: UMADock's `get_binding_site_xyz` parser matches a 2-3 digit atom count, so binding sites are currently limited to <1000 atoms. A 4 A single-ligand site is typically well under that (a ~20-residue site is ~500-600 atoms).

## Run end-to-end on Modal (`modal_test.py`)
`modal_test.py` is a **general cloud runner**: dock any small molecule (SMILES) into any protein binding site defined by a crystal ligand in a PDB, entirely on a cloud GPU. It builds the binding site on the fly with `prepare_binding_site` (repair + 4 Å residue selection + ACE/NME capping + ff14SB charges), generates ligand conformers, docks with UMA, optimizes the best pose, and computes the electronic binding energy (optimized interaction + desolvation + strain). The hand-cut `CafChem/data/*_QM_site.xyz` structures are not used by this path (they still work for the original DUDE targets).

The PDB comes from either an RCSB id (`--pdb-id`, fetched inside the container) or a local file (`--pdb-path`, shipped to the container — no image rebuild). Defaults reproduce the validated paracetamol / SULT1A3 test (below).

```sh
pip install modal && modal token new
modal secret put huggingface-secret HF_TOKEN              # one time; needs Meta FAIR-Chem repo access

# default = paracetamol into SULT1A3 (RCSB 2A3R, ligand resname LDP), T4 (~$0.59/hr)
modal run modal_test.py

# any molecule into any target:
modal run modal_test.py --smiles "CC(=O)Nc1ccc(O)cc1" --pdb-id 2A3R --ligand-resname LDP --name SULT1A3
modal run modal_test.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --pdb-path ./my_protein.pdb --ligand-resname LIG --name MYT

# faster GPU, more conformers:
UMADOCK_GPU=A10G modal run modal_test.py --num-confs 20 --number-tries 200
```

GPU is set by the `UMADOCK_GPU` env var (default T4; A10G ~$1.10, A100 ~$2.10/hr) — not a flag, because the flag would be read after the container's GPU is already bound. The UMA model defaults to `uma-s-1p1` (current fairchem-core renamed the bare `uma-s-1`); override with `UMADOCK_UMA_MODEL`. The UMA weights are cached in a Modal Volume after the first run so later runs skip the (multi-GB) download.

**Sampling:** placement tries the ligand center on a Gaussian whose σ is the binding site's spatial spread and keeps a pose only if it lands within 5 Å of the site center. A ~550-atom site has a large σ, so the old 10-try default gives ~0 accepted poses — the script default is now 200 tries. The dock phase (single-point UMA evals) is cheap; only the per-pose optimization + desolvation is costly, so scaling `--number-tries` is nearly free — scale it up for larger sites.

### Validated result: paracetamol / SULT1A3
A full run on a T4 (3 conformers × 200 placement tries, ~30 min, ~$0.30) completed end-to-end. The binding site prepared from 2A3R is **550 atoms, net charge −2, 216 cap-constraint atoms** (20 residues within 4 Å of the dopamine ligand LDP, capped with ACE/NME). Electronic binding energies (kcal/mol):

| conformer | electronic binding energy |
|---|---|
| conf_0 | +17.50 |
| **conf_1 (best, pose 0)** | **+9.87** |
| conf_2 | +11.05 |

The positive (unfavorable) value is chemically reasonable — paracetamol is not a specific SULT1A3 binder (SULT1A3 sulfates catecholamines such as dopamine), and the desolvation penalty dominates at minimal sampling. The run validated the full PDB-forced pipeline: fetch → `prepare_binding_site` → conformers → dock → optimize → desolvation/strain → binding energy.

> Two bugs were fixed during this run and are reflected in the repo: (1) fairchem-core renamed `uma-s-1` → `uma-s-1p1`/`uma-s-1p2`; (2) a NumPy 2.x incompatibility in the solvation step (`get_water_coordinates` stored column-vector arrays that `float()` rejects in NumPy 2.x) — fixed with `.item()` in both `UMADock.py` and `CLI_version/UMADock.py`.

## CLI_version
-  See the CLI_version folder for an implementation to be run from the command line.
-  inludes example scripts for running docking and MD.

## Run from an Agent
See the sample notebook for calling UMADock from an AI agent. 
- Langgraph agent
- Huggingface models (Phi4-mini-instruct)

## Set-up  
(UMADock has it's own dependncies, including RDKit and py3Dmol; see notebooks for examples)

Two tiers of dependencies:
- **`requirements.txt`** (openmm, pdbfixer, rdkit, ase, py3Dmol, numpy, scipy, pandas, matplotlib) — enough to run **`prep_binding_site.py`** and the plotting/analysis helpers.
- **`requirements-mlip.txt`** (torch + fairchem-core) — **required to import `UMADock.py` at all** (it imports torch/fairchem at module top), and to actually score poses with UMA. Needs a HuggingFace token + access to Meta's FAIR-Chem repo (see `requirements-mlip.txt`).

### Local (venv, Python 3.11)
```sh
uv venv --python 3.11 .venv && source .venv/bin/activate
pip install -r requirements.txt          # core
pip install -r requirements-mlip.txt     # torch + fairchem-core (HF token needed for the UMA weights)
```
The root `UMADock.py` imports fine locally (the Colab-only `google.colab` import is guarded). The `CLI_version/UMADock.py` variant has the Colab bits commented out for a pure-CLI run.

### Colab
```
!git clone https://github.com/MauricioCafiero/CafChem.git
!git clone https://github.com/MauricioCafiero/UMADock.git

import torch
import numpy as np
from fairchem.core import FAIRChemCalculator, pretrained_mlip

import UMADock.UMADock as ud

device = "cuda" if torch.cuda.is_available() else "cpu"

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)   # current fairchem renamed "uma-s-1" -> "uma-s-1p1"/"uma-s-1p2"
calculator = FAIRChemCalculator(predictor, task_name="omol")
model = "UMA-OMOL"

```

## Run with mostly defaults
The example below prepares the binding site straight from a PDB + ligand name, then docks. Replace the old hand-built `ud.DRD2_data` / `ud.HMGCR_data` dicts with a `prepare_binding_site` call for any target.
```

def dock_total(smiles: str, pdb_path: str, ligand_resname: str):
  '''
    Dock `smiles` into the binding site defined by `ligand_resname` in `pdb_path`.
    The binding site is prepared on the fly (no hand-pruning needed).
  '''
  test_confs = ud.conformers(smiles,20)
  em_mols = test_confs.get_confs(use_random=True)
  ex_mols = test_confs.expand_conf()
  xyz_strings = test_confs.get_XYZ_strings()
  confs = test_confs.prep_XYZ_docking()

  bs = prep.prepare_binding_site(pdb_path, ligand_resname, cutoff=4.0, name=ligand_resname)
  ldopa_dock = ud.UMA_Dock(confs, 20, calculator, bs)

  new_molecules, ies, distances = ldopa_dock.dock()
  ies, ebes = ldopa_dock.post_process(criteria='distance')

  best_conf_idx = np.argmin(ebes)
  best_energy = ebes[best_conf_idx]

  best_pose_idx = np.argmin(distances[best_conf_idx])

  out_text = f"The lowest elecronic binding energy came from conformer {best_conf_idx}, \
  and pose {best_pose_idx} = {best_energy:.3f} kcal/mol"

  return out_text
```

The pre-built `ud.DRD2_data` / `ud.HMGCR_data` / `ud.MAOB_data` / `ud.ADRB2_data` dicts (pointing at the hand-cut `CafChem/data/*_QM_site.xyz` files) still work for the original DUDE targets; `prepare_binding_site` is the general way to add any new target.

## Sample Output
![UMADock 2JPG](https://github.com/user-attachments/assets/2fc5e47e-ab8a-4fa3-b67c-a31ee0a175d1)

## Agent output
```
query_smiles:  O=C(O)[C@@](NN)(Cc1cc(O)c(O)cc1)C
query_protein: DRD2
The docking calculation results indicate that the molecule represented by the SMILES string O=C(O)[C
@@](NN)(Cc1cc(O)c(O)cc1)C was docked into the DRD2 protein. The specific conformer and pose that res
ulted in the lowest electronic binding energy were identified as conformer 8 and pose 0, respectivel
y. The binding energy for this pose was calculated to be -12.034 kcal/mol. This negative value sugge
sts a favorable interaction between the molecule and the DRD2 protein, indicating that the molecule 
may be a potential ligand or inhibitor that could bind effectively to the DRD2 protein. The docking 
results provide valuable insights into the potential binding affinity and interaction of the molecul
e with the target protein, which can be further investigated for drug discovery and development purp
oses.
```


## To-do list
- Convert all lists/arrays to Numpy or Torch and either compile to C or use GPU
- use as a direct tool for the Langraph agent rather than calling a Gradio client
- Look at defaults for pose selection criteria

