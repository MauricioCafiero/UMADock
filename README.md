# UMA-Dock
Docking molecules in protein binding sites using Meta's UMA MLIP as the energy scoring function. Also runs with an AI Agent.
- creates conformations of the input molecule with RDKit
- Docks in a pruned version of the DUDE protein structures
-  Evaluates pose energy with UMA
-  Optimizes best pose from each conformer
-  Calculates an explcit desolvation energy and a ligand strain energy; combines these with the interaction energy for an electronic binding energy.
-  Chooses best overall, and performs rudimentary dynamics to examine stability.
-  See notebooks folder for Colab examples
-  Needs HuggingFace token and access to the Meta repo.

### Run from an Agent
See the sample notebook for calling UMADock from an AI agent. 
- Langgraph agent
- Huggingface models (Phi4-mini-instruct)

## Set-up  
(UMADock has it's own dependncies, including RDKit and py3Dmol; see notebooks for examples)

```
!git clone https://github.com/MauricioCafiero/CafChem.git
!git clone https://github.com/MauricioCafiero/UMADock.git

import torch
import numpy as np
from fairchem.core import FAIRChemCalculator, pretrained_mlip

import UMADock.UMADock as ud

device = "cuda" if torch.cuda.is_available() else "cpu"

predictor = pretrained_mlip.get_predict_unit("uma-s-1", device=device)
calculator = FAIRChemCalculator(predictor, task_name="omol")
model = "UMA-OMOL"

```

## Run with mostly defaults
```

def dock_total(smiles: str, target: str):
  '''
    
  '''
  test_confs = ud.conformers(smiles,20)
  em_mols = test_confs.get_confs(use_random=True)
  ex_mols = test_confs.expand_conf()
  xyz_strings = test_confs.get_XYZ_strings()
  confs = test_confs.prep_XYZ_docking()

  if target == "DRD2":
    ldopa_dock = ud.UMA_Dock(confs, 20, calculator, ud.DRD2_data)
  
  elif target == "HMGCR":
    ldopa_dock = ud.UMA_Dock(confs, 20, calculator, ud.HMGCR_data)
  
  elif target == "MAOB":
    ldopa_dock = ud.UMA_Dock(confs, 20, calculator, ud.MAOB_data)
  
  elif target == "ADRB2":
    ldopa_dock = ud.UMA_Dock(confs, 20, calculator, ud.ADRB2_data)

  new_molecules, ies, distances = ldopa_dock.dock()
  ies, ebes = ldopa_dock.post_process(criteria='distance')

  best_conf_idx = np.argmin(ebes)
  best_energy = ebes[best_conf_idx]

  best_pose_idx = np.argmin(distances[best_conf_idx])

  out_text = f"The lowest elecronic binding energy came from conformer {best_conf_idx}, \
  and pose {best_pose_idx} = {best_energy:.3f} kcal/mol"

  return out_text
```

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
## Dynamics output (10 fs, alpha carbons fixed)
![Watch the video here!](https://github.com/MauricioCafiero/UMADock/blob/main/video/DRD2_Carbidopa_dynamics_10fs.mp4)


## To-do list
- Convert all lists/arrays to Numpy or Torch and either compile to C or use GPU
- use as a direct tool for the Langraph agent rather than calling a Gradio client
- Look at defaults for pose selection criteria

