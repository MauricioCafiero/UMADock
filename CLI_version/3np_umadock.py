import os
import torch
import numpy as np
from fairchem.core import FAIRChemCalculator, pretrained_mlip

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import CafChem.UMADock as ud

print('============================================')
print('UMADock - by Mauricio Cafiero')
print('')

device = "cuda" if torch.cuda.is_available() else "cpu"

predictor = pretrained_mlip.get_predict_unit("uma-s-1", device=device)
calculator = FAIRChemCalculator(predictor, task_name="omol")
model = "UMA-OMOL"
print(f'Loaded the {model} model. ')
print(f'Running on {device}')
print('')

'''
ligands:
vanilic acid   'COc1cc(ccc1O)C(=O)O' 
rosmarininc acid   'O=C(O)C(OC(=O)\C=C\c1ccc(O)c(O)c1)Cc2cc(O)c(O)cc2'  
3-nitroxypropanol   'C(CO)CO[N+](=O)[O-]'
cofactor M    'SCCS(=O)(=O)[O-]'
'''

test_mol =  'C(CO)CO[N+](=O)[O-]'
charge = 0
print(f'Ligand is: {test_mol}')

test_confs = ud.conformers(test_mol,40)
em_mols = test_confs.get_confs(use_random=True)
confs = test_confs.prep_XYZ_docking(charge = charge)
bs = ud.MCR_data
num_confs = 40

print(f'Binding site: {bs['name']}')
print(f'Generating {num_confs} conformations for the ligand.')
print('')

ldopa_dock = ud.UMA_Dock(confs, num_confs, calculator, bs)

new_molecules, ies, distances = ldopa_dock.dock()

ies, ebes = ldopa_dock.post_process(criteria='distance')
ldopa_dock.show_best()

ldopa_dock.run_md_from_xyz(steps = 10_000, output_traj = "best_md.traj", log_file = "best_md.log")




