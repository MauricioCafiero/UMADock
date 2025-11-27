import os
import torch
import numpy as np
from fairchem.core import FAIRChemCalculator, pretrained_mlip

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import CafChem.UMADock as ud

device = "cuda" if torch.cuda.is_available() else "cpu"
molecule = '3np'
charge = 0
spin = 2 
num_steps = 10000
bs = ud.MCR_data
conf_run = 260
outfile = f"{bs['name']}_{molecule}_{conf_run}_{num_steps}steps_md"

print('========================================================')
print(f'MD for {molecule} in {bs['name']}')
print(f'{num_steps} steps')
print(f'charge: {charge}, spin: {spin}')

predictor = pretrained_mlip.get_predict_unit("uma-s-1", device=device)
calculator = FAIRChemCalculator(predictor, task_name="omol")
model = "UMA-OMOL"

filename = f'MCR_w_conf_{conf_run}_OPTIMIZED'
ud.run_md_from_any_xyz(calculator=calculator, bs_object=bs, timestep_fs = 1.0,
        steps = num_steps, output_traj = f"{outfile}.traj", 
        total_spin = spin, total_charge = charge,
        input_file = f"opt_files/{filename}.xyz", log_file = f"{outfile}.log")




