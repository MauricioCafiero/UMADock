import random
import re
import numpy as np
import copy
import math
import py3Dmol
import os
import ase.io
import torch
from google.colab import files
from ase.calculators.calculator import all_changes
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase import Atoms, Atom
import numpy as np
from fairchem.core import FAIRChemCalculator, pretrained_mlip

from rdkit import Chem
from rdkit.Chem import AllChem

global HMGCR_data
HMGCR_data = {
        "file_location":"CafChem/data/HMGCR_dude_QM_site.xyz",
        "name": "HMGCR",
        "charge": 3,
        "spin": 1,
        "constraints": [1, 11, 16, 24, 33, 41, 54, 60, 72, 83, 92, 98, 107, 124, 132, 140, 148, 159, 168, 181],
        "size": 331
        }

global DRD2_data
DRD2_data = {
        "file_location":"CafChem/data/DRD2_dude_QM_site.xyz",
        "name": "DRD2",
        "charge": -1,
        "spin": 1,
        "constraints": [1, 10, 18, 27, 33, 42, 54, 62, 78, 89, 101, 110],
        "size": 216
        }

global MAOB_data
MAOB_data = {
        "file_location":"CafChem/data/MAOB_dude_QM_site.xyz",
        "name": "MAOB",
        "charge": -1,
        "spin": 1,
        "constraints": [1, 7, 12, 17, 22, 31, 39, 44, 51, 61, 67, 84, 89, 94, 111, 120, 129, 134, 139, 147, 161, 172, 180, 188, 197, 206, 218, 235, 242, 250, 256, 265, 272, 281, 295, 306, 322, 335, 342, 356, 361, 370, 375, 389, 398, 408],
        "size": 809
        }

global MAOBnoFAD_data
MAOBnoFAD_data = {
        "file_location":"CafChem/data/MAOBnoFAD_dude_QM_site.xyz",
        "name": "MAOBnoFAD",
        "charge": 1,
        "spin": 1,
        "constraints": [1, 7, 12, 17, 22, 31, 39, 44, 51, 61, 67, 84, 89, 94, 111, 120, 129, 134, 139, 147, 161, 172, 180, 188, 197, 206, 218, 235, 242, 250, 256, 265, 272, 281, 295, 306, 322, 335, 341, 355, 360, 369, 374, 388, 397, 407],
        "size": 727
        }

global ADRB2_data
ADRB2_data = {
        "file_location":"CafChem/data/ADRB2_dude_QM_site.xyz",
        "name": "ADRB2",
        "charge": -2,
        "spin": 1,
        "constraints": [1, 16, 25, 33, 41, 49, 58, 66, 78, 84, 92, 100, 107, 122, 133, 145, 156, 170, 179, 189],
        "size": 349
        }
        
class conformers():
  '''
    Class for generating conformers of a list of molecules
  '''
  def __init__(self,smiles: str, num_confs: int):
    '''
      read in a SMILES string and generate num_confs of conformers

      Args:
        smiles: SMILES string
        num_confs: number of conformers to generate
      Returns:
        None
    '''
    self.smiles = [smiles]
    self.num_confs = num_confs

  def get_confs(self, use_random = False):
    '''
      Generates the conformers with or without using random coordinates.

      Args:
        use_random: boolean, whether to use random coordinates or not
      Returns:
        the embedded molecule objects containing the conformers
    '''
    ps = AllChem.ETKDGv3()
    ps.randomSeed=0xf00d
    ps.numThreads = 2
    if use_random == True:
      ps.useRandomCoords = True

    mols = [Chem.MolFromSmiles(x) for x in self.smiles]
    mols = [Chem.AddHs(m) for m in mols]

    embedded_mols = []
    for i,m in enumerate(mols):
      if not (i+1)%10:
        print(f'Doing {i+1} of {len(mols)}')
      m = Chem.Mol(m)
      AllChem.EmbedMultipleConfs(m,self.num_confs,ps)
      embedded_mols.append(m)

    self.embedded_mols = embedded_mols

    return self.embedded_mols

  def expand_conf(self, mol_idx = 0):
    '''
      Expands the molecule object by extracting each conformer into it's own molecule object.

      Args:
        mol_idx: index of the molecule object to expand
      Returns:
        the expanded molecule objects containing the conformers
    '''
    expanded_confs = []
    for i in range(self.embedded_mols[mol_idx].GetNumConformers()):
      m = Chem.Mol(self.embedded_mols[mol_idx],confId=i)
      expanded_confs.append(m)

    return expanded_confs

  def get_XYZ_strings(self, mol_idx = 0):
    '''
      Generates XYZ strings for each conformer of a molecule object.

      Args:
        mol_idx: index of the molecule object to generate XYZ strings for
      Returns:
        A list of XYZ strings for each conformer of the molecule object
    '''
    expanded_confs = self.expand_conf()
    xyz_strings = []

    for conf in expanded_confs:
      xyz_string = Chem.MolToXYZBlock(conf)
      xyz_strings.append(xyz_string)

    self.xyz_strings = xyz_strings

    return self.xyz_strings

  def prep_XYZ_docking(self, charge = 0, spin =1):
    '''
    '''
    xyz_strings = self.get_XYZ_strings()
    confs_to_dock = []

    num_atoms = int(xyz_strings[0].split()[0])

    atoms_list = []
    for line in xyz_strings[0].split('\n'):
      m = re.search(r'^\D\D?', line)
      if m:
        atoms_list.append(str(line.split()[0]).strip())
    assert num_atoms == len(atoms_list), "Number of atoms in XYZ string does not match number of atoms in molecule object"

    for i,xyz in enumerate(xyz_strings):
      name = f"conf_{i}"
      coords = self.center_XYZ_strings(xyz)
      #print(coords)
      confs_to_dock.append({
            "num_atoms": num_atoms,
            "name": name,
            "atoms": atoms_list,
            "coords": coords,
            "charge": charge,
            "spin": spin,
            "size": 0.0})

    self.confs_to_dock = confs_to_dock
    return self.confs_to_dock

  def center_XYZ_strings(self, frag_raw: str ):
    '''
      Centers the XYZ strings around the origin.

      Args:
        frag_raw: XYZ string to center
      Returns:
        new_frag: list of lists containing the centered coordinates
    '''
    frag_lines = frag_raw.split("\n")
    #print(frag_lines)
    frag = []
    for line in frag_lines[2:-1]:
      vec = []
      parts = line.split()

      for i in range(3):
        vec.append(float(parts[i+1]))
      frag.append(vec)

    center = [0.0]*3
    for row in frag:
      for i in range(3):
        center[i] += row[i]

    for i in range(3):
      center[i] = center[i]/len(frag)

    new_frag = []
    for row in frag:
      tvec = []
      for i in range(3):
        tvec.append(row[i] - center[i])
      new_frag.append(tvec)

    return new_frag

  def make_xyz_files(self, mol_idx):
    '''
      Generates XYZ files for each conformer of a molecule object. The files are saved in the current directory.

      Args:
        mol_idx: index of the molecule object to generate XYZ files for
      Returns:
        None
    '''
    xyz_strings = self.get_XYZ_strings(mol_idx)

    for i,xyz in enumerate(xyz_strings):
      with open(f'conf_{i}.xyz','w') as f:
        f.write(xyz)

class solvation():
  '''
  Class to hold all functions related to adding waters to a molecule
  '''
  def __init__(self, atoms_l, how_many_water_radii = 2.0):
    '''
    Add randomly placed waters to a molecule

      Args:
        atoms: atoms object for the molecule
        how_many_water_radii: number of water radii around the molecules for the
                              box dimensions
    '''
    self.atoms_l = atoms_l
    self.how_many_water_radii = how_many_water_radii
    self.water_vdw_rad = 1.7
    self.water_file_loc = 'temp_files/water_temp.xyz'
    ase.io.write(self.water_file_loc, self.atoms_l, format='xyz')
    print("add_waters class initialized")
  
  def add_box(self, val):
    new_val = val + self.how_many_water_radii*self.water_vdw_rad
    return new_val
    
  def sub_box(self, val):
    new_val = val - self.how_many_water_radii*self.water_vdw_rad
    return new_val
  
  def calc_distance(self,water: list, atom: list):
    '''
    Calculate the distance between the O in the newly added water and an atom
    in the molecule.

      Args:
        water: list of XYZ coordinates for the O atom in water
        atom: list of XYZ coordinates for an atom in the molecule
      Returns:
        distance: distance between the two atoms
    '''
    distance = 0.0
    for i in range(3):
        distance += (float(water[i])-float(atom[i]))**2
    distance = np.sqrt(distance)
    return distance

  def get_box_size(self):
    '''
      Get the box size for the molecule by finding the maximum dimensions and then
      adding a specified number of water van der Waals radii.

        Args:
          None
        Returns:
          max_values: list of maximum XYZ dimensions
          min_values: list of minimum XYZ dimensions
    '''
    f = open(self.water_file_loc,"r")
    lines = f.readlines()
    f.close()
  
    max_values = np.zeros((3))
    min_values = np.zeros((3))
    x_list = []
    y_list = []
    z_list = []
    for row in lines[2:]:
        parts = row.split()
        x_list.append(parts[1]) 
        y_list.append(parts[2])
        z_list.append(parts[3]) 
    max_values[0] = max(x_list)
    max_values[1] = max(y_list)
    max_values[2] = max(z_list)
    min_values[0] = min(x_list)
    min_values[1] = min(y_list)
    min_values[2] = min(z_list)
    
    max_values = [self.add_box(val) for val in max_values]
    min_values = [self.sub_box(val) for val in min_values]
    
    return max_values, min_values
  
  def get_water_coordinates(self,o_coordinates: list):
    '''
    Takes coordinates for an oxygen atom, adds two H-atoms, translates the molecule
    to the new location and adds a random rotation.

      Args:
        o_coordinates: oxygen atom coordinates
      Returns:
        rot_water_xyz: string of coordinates for the water molecule
        rot_water_coordinates: list of coordinates for the water molecule
    '''
    o_xyz = np.asarray([0.0, 0.0, 0.0]).reshape(3,1)
    h1_xyz = np.asarray([0.580743,  0.000000,  0.758810]).reshape(3,1)
    h2_xyz = np.asarray([0.580743,  0.000000,  -0.758810]).reshape(3,1)

    theta_x = random.uniform(0,2*np.pi)
    theta_y = random.uniform(0,2*np.pi)
    theta_z = random.uniform(0,2*np.pi)
    
    x_rotation_matrix = np.asarray(([1.0,0.0,0.0],[0.0,np.cos(theta_x),-np.sin(theta_x)],[0.0,np.sin(theta_x),np.cos(theta_x)])).reshape(3,3)
    y_rotation_matrix = np.asarray(([np.cos(theta_y),0.0,-np.sin(theta_y)],[0.0,1.0,0.0],[np.sin(theta_y),0.0,np.cos(theta_y)])).reshape(3,3)
    z_rotation_matrix = np.asarray(([np.cos(theta_z),-np.sin(theta_z),0.0],[np.sin(theta_z),np.cos(theta_z),0.0],[0.0,0.0,1.0])).reshape(3,3)
    
    rot_h1_xyz = np.matmul(x_rotation_matrix,h1_xyz)
    rot_h1_xyz = np.matmul(y_rotation_matrix,rot_h1_xyz)
    rot_h1_xyz = np.matmul(z_rotation_matrix,rot_h1_xyz)
    rot_h2_xyz = np.matmul(x_rotation_matrix,h2_xyz)
    rot_h2_xyz = np.matmul(y_rotation_matrix,rot_h2_xyz)
    rot_h2_xyz = np.matmul(z_rotation_matrix,rot_h2_xyz)
        
    for i in range(3):
        o_xyz[i] = o_xyz[i] + o_coordinates[i]
        rot_h1_xyz[i] = rot_h1_xyz[i] + o_coordinates[i]
        rot_h2_xyz[i] = rot_h2_xyz[i] + o_coordinates[i]
    
    rot_water_xyz =  f"O {o_xyz[0].item()}   {o_xyz[1].item()}    {o_xyz[2].item()}\n"
    rot_water_xyz += f"H {rot_h1_xyz[0].item()}   {rot_h1_xyz[1].item()}    {rot_h1_xyz[2].item()}\n"
    rot_water_xyz += f"H {rot_h2_xyz[0].item()}   {rot_h2_xyz[1].item()}    {rot_h2_xyz[2].item()}\n"

    rot_water_coordinates = []
    rot_water_coordinates.append([o_xyz[0], o_xyz[1], o_xyz[2]])
    rot_water_coordinates.append([rot_h1_xyz[0], rot_h1_xyz[1], rot_h1_xyz[2]])
    rot_water_coordinates.append([rot_h2_xyz[0], rot_h2_xyz[1], rot_h2_xyz[2]])

    return rot_water_xyz, rot_water_coordinates

  def add_waters(self, max_waters: int, stopping_criteria = 10):
    '''
    Adds water molecules up to max waters. Tries randomly adding waters and fails 
    if it is too close to an existing atom. 

      Args:
        max_waters: maximum waters to add
        stopping_criteria: number of failed attempts before stopping
      Returns:
        molecule_text: string of coordinates for the molecule with waters
    '''
    f = open(self.water_file_loc,"r")
    lines = f.readlines()
    f.seek(0)
    molecule_text = f.read()
    f.close()
  
    max_values, min_values = self.get_box_size()

    waters_to_add = []
    add_water = True

    water_counter = 0
    fail_counter = []
    
    for _ in range(max_waters):
        add_water = True

        if len(fail_counter) >= stopping_criteria:
            if 0 not in fail_counter[-stopping_criteria:]:
                print("Stopping criteria met! Exiting water addition")
                break
        
        new_water = []
        for i in range(3):
            new_water.append(random.uniform(min_values[i],max_values[i]))
           
        for row in lines[2:]:
            parts = row.split()
            mol_vec = [parts[1],parts[2],parts[3]]
            distance = self.calc_distance(new_water, mol_vec)
            if distance < self.water_vdw_rad:
                #print(f"distance: {distance} is close to another atom, breaking loop")
                fail_counter.append(1)
                add_water = False
                break
                
        if len(waters_to_add) > 0:
          for water in waters_to_add:
            for row in water:
                distance = self.calc_distance(new_water,row)
                if distance < self.water_vdw_rad:
                    #print(f"distance: {distance} is close to another water, breaking loop")
                    fail_counter.append(1)
                    add_water = False
                    break
                
        if add_water:
            water_counter += 1
            fail_counter.append(0)
            new_water_string, new_water_coordinates = self.get_water_coordinates(new_water)
            molecule_text += new_water_string
            waters_to_add.append(new_water_coordinates)

    old_length = int(lines[0])
    new_length = old_length + 3*water_counter
    molecule_text = f"{new_length}" + molecule_text[2:]
    f = open('temp_files/solvated_conf.xyz',"w")
    f.write(molecule_text)
    f.close()

    print(f"Added {water_counter}/{max_waters} waters.")
    print("==================================================")

class UMA_Dock():
  '''
    Class for docking a ligand into a binding site
  '''
  def __init__(self, frags: list,number_tries: int, calculator, bs_object: dict):
    '''
      Main function for docking a ligand into a binding site

        Args:
            frags: list of conformer dictionaries
            number_tries: how many times at attempt fragment placement
            calculator: ASE calculator
            bs_object: dictionary for the binding site
    '''
    self.frags = frags
    self.number_tries = number_tries
    self.calculator = calculator
    self.bs_object = bs_object

    self.get_binding_site_xyz()

  def get_binding_site_xyz(self):
    '''
        opens a XYZ file with a binding site and loads it

        Args:
            None
        Returns:
            all_molecules: array with binding site coordinates
            atoms_symbols: list of atoms symbols
    '''
    f = open(self.bs_object['file_location'], "r")

    start_token = ""
    start_token = re.compile(r"^\d\d(\d)?\s")

    total_input = f.readlines()
    f.close()

    number_molecules = 0
    for line in total_input:
        start_match = start_token.search(line)
        if start_match:
            number_molecules += 1
            molecule_size = line.strip("\n")

    molecule_size = int(molecule_size)
    print(f"There are {number_molecules} molecules with size: {molecule_size}")

    all_molecules = []
    temp_array = np.zeros((molecule_size,3))
    all_molecules.append(temp_array)

    current_line = 0
    atom_symbols = []

    internal_i = 0
    current_line += 2
    print(f"for {current_line}, {current_line+int(molecule_size)}")

    for j in range(current_line,current_line+molecule_size,1):
        temp_vec = total_input[j].split()
        atom_symbols.append(temp_vec[0])
        for k in range(1,4,1):
            all_molecules[0][internal_i,k-1] = temp_vec[k]
        internal_i += 1
        current_line += 1

    self.all_molecules = all_molecules
    self.atom_symbols = atom_symbols

  def get_binding_site_center(self):
      '''
          Uses the array of atom positions to calculate the center of the binding site
          and the standard deviation of the center.

          Args:
            None

          Returns:
              centers: array, center of binding site
              sigmas: np.array, standard deviation of binding site center coordinates
      '''
      centers = [0.0]*3
      sigmas = [0.0]*3
      for row in self.all_molecules[0]:
        for i in range(3):
          centers[i] += row[i]

      for i in range(3):
          centers[i] = centers[i]/len(self.all_molecules[0])

      for row in self.all_molecules[0]:
        for i in range(3):
          sigmas[i] += (row[i] - centers[i])**2

      for i in range(3):
          sigmas[i] = np.sqrt(sigmas[i]/(len(self.all_molecules[0])-1))

      return centers,sigmas

  def get_frag_coordinates(self, new_origin: list, which_frag: dict):
    '''
        Given an origin set of coordinates for a new fragment, this function
        reads in the standard fragment coordinates, rotates them randomly
        in each direction, and then translates them to the new origin.

        Args:
            new_origin: the location of the new fragment
            which_frag: a dictionary for the fragment in question.
        Returns:
            atoms_list: a list of coordinates for the new fragment
    '''

    how_many_atoms = which_frag["num_atoms"]

    theta_x = random.uniform(0,2*np.pi)
    theta_y = random.uniform(0,2*np.pi)
    theta_z = random.uniform(0,2*np.pi)

    x_rotation_matrix = np.asarray(([1.0,0.0,0.0],[0.0,np.cos(theta_x),-np.sin(theta_x)],[0.0,np.sin(theta_x),np.cos(theta_x)])).reshape(3,3)
    y_rotation_matrix = np.asarray(([np.cos(theta_y),0.0,-np.sin(theta_y)],[0.0,1.0,0.0],[np.sin(theta_y),0.0,np.cos(theta_y)])).reshape(3,3)
    z_rotation_matrix = np.asarray(([np.cos(theta_z),-np.sin(theta_z),0.0],[np.sin(theta_z),np.cos(theta_z),0.0],[0.0,0.0,1.0])).reshape(3,3)

    atoms_list = []
    for vec in which_frag["coords"]:
        temp_vec = np.array(vec)

        rot_temp = np.matmul(x_rotation_matrix,temp_vec)
        rot_temp = np.matmul(y_rotation_matrix,rot_temp)
        rot_temp = np.matmul(z_rotation_matrix,rot_temp)

        temp_vec = rot_temp + new_origin
        temp_vec = list(temp_vec)
        atoms_list.append(temp_vec)

    return atoms_list

  def calc_distance(self, ref: list, atom: list):
    '''
        Calculates the distance between two points

        Args:
            ref: a reference point
            atom: the new atom coordinates
        Returns:
            distance: the distance
    '''
    distance = 0.0
    for i in range(3):
        distance += (ref[i]-atom[i])**2
    distance = np.sqrt(distance)
    return distance

  def calc_frag_energy(self, new_molecule: list, frag: dict):
    '''
        Calculates the energies of the fragments in the binding site

        Args:
            new_molecules: list of arrays of coordinates of the binding site and fragment
            frag: the dictionary for the fragment in question
        Returns:
            ies: the interaction energies between the binding site and the fragment
    '''
    path = "temp_files/"
    test_files = ["complex.xyz"]

    all_symbols = self.atom_symbols + frag["atoms"]

    f = open(path+test_files[0],"w")
    f.write(f"{len(all_symbols)}\n")
    f.write("\n")

    for i in range(len(new_molecule)):
        row_string = f"{all_symbols[i]}"
        for coord in new_molecule[i]:
            row_string += f"    {coord}"
        if i != len(new_molecule):
            f.write(row_string+"\n")
        else:
            f.write(row_string)
    f.close()

    atoms_tot = ase.io.read(path+test_files[0], format="xyz")
    total_spin = self.bs_object['spin'] + frag['spin'] - 1
    total_charge = self.bs_object['charge'] + frag['charge']
    atoms_tot.info.update({"spin": total_spin, "charge": total_charge})
    os.remove(path+test_files[0])

    bs_length = len(new_molecule) - frag["num_atoms"]
    atoms_bs = atoms_tot[:bs_length]
    atoms_bs.info.update({"spin": self.bs_object['spin'], "charge": self.bs_object['charge']})
    atoms_l = atoms_tot[bs_length:]
    atoms_l.info.update({"spin": frag['spin'], "charge": frag['charge']})
    atoms_to_calc = [atoms_tot,atoms_bs,atoms_l]

    results = []
    for atoms in atoms_to_calc:
        atoms.calc = self.calculator
        results.append(atoms.get_potential_energy())

    ie = 23.06035*(results[0] - results[1] - results[2])

    return ie

  def distance_too_short(self, binding_site, ligand):
    '''
      Calculate the distance between atoms and return True if they are closer than the
      cutoff value of 1.4

        Args:
          bindingsite: list of binding site atom corrdinates
          ligand: list of ligand atom coordinates
        Returns:
          True if the distance is less than 1.4, False otherwise
    '''
    for row in binding_site:
      for conf_row in ligand:
        atom_atom_distance = self.calc_distance(row,conf_row)
        if atom_atom_distance < 1.4:
          return True
    return False

  def nudge_conf(self, conf: np.array):
    '''
      Add a random number between 1 and -1 to each coordinate for each atom in
      the ligand

        Args:
          conf: list of ligand atom coordinates
    '''
    rn_eps = np.random.uniform(low=-1.0,high=1.0,size=3)
    for row in conf:
      for i in range(3):
        row[i] += rn_eps[i]

    return conf

  def dock(self):
      '''
          Main function for docking a ligand into a binding site

          Args:
              None
          Returns:
              new_molecules: list of positions of successfully placed fragments
      '''
      try:
        os.mkdir("temp_files")
      except:
        print("temp_files directory already exists")

      new_molecules = []
      ies = []
      distances = []

      centers, sigmas = self.get_binding_site_center()

      for mi,molecule in enumerate(self.all_molecules):
          for frag in self.frags:
              new_sheet = []
              ie_sheet = []
              distance_sheet = []

              how_many_added = 0
              for _ in range(self.number_tries):
                  add_mol = False
                  new_mol_origin = np.empty((3))
                  cutoff = 5.0
                  steric = 1.0

                  for i in range(3):
                      new_mol_origin[i] = np.random.normal(loc = centers[i], scale = sigmas[i])

                  new_vec = self.get_frag_coordinates(new_mol_origin, frag)

                  '''
                  Calculate the distance from the center of the ligand the center of the binding site. We
                  want this distance to be fairly small to get the right binding mode.
                  '''
                  conf_centers = [0.0]*3
                  for row in new_vec:
                    for i in range(3):
                      conf_centers[i] += row[i]

                  for i in range(3):
                      conf_centers[i] = conf_centers[i]/len(new_vec)

                  conf_distance = self.calc_distance(conf_centers,centers)

                  '''
                  Combine new conformer and binding site into one object and calculate UMA
                  binding energy
                  '''
                  new_vec = np.array(new_vec)
                  single_frag = np.append(molecule, new_vec, axis=0)
                  ie = self.calc_frag_energy(single_frag, frag)

                  #print(f'Distance is: {conf_distance:.3f} and binding energy is: {ie:.3f}.')

                  '''
                  Keep only poses that have small distances and decent energies
                  '''
                  if ie < 2000.0 and conf_distance < 5:
                    '''
                    Check of the ligand overlaps with binding site atoms. if it does, set the nudge flag and nudge the conf
                    randomly. If it lowers the energy check if the overlap has been eliminated. Keep it if it no longer
                    has overlap and has lowered the energy.
                    '''
                    add_mol = True
                    nudge_flag = False
                    atom_atom_distance = self.distance_too_short(molecule, new_vec)
                    if atom_atom_distance == True:
                      nudge_flag = True
                      add_mol = False
                      #print('setting nudge flag')

                    if nudge_flag:
                        print('nudging')
                        for nudge_idx in range(50):
                          nudged_vec = self.nudge_conf(new_vec)
                          nudged_vec = np.array(nudged_vec)
                          nudged_frag = np.append(molecule, nudged_vec, axis=0)
                          nudged_ie = self.calc_frag_energy(nudged_frag, frag)

                          if nudged_ie < ie:
                            atom_atom_distance = self.distance_too_short(molecule, nudged_vec)
                            if atom_atom_distance == False:
                              single_frag = nudged_frag
                              ie = nudged_ie
                              add_mol = True
                              print(f'Nudged distance is: {conf_distance:.3f} and binding energy is: {ie:.3f}.')
                              break
                            else:
                              new_vec = nudged_vec

                  if add_mol:
                        how_many_added += 1
                        print(f"adding fragment: {frag['name']}")
                        new_sheet.append(single_frag)
                        ie_sheet.append(ie)
                        distance_sheet.append(conf_distance)
                      # else:
                      #     print("fragment rejected due to repulsive energy.")

              print(f"Added {how_many_added} {frag['name']} fragments")
              print("================================================")
              new_molecules.append(new_sheet)
              ies.append(ie_sheet)
              distances.append(distance_sheet)
              self.new_molecules = new_molecules
              self.ies = ies
              self.distances = distances

      return new_molecules, ies, distances

  def save_xyz_files(self):
      '''
          accepts the new_molecules array and creates an XYZ file for each fragment placement

          Args:
              None
          Returns:
              None; saves XYZ files
      '''
      try:
        os.mkdir("frag_files")
      except:
        print("frag_files directory already exists")

      try:
          files = os.listdir("frag_files")
          files_to_remove = [file for file in files if (os.path.splitext(file)[1]==".xyz")]
          for file in files_to_remove:
            os.remove(f"frag_files/{file}")
      except:
          print("frag_files directory is empty")

      for k,frag in enumerate(self.frags):
          for j in range(len(self.new_molecules[k])):
              mol_file = f"frag_files/{self.bs_object['name']}_w_{frag['name']}{j}.xyz"

              all_symbols = self.atom_symbols + frag["atoms"]
              f = open(mol_file,"w")
              f.write(f"{len(all_symbols)}\n")
              f.write("\n")
              for i in range(len(self.new_molecules[k][j])):
                  row_string = f"{all_symbols[i]}"
                  for coord in self.new_molecules[k][j][i]:
                      row_string += f"    {coord}"
                  if i != len(self.new_molecules[k][j]):
                      f.write(row_string+"\n")
                  else:
                      f.write(row_string)
              f.close()
              #print("File Written")

          print(f"{len(self.new_molecules[k])} files written for {frag['name']}.")

  def view_frag_pose(self, frag_idx: int, pose_idx: int):
      '''
        Displays a fragment pose

        Args:
          frag_idx: index of the fragment to view
          pose_idx: index of the pose to view
        Returns:
          None; displays fragment pose
      '''
      frag_name = self.frags[frag_idx]["name"]
      view_file = mol_file = f"frag_files/{self.bs_object['name']}_w_{frag_name}{pose_idx}.xyz"
      f = open(view_file,"r")
      lines = f.readlines()
      mol_data = "".join(lines)
      f.close

      viewer = py3Dmol.view(width=800, height=400)
      viewer.addModel(mol_data, "xyz")  # Add the trajectory frame

      for i in range(self.bs_object["size"]):
        viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5}})

      for i in range(self.bs_object["size"],self.bs_object['size']+self.frags[frag_idx]["num_atoms"],1):
        viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5, 'color': 'green'}})

      viewer.zoomTo()
      viewer.show()

  def get_best_poses(self):
      '''
        Accepts the list of interaction energies and determines the best pose for each fragment

        Args:
          None
        Returns:
          best_pose_for_fragments: list of best poses for each fragment
      '''
      best_pose_for_fragments = []
      for energy,frag in zip(self.ies,self.frags):
          if len(energy) != 0:
            min = np.min(energy)
            min_idx = np.argmin(energy)
            best_pose_for_fragments.append(min_idx)
            print(f"best pose for {frag['name']} is: {min:.3f} at location: {min_idx}")
            files.download(f"frag_files/{self.bs_object['name']}_w_{frag['name']}{min_idx}.xyz")
          else:
            best_pose_for_fragments.append(-1)
            print(f"No poses for {frag['name']}")

      self.best_pose_by_energy = best_pose_for_fragments
      return best_pose_for_fragments

  def get_best_poses_by_distance(self):
      '''
        Accepts the list of distances and determines the best pose for each fragment

        Args:
          None
        Returns:
          best_pose_for_fragments: list of best poses for each fragment
      '''
      best_pose_by_distance = []
      for distance,frag in zip(self.distances,self.frags):
          if len(distance) != 0:
            min = np.min(distance)
            min_idx = np.argmin(distance)
            best_pose_by_distance.append(min_idx)
            print(f"best pose by distance for {frag['name']} is: {min:.3f} at location: {min_idx}")
            files.download(f"frag_files/{self.bs_object['name']}_w_{frag['name']}{min_idx}.xyz")
          else:
            best_pose_by_distance.append(-1)
            print(f"No poses for {frag['name']}")

      self.best_pose_by_distance = best_pose_by_distance
      return best_pose_by_distance

  def opt_conformation(self, conf_file_path: str, frag: dict):
      '''
          Calculates the energies of the fragments in the binding site

          Args:
              conf_file_path: path to the conformation file
              frag: the dictionary for the fragment in question
          Returns:
              ie: the interaction energies between the binding site and the fragment
      '''
      print("=================================================")
      print(f"Optimizing best pose for fragment {frag['name']}.")
      atoms_tot = ase.io.read(conf_file_path, format="xyz")
      total_spin = self.bs_object['spin'] + frag['spin'] - 1
      total_charge = self.bs_object['charge'] + frag['charge']
      atoms_tot.info.update({"spin": total_spin, "charge": total_charge})
      atoms_tot.calc = self.calculator

      initial_energy = atoms_tot.get_potential_energy()
      print(f"Initial energy: {0.0367493*initial_energy:.6f} ha")

      c = FixAtoms(indices = self.bs_object['constraints'])
      atoms_tot.set_constraint(c)

      opt = BFGS(atoms_tot)
      opt.run(fmax=0.30)
      energy = atoms_tot.get_potential_energy()
      print(f"Final energy: {0.0367493*energy:.6f} ha")
      print(f"Energy difference: {0.0367493*(energy-initial_energy):.6f} ha")

      atoms_opt = atoms_tot.copy()
      bs_length = self.bs_object['size']
      atoms_bs = atoms_tot[:bs_length]
      atoms_bs.info.update({"spin": self.bs_object['spin'], "charge": self.bs_object['charge']})
      atoms_l = atoms_tot[bs_length:]
      atoms_l.info.update({"spin": frag['spin'], "charge": frag['charge']})
      atoms_to_calc = [atoms_tot,atoms_bs,atoms_l]

      results = []
      for atoms in atoms_to_calc:
          atoms.calc = self.calculator
          results.append(atoms.get_potential_energy())

      ie = 23.06035*(results[0] - results[1] - results[2])
      print(f'new IE after optimization = {ie} kcal/mol')

      return ie, atoms_opt, atoms_l

  def desolvation_strain(self, atoms_ligand, frag):
    '''
      Calculated the desolvation and strain energies for the bound ligand

      Args:
        atoms_ligand: atoms object for the optimized ligand
        frag: the dictionary for the fragment in question
      Returns:
        strain_energy: the strain energy
        desolvation_energy: the desolvation energy
    '''
    solvate = solvation(atoms_ligand,2)
    solvate.add_waters(20)
    solv_atoms = ase.io.read('temp_files/solvated_conf.xyz', format='xyz')
    solv_atoms.info.update({"spin": frag['spin'], "charge": frag['charge']})
    solv_atoms.calc = self.calculator

    bound_ligand = solv_atoms[:frag['num_atoms']]
    bound_ligand.info.update({"spin": frag['spin'], "charge": frag['charge']})
    bound_ligand.calc = self.calculator
    bound_energy = bound_ligand.get_potential_energy()

    opt = BFGS(solv_atoms)
    opt.run(fmax=0.30)
  
    just_ligand = solv_atoms[:frag['num_atoms']]
    just_ligand.info.update({"spin": frag['spin'], "charge": frag['charge']})

    waters = solv_atoms[frag['num_atoms']:]
    waters.info.update({"spin": 1, "charge": 0})
    waters.calc = self.calculator
    opt_water = BFGS(waters)
    opt_water.run(fmax=0.30)

    solv_atoms_to_calc = [solv_atoms, just_ligand, waters]
    results = []
    for atoms in solv_atoms_to_calc:
        atoms.calc = self.calculator
        results.append(atoms.get_potential_energy())
    desolvation_energy = 23.06035*(results[1] + results[2] - results[0])
    print(f'Desolvation energy = {desolvation_energy} kcal/mol')

    strain_energy = 23.06035*(bound_energy - results[1])
    print(f'Strain energy = {strain_energy} kcal/mol')

    return strain_energy, desolvation_energy

  def post_process(self, criteria = 'distance'):
    '''
      After docking, this routine saves poses to XYZ files, gets the best poses 
      by either energy or distance from ligand to binding site center, and optimizes the best
      pose for each configuration. The ligand desolvation and train energies are calculated and
      added to the binding optimized interaction energy to produce an electronic
      binding energy.

        Args:
          criteria: 'energy' or 'distance'
        Returns:
          opt_ies: list of optimized interaction energies
          elec_bind_es: list of electronic binding energies
    '''
    try:
        os.mkdir("opt_files")
    except:
      print("opt_files directory already exists")

    try:
        files = os.listdir("opt_files")
        files_to_remove = [file for file in files if (os.path.splitext(file)[1]==".xyz")]
        for file in files_to_remove:
          os.remove(f"opt_files/{file}")
    except:
        print("opt_files directory is empty")

    self.save_xyz_files()
    if criteria == 'energy':
      poses =  self.get_best_poses()
    elif criteria == 'distance':
      poses = self.get_best_poses_by_distance()
    else:
      print('Invalid criteria')
      return

    opt_ies = []
    elec_bind_es = []
    for pose, frag in zip(poses, self.frags):
      if pose != -1:
        opt_e, atoms, atoms_ligand = self.opt_conformation(f"frag_files/{self.bs_object['name']}_w_{frag['name']}{pose}.xyz", frag)
        ase.io.write(f"opt_files/{self.bs_object['name']}_w_{frag['name']}{pose}_OPTIMIZED.xyz", atoms, 'xyz')
        opt_ies.append(opt_e)

        strain_energy, desolvation_energy = self.desolvation_strain(atoms_ligand, frag)
        elec_bind_es.append(opt_e+strain_energy+desolvation_energy)
        print(f'Total energy = {opt_e+strain_energy+desolvation_energy} kcal/mol')
        
      else:
        opt_ies.append(-1)
        elec_bind_es.append(-1)
        print(f"No poses for {frag['name']}")

    return opt_ies, elec_bind_es

def view_from_file(filename: str, bs_object, frag):
      '''
        Displays a fragment pose

        Args:
          filename: path to the conformation file
          bs_object: the dictionary for the binding site
          frag: the dictionary for the fragment in question
        Returns:
          None; displays fragment pose
      '''
      view_file =  filename
      f = open(view_file,"r")
      lines = f.readlines()
      mol_data = "".join(lines)
      f.close

      viewer = py3Dmol.view(width=800, height=400)
      viewer.addModel(mol_data, "xyz")  # Add the trajectory frame

      for i in range(bs_object["size"]):
        viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5}})

      for i in range(bs_object["size"],bs_object['size']+frag["num_atoms"],1):
        viewer.setStyle({'model': -1, 'serial': i}, {"stick": {}, "sphere": {"radius": 0.5, 'color': 'green'}})

      viewer.zoomTo()
      viewer.show()
