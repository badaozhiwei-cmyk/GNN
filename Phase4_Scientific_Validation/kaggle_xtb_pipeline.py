"""
=====================================================================
 GFN2-xTB Unified Descriptor Generation Pipeline
 For: Kaggle Notebook (Linux environment)
=====================================================================
"""

import pandas as pd
import numpy as np
import os
import subprocess
import re
from rdkit import Chem
from rdkit.Chem import AllChem

# --- KAGGLE ENVIRONMENT FIX ---
# Grimme's xTB tarball extracts to a folder named "xtb-dist"
xtb_root = '/kaggle/working/xtb-dist'
if os.path.exists(xtb_root):
    os.environ['PATH'] = f"{xtb_root}/bin:" + os.environ.get('PATH', '')
    os.environ['XTBPATH'] = f"{xtb_root}/share/xtb"
else:
    print(f"[WARNING] Could not find xTB installation at {xtb_root}. Subprocess may fail if xtb is not in PATH.")

# --- CONFIGURATION (Relative paths for Kaggle git clone workflow) ---
# When you run `!python Phase4_Scientific_Validation/kaggle_xtb_pipeline.py`
# from the /kaggle/working/GNN directory, Python's working directory is /kaggle/working/GNN
INPUT_CSV = 'index_with_anion.csv'
OUTPUT_CSV = 'Phase4_Scientific_Validation/xTB_Physics_Descriptors.csv'
LOG_DIR = 'Phase4_Scientific_Validation/xtb_logs'
WORK_DIR = 'Phase4_Scientific_Validation/xtb_work'

VEGA_REFRIGERANTS = {
    'R41', 'R32', 'R23', 'R161', 'R152a', 'R134a', 'R125',
    'R245fa', 'R236fa', 'R227ea', 'R1234yf', 'R1234ze(E)',
    'R1336mzz(Z)', 'R1233zd(E)'
}

def safe_filename(name):
    return re.sub(r'[^\w\-.]', '_', name)

def generate_xyz(smiles, molecule_name, output_xyz):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  [ERROR] RDKit cannot parse SMILES: {smiles}")
        return False

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol, params)
    if res == -1:
        res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        if res == -1:
            print(f"  [ERROR] Cannot embed 3D coords for {molecule_name}")
            return False

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass

    Chem.MolToXYZFile(mol, output_xyz)
    return True

def parse_xtb_output(output_text):
    results = {'Dipole_Debye': None, 'Polarizability_au': None, 'Volume_Bohr3': None, 'Total_Energy_Eh': None}

    dipole_matches = re.findall(r'full:\s+([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)\s+([\d.]+)', output_text)
    if dipole_matches:
        results['Dipole_Debye'] = float(dipole_matches[-1][3])

    alpha_match = re.search(r'Mol\.\s+alpha\(0\)\s+/au\s*:?\s+([\d.]+)', output_text)
    if alpha_match:
        results['Polarizability_au'] = float(alpha_match.group(1))

    vol_match = re.search(r'molecular volume.*?:\s+([\d.]+)\s+', output_text)
    if vol_match:
        results['Volume_Bohr3'] = float(vol_match.group(1))

    energy_match = re.search(r'TOTAL ENERGY\s+([\-\d.]+)\s+Eh', output_text)
    if energy_match:
        results['Total_Energy_Eh'] = float(energy_match.group(1))

    return results

def run_xtb(name, smiles, charge, category, work_dir, log_dir):
    safe_name = safe_filename(name)
    mol_work = os.path.join(work_dir, safe_name)
    os.makedirs(mol_work, exist_ok=True)

    xyz_file = os.path.join(mol_work, f"{safe_name}.xyz")
    log_file = os.path.join(log_dir, f"{safe_name}.log")

    print(f"\n{'='*60}\n[{category}] {name}  (charge={charge})\n  SMILES: {smiles}")

    if not generate_xyz(smiles, name, xyz_file):
        with open(log_file, 'w') as f:
            f.write(f"FAILED: RDKit could not generate 3D geometry for {smiles}\n")
        return None

    print(f"  Step 2: Running geometry optimization...")
    opt_cmd = ['xtb', xyz_file, '--opt', 'tight', '--chrg', str(charge), '--gfn', '2', '--namespace', safe_name]
    opt_result = subprocess.run(opt_cmd, capture_output=True, text=True, cwd=mol_work, timeout=600)

    opt_xyz = os.path.join(mol_work, 'xtbopt.xyz')
    if not os.path.exists(opt_xyz):
        alt_opt = os.path.join(mol_work, f'{safe_name}.xtbopt.xyz')
        if os.path.exists(alt_opt):
            opt_xyz = alt_opt
        else:
            opt_xyz = xyz_file

    print(f"  Step 3: Computing polarizability on optimized geometry...")
    sp_cmd = ['xtb', opt_xyz, '--sp', '--alpha', '--chrg', str(charge), '--gfn', '2']
    sp_result = subprocess.run(sp_cmd, capture_output=True, text=True, cwd=mol_work, timeout=600)

    full_output = opt_result.stdout + "\n" + sp_result.stdout

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== MOLECULE: {name} ===\nSMILES: {smiles}\nCharge: {charge}\nCategory: {category}\n\n")
        f.write("=== OPT STDOUT ===\n" + opt_result.stdout + "\n=== OPT STDERR ===\n" + opt_result.stderr)
        f.write("\n=== SP+ALPHA STDOUT ===\n" + sp_result.stdout + "\n=== SP+ALPHA STDERR ===\n" + sp_result.stderr)

    parsed = parse_xtb_output(full_output)
    print(f"  Results: μ={parsed['Dipole_Debye']}, α={parsed['Polarizability_au']}, V={parsed['Volume_Bohr3']}")
    return parsed

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Cannot find input dataset at {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    refrigerants = df[['refrigerant', 'refri_smiles']].drop_duplicates().values
    cations = df[['cation', 'cation_smiles']].drop_duplicates().values
    anions = df[['anion', 'anion_smiles']].drop_duplicates().values

    all_results = []

    print("\n\n>>> PHASE 1: REFRIGERANTS (charge=0) <<<")
    for name, smi in refrigerants:
        res = run_xtb(name, smi, charge=0, category='Refrigerant', work_dir=WORK_DIR, log_dir=LOG_DIR)
        if res:
            res.update({'Molecule': name, 'SMILES': smi, 'Category': 'Refrigerant', 'Charge': 0, 'Source': 'GFN2-xTB', 'Has_Vega_Literature': name in VEGA_REFRIGERANTS})
            all_results.append(res)

    print("\n\n>>> PHASE 2: CATIONS (charge=+1) <<<")
    for name, smi in cations:
        res = run_xtb(name, smi, charge=1, category='Cation', work_dir=WORK_DIR, log_dir=LOG_DIR)
        if res:
            res.update({'Molecule': name, 'SMILES': smi, 'Category': 'Cation', 'Charge': 1, 'Source': 'GFN2-xTB', 'Has_Vega_Literature': False})
            all_results.append(res)

    print("\n\n>>> PHASE 3: ANIONS (charge=-1) <<<")
    for name, smi in anions:
        res = run_xtb(name, smi, charge=-1, category='Anion', work_dir=WORK_DIR, log_dir=LOG_DIR)
        if res:
            res.update({'Molecule': name, 'SMILES': smi, 'Category': 'Anion', 'Charge': -1, 'Source': 'GFN2-xTB', 'Has_Vega_Literature': False})
            all_results.append(res)

    out_df = pd.DataFrame(all_results)
    out_df = out_df[['Category', 'Molecule', 'SMILES', 'Charge', 'Dipole_Debye', 'Polarizability_au', 'Volume_Bohr3', 'Total_Energy_Eh', 'Source', 'Has_Vega_Literature']]
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ All computations complete! Saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
