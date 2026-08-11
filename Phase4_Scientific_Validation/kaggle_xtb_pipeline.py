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
xtb_root = '/kaggle/working/xtb-dist'
if os.path.exists(xtb_root):
    os.environ['PATH'] = f"{xtb_root}/bin:" + os.environ.get('PATH', '')
    os.environ['XTBPATH'] = f"{xtb_root}/share/xtb"
else:
    print(f"[WARNING] Could not find xTB installation at {xtb_root}.")

# --- CONFIGURATION ---
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
        mmff_res = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        mmff_res = -1  # MMFF params unavailable on some RDKit versions
    if mmff_res != 0:
        if mmff_res == 1:
            # MMFF ran but didn't converge — re-embed to avoid half-optimized coords
            AllChem.EmbedMolecule(mol, params)
        try:
            uff_res = AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            uff_res = -1
        if uff_res != 0:
            print(f"  [WARNING] Force field optimization did not converge for {molecule_name}")
    Chem.MolToXYZFile(mol, output_xyz)
    return True


def to_float(token):
    if token is None: return None
    return float(token.replace("D", "E").replace("d", "e"))

FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"

def parse_dipole(text):
    """
    Extract dipole from the 'molecular dipole:' section ONLY.
    Anchors to the section header to avoid matching quadrupole 'full:' lines.
    """
    pattern = r'molecular dipole:.*?full:\s+(' + FLOAT_RE + r')\s+(' + FLOAT_RE + r')\s+(' + FLOAT_RE + r')\s+(' + FLOAT_RE + ')'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return to_float(matches[-1][3])
    return None


def parse_energy(text):
    """
    Extract total energy.
    """
    pattern = r'TOTAL ENERGY\s+(' + FLOAT_RE + r')\s+Eh'
    matches = re.findall(pattern, text)
    if matches:
        return to_float(matches[-1])
    return None


def parse_alpha(text):
    """
    Extract molecular polarizability alpha(0).
    """
    patterns = [
        r'(?:Mol\.\s+)?(?:alpha|α)\s*(?:\(0\))?\s*/au\s*:?\s*(' + FLOAT_RE + ')',
        r'Mol\.\s+C6AA\s+/au.*?(?:alpha|α)\s*(?:\(0\))?\s*/au\s*:?\s*(' + FLOAT_RE + ')',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            return to_float(m.group(1))
    return None


def get_rdkit_volume(xyz_file):
    """Calculate vdW volume (Angstrom^3) using RDKit Monte Carlo on optimized xyz."""
    try:
        mol = Chem.rdmolfiles.MolFromXYZFile(xyz_file)
        if mol:
            return AllChem.ComputeMolVolume(mol)
    except Exception:
        pass
    return None


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

    # Step 2: Geometry optimization
    print(f"  Step 2: Running geometry optimization...")
    opt_cmd = ['xtb', f"{safe_name}.xyz", '--opt', 'tight', '--chrg', str(charge), '--gfn', '2', '--namespace', safe_name]
    opt_result = subprocess.run(opt_cmd, capture_output=True, text=True, cwd=mol_work, timeout=600)

    if opt_result.returncode != 0 or ('normal termination of xtb' not in opt_result.stdout.lower() and 'normal termination of xtb' not in opt_result.stderr.lower()):
        print(f"  [ERROR] xTB geometry optimization failed for {name}. See log.")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== OPT STDOUT ===\n{opt_result.stdout}\n=== OPT STDERR ===\n{opt_result.stderr}")
        return None

    opt_xyz = os.path.join(mol_work, 'xtbopt.xyz')
    if not os.path.exists(opt_xyz):
        alt_opt = os.path.join(mol_work, f'{safe_name}.xtbopt.xyz')
        if os.path.exists(alt_opt):
            opt_xyz = alt_opt
        else:
            print(f"  [ERROR] Optimized xyz file not found for {name}.")
            return None

    # Step 3: Single-point + polarizability on optimized geometry
    print(f"  Step 3: Computing polarizability on optimized geometry...")
    sp_cmd = ['xtb', os.path.basename(opt_xyz), '--sp', '--alpha', '--chrg', str(charge), '--gfn', '2']
    sp_result = subprocess.run(sp_cmd, capture_output=True, text=True, cwd=mol_work, timeout=600)

    # Save full log for debugging
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== MOLECULE: {name} ===\nSMILES: {smiles}\nCharge: {charge}\nCategory: {category}\n\n")
        f.write("=== OPT STDOUT ===\n" + opt_result.stdout + "\n=== OPT STDERR ===\n" + opt_result.stderr)
        f.write("\n=== SP+ALPHA STDOUT ===\n" + sp_result.stdout + "\n=== SP+ALPHA STDERR ===\n" + sp_result.stderr)
        
    if sp_result.returncode != 0 or ('normal termination of xtb' not in sp_result.stdout.lower() and 'normal termination of xtb' not in sp_result.stderr.lower()):
        print(f"  [ERROR] xTB single point / alpha failed for {name}. See log.")
        return None

    # ---- PARSE: Use SP output ONLY for dipole and energy ----
    # This avoids contamination from intermediate OPT step outputs
    sp_text = sp_result.stdout

    dipole = parse_dipole(sp_text)
    energy = parse_energy(sp_text)
    alpha  = parse_alpha(sp_text)

    # If SP output parsing failed, warn (both steps already confirmed successful)
    if dipole is None:
        print(f"  [WARNING] Could not parse dipole from SP output for {name}, trying OPT output")
        dipole = parse_dipole(opt_result.stdout)
    if energy is None:
        print(f"  [WARNING] Could not parse energy from SP output for {name}, trying OPT output")
        energy = parse_energy(opt_result.stdout)

    # Volume: RDKit on xTB-optimized geometry (xTB does not output vdW volume)
    volume_A3 = get_rdkit_volume(opt_xyz)

    results = {
        'Dipole_Debye': dipole,
        'Polarizability_au': alpha,
        'Volume_A3': round(volume_A3, 4) if volume_A3 is not None else None,
        'Total_Energy_Eh': energy,
    }

    print(f"  Results: μ={results['Dipole_Debye']} D, α={results['Polarizability_au']} au, V={results['Volume_A3']} Å³")
    return results


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
            res.update({'Molecule': name, 'SMILES': smi, 'Category': 'Refrigerant', 'Charge': 0,
                        'Source': 'GFN2-xTB', 'Has_Vega_Literature': name in VEGA_REFRIGERANTS})
            all_results.append(res)

    print("\n\n>>> PHASE 2: CATIONS (charge=+1) <<<")
    for name, smi in cations:
        res = run_xtb(name, smi, charge=1, category='Cation', work_dir=WORK_DIR, log_dir=LOG_DIR)
        if res:
            res.update({'Molecule': name, 'SMILES': smi, 'Category': 'Cation', 'Charge': 1,
                        'Source': 'GFN2-xTB', 'Has_Vega_Literature': False})
            all_results.append(res)

    print("\n\n>>> PHASE 3: ANIONS (charge=-1) <<<")
    for name, smi in anions:
        res = run_xtb(name, smi, charge=-1, category='Anion', work_dir=WORK_DIR, log_dir=LOG_DIR)
        if res:
            res.update({'Molecule': name, 'SMILES': smi, 'Category': 'Anion', 'Charge': -1,
                        'Source': 'GFN2-xTB', 'Has_Vega_Literature': False})
            all_results.append(res)

    out_df = pd.DataFrame(all_results)
    out_df = out_df[['Category', 'Molecule', 'SMILES', 'Charge',
                     'Dipole_Debye', 'Polarizability_au', 'Volume_A3',
                     'Total_Energy_Eh', 'Source', 'Has_Vega_Literature']]
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ All computations complete! Saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
