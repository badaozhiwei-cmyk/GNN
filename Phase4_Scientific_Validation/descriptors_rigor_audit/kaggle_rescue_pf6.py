"""
Kaggle script for rescuing the calculation of [PF6]-.

[PF6]- is a perfectly symmetric octahedral (O_h) molecule.
Numerical integration grid noise in xTB often causes SCF oscillation for perfect symmetries
under `--opt tight`. This script attempts progressive fallback strategies:
1. Normal optimization
2. Loose optimization
"""

import os
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# --- KAGGLE ENVIRONMENT FIX ---
xtb_root = '/kaggle/working/xtb-dist'
if os.path.exists(xtb_root):
    os.environ['PATH'] = f"{xtb_root}/bin:" + os.environ.get('PATH', '')
    os.environ['XTBPATH'] = f"{xtb_root}/share/xtb"
else:
    print(f"[WARNING] Could not find xTB installation at {xtb_root}.")

def parse_xtb_output(log_text):
    mu, alpha, vol, energy = None, None, None, None
    for line in log_text.split('\n'):
        if "TOTAL ENERGY" in line:
            try:
                energy = float(line.split()[3])
            except: pass
        elif "molecular dipole:" in line:
            try:
                mu = float(line.split()[-1])
            except: pass
        elif "Mol. α(0) /au" in line:
            try:
                alpha = float(line.split()[-1])
            except: pass
        elif "vdW volume" in line:
            try:
                vol = float(line.split()[3])
            except: pass
    return energy, mu, alpha, vol

def rescue_molecule(smiles, name, charge):
    work_dir = f"/kaggle/working/rescue_{name}"
    os.makedirs(work_dir, exist_ok=True)
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    xyz_path = os.path.join(work_dir, f"{name}.xyz")
    Chem.rdmolfiles.MolToXYZFile(mol, xyz_path)
    
    strategies = [
        ("normal", ["xtb", f"{name}.xyz", "--opt", "normal", "--chrg", str(charge)]),
        ("loose", ["xtb", f"{name}.xyz", "--opt", "loose", "--chrg", str(charge)])
    ]
    
    success = False
    opt_log = ""
    
    print(f"\n--- Rescuing {name} (Charge: {charge}) ---")
    for strat_name, opt_cmd in strategies:
        print(f"Attempting {strat_name} optimization...")
        opt_res = subprocess.run(opt_cmd, cwd=work_dir, capture_output=True, text=True)
        opt_log = opt_res.stdout + "\n" + opt_res.stderr
        
        if "normal termination of xtb" in opt_log:
            print(f"✅ Success with {strat_name} optimization!")
            success = True
            break
        else:
            print(f"❌ Failed with {strat_name} optimization.")
            
    if not success:
        print(f"❌ All rescue strategies failed for {name}.")
        return None
        
    print("Running SP+ALPHA on optimized geometry...")
    sp_cmd = ["xtb", "xtbopt.xyz", "--alpha", "--chrg", str(charge)]
    sp_res = subprocess.run(sp_cmd, cwd=work_dir, capture_output=True, text=True)
    sp_log = sp_res.stdout + "\n" + sp_res.stderr
    
    if "normal termination of xtb" not in sp_log:
        print("❌ ALPHA calculation failed on optimized geometry.")
        return None
        
    energy, mu, alpha, vol = parse_xtb_output(opt_log + "\n" + sp_log)
    
    print("\n=== RESCUE SUCCESSFUL ===")
    print(f"Molecule : {name}")
    print(f"Energy   : {energy} Eh")
    print(f"Dipole   : {mu} D (Note: origin dependent for ions)")
    print(f"Alpha    : {alpha} au")
    print(f"Volume   : {vol} A^3")
    
    return {
        "Molecule": name,
        "SMILES": smiles,
        "Charge": charge,
        "Energy_Eh": energy,
        "Dipole_Debye": mu,
        "Polarizability_au": alpha,
        "Volume_A3": vol
    }

if __name__ == "__main__":
    # Rescue PF6-
    res = rescue_molecule(smiles="F[P-](F)(F)(F)(F)F", name="_PF6_", charge=-1)
    
    if res:
        df = pd.DataFrame([res])
        df.to_csv("/kaggle/working/rescued_PF6.csv", index=False)
        print("\nSaved rescued results to /kaggle/working/rescued_PF6.csv")
        print("You can manually copy these Alpha and Volume values into your main CSV!")
