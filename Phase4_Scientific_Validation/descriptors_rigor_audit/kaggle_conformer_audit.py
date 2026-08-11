"""
Kaggle script for rigorous multi-conformer stability analysis of xTB descriptors.
Target: 7 HFC refrigerants with rotatable bonds/flexible geometries.

Process:
1. Generate 20 conformers using ETKDGv3 (seed 0-19).
2. MMFF94 optimization.
3. GFN2-xTB --opt tight
4. GFN2-xTB --alpha
5. Collect E, μ, α, V
6. Calculate Boltzmann weighted averages at 298.15K using stable softmax.
"""
import os
import subprocess
import shutil
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# --- KAGGLE ENVIRONMENT FIX ---
xtb_root = '/kaggle/working/xtb-dist'
if os.path.exists(xtb_root):
    os.environ['PATH'] = f"{xtb_root}/bin:" + os.environ.get('PATH', '')
    os.environ['XTBPATH'] = f"{xtb_root}/share/xtb"
else:
    print(f"[WARNING] Could not find xTB installation at {xtb_root}.")

TARGETS = {
    "R134": "FC(F)C(F)F",
    "R134a": "FC(F)(F)CF",
    "R152a": "CC(F)F",
    "R143a": "CC(F)(F)F",
    "R245fa": "FC(F)(F)CC(F)F",
    "R236fa": "FC(F)(F)CC(F)(F)F",
    "R227ea": "FC(F)(F)C(F)C(F)(F)F"
}

N_CONFORMERS = 20
KT = 0.00094448  # k_B * T in Hartrees at 298.15 K

def softmax_boltzmann(energies, kT):
    """Numerically stable softmax for Boltzmann weights."""
    e_np = np.asarray(energies, dtype=float)
    if e_np.size == 0 or not np.all(np.isfinite(e_np)):
        raise ValueError("No finite conformer energies")
    # Subtract min for numerical stability
    e_shifted = e_np - np.min(e_np)
    weights = np.exp(-e_shifted / kT)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid Boltzmann normalization")
    return weights / total

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

def run_xtb_for_conformer(mol, conf_id, work_dir):
    """Runs xTB opt and alpha for a specific conformer."""
    base_name = f"conf_{conf_id}"
    xyz_path = os.path.join(work_dir, f"{base_name}.xyz")
    Chem.rdmolfiles.MolToXYZFile(mol, xyz_path, confId=conf_id)
    
    # 1. OPT
    opt_cmd = ["xtb", f"{base_name}.xyz", "--opt", "tight"]
    opt_res = subprocess.run(opt_cmd, cwd=work_dir, capture_output=True, text=True, timeout=600)
    if opt_res.returncode != 0 or "normal termination of xtb" not in (opt_res.stdout + opt_res.stderr).lower():
        print(f"    [OPT FAILED] {opt_cmd}")
        print(f"    STDOUT: {opt_res.stdout.strip()[-200:]}")
        print(f"    STDERR: {opt_res.stderr.strip()}")
        return None
        
    # 2. ALPHA (SP)
    sp_cmd = ["xtb", "xtbopt.xyz", "--alpha"]
    sp_res = subprocess.run(sp_cmd, cwd=work_dir, capture_output=True, text=True, timeout=600)
    if sp_res.returncode != 0 or "normal termination of xtb" not in (sp_res.stdout + sp_res.stderr).lower():
        print(f"    [SP FAILED] {sp_cmd}")
        print(f"    STDOUT: {sp_res.stdout.strip()[-200:]}")
        print(f"    STDERR: {sp_res.stderr.strip()}")
        return None
        
    return parse_xtb_output(opt_res.stdout + "\n" + sp_res.stdout)

def main():
    base_dir = "/kaggle/working/conformer_audit"
    os.makedirs(base_dir, exist_ok=True)
    
    results = []
    
    for name, smiles in TARGETS.items():
        print(f"\n--- Processing {name} ---")
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # Embed 20 conformers
        cids = []
        for seed in range(N_CONFORMERS):
            ps = AllChem.ETKDGv3(); ps.randomSeed = seed; ps.pruneRmsThresh = -1
            cid = AllChem.EmbedMolecule(mol, ps)
            if cid >= 0: cids.append(cid)
        
        if len(cids) == 0:
            print(f"Failed to embed {name}")
            continue
            
        # MMFF94 Optimize
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
        
        mol_dir = os.path.join(base_dir, name)
        os.makedirs(mol_dir, exist_ok=True)
        
        conf_data = []
        failures = []
        
        for cid in cids:
            conf_dir = os.path.join(mol_dir, f"conf_{cid}")
            os.makedirs(conf_dir, exist_ok=True)
            
            xtb_res = run_xtb_for_conformer(mol, cid, conf_dir)
            if xtb_res and None not in xtb_res:
                e, mu, alpha, vol = xtb_res
                conf_data.append({
                    "Conformer": cid,
                    "Energy_Eh": e,
                    "Dipole_D": mu,
                    "Alpha_au": alpha,
                    "Volume_A3": vol
                })
            else:
                failures.append({'Molecule': name, 'Conformer': cid, 'status': 'xtb_or_parse_failed'})
        
        pd.DataFrame(failures).to_csv(os.path.join(base_dir, f"{name}_failures.csv"), index=False)
        if not conf_data:
            print(f"All xTB calculations failed for {name}")
            continue
            
        df_conf = pd.DataFrame(conf_data)
        df_conf.to_csv(os.path.join(base_dir, f"{name}_conformers.csv"), index=False)
        
        # Calculate Boltzmann statistics
        energies = df_conf["Energy_Eh"].values
        weights = softmax_boltzmann(energies, KT)
        
        mu_vals = df_conf["Dipole_D"].values
        alpha_vals = df_conf["Alpha_au"].values
        vol_vals = df_conf["Volume_A3"].values
        
        mu_boltz = np.sum(weights * mu_vals)
        alpha_boltz = np.sum(weights * alpha_vals)
        vol_boltz = np.sum(weights * vol_vals)
        
        min_idx = np.argmin(energies)
        
        results.append({
            "Molecule": name,
            "Success_Confs": len(df_conf),
            "MinE_Dipole": mu_vals[min_idx],
            "Boltz_Dipole": mu_boltz,
            "Dipole_Std": np.std(mu_vals),
            "Dipole_CV": np.std(mu_vals) / (np.mean(mu_vals) + 1e-9),
            "Dipole_Delta_Min_Boltz": abs(mu_vals[min_idx] - mu_boltz),
            "MinE_Alpha": alpha_vals[min_idx],
            "Boltz_Alpha": alpha_boltz,
            "MinE_Vol": vol_vals[min_idx],
            "Boltz_Vol": vol_boltz
        })
        
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(os.path.join(base_dir, "conformer_summary.csv"), index=False)
        print("\n=== Conformer Audit Complete ===")
        print(final_df[["Molecule", "Success_Confs", "MinE_Dipole", "Boltz_Dipole", "Dipole_Std", "Dipole_CV"]])

if __name__ == "__main__":
    main()
