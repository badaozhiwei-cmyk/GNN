import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

ROOT_DIR = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT_DIR / "Phase4_Scientific_Validation"
AUDIT_OUT = Path(__file__).resolve().parent / "audit_outputs"
AUDIT_OUT.mkdir(parents=True, exist_ok=True)

DESC_CSV = WORK_DIR / "xTB_Physics_Descriptors.csv"
XTB_WORK = WORK_DIR / "xtb_work"

def safe_filename(name):
    import re
    return re.sub(r'[^\w\-.]', '_', name)

def get_rdkit_volume_with_topology(smiles, mol_name):
    """
    Builds the RDKit Mol from SMILES to get correct bond orders/aromaticity,
    then copies the 3D coordinates from the xTB optimized XYZ file,
    and computes the vdW volume.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    
    safe_name = safe_filename(mol_name)
    xyz_file = XTB_WORK / safe_name / "xtbopt.xyz"
    
    if not xyz_file.is_file():
        # Fallback to alternative naming if exists
        alt_xyz = XTB_WORK / safe_name / f"{safe_name}.xtbopt.xyz"
        if alt_xyz.is_file():
            xyz_file = alt_xyz
        else:
            return None
            
    try:
        lines = xyz_file.read_text(encoding="utf-8").strip().split('\n')
        n_atoms = int(lines[0].strip())
        if n_atoms != mol.GetNumAtoms() or len(lines) < n_atoms + 2:
            return None
            
        conf = Chem.Conformer(n_atoms)
        from rdkit.Geometry import Point3D
        for i in range(n_atoms):
            parts = lines[i+2].split()
            if len(parts) < 4 or parts[0] != mol.GetAtomWithIdx(i).GetSymbol():
                raise ValueError("XYZ atom order/element does not match topology")
            # xTB format: Element x y z
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            conf.SetAtomPosition(i, Point3D(x, y, z))
            
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        
        # Calculate volume (fixing gridSpacing=0.2 for numerical stability)
        vol = AllChem.ComputeMolVolume(mol, confId=0, gridSpacing=0.2,
                                       boxMargin=2.0)
        return vol
    except Exception as e:
        print(f"Error computing volume for {mol_name}: {e}")
        return None


def main():
    if not DESC_CSV.is_file():
        print("Descriptor CSV not found.")
        return
        
    df = pd.read_csv(DESC_CSV)
    
    print("Computing 2D sizes and topologically-correct 3D volumes...")
    volumes_3d = []
    mw_list = []
    hac_list = []
    
    for _, row in df.iterrows():
        smi = row['SMILES']
        name = row['Molecule']
        
        # 2D Descriptors
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mw = Descriptors.MolWt(mol)
            hac = mol.GetNumHeavyAtoms()
        else:
            mw = np.nan
            hac = np.nan
            
        # 3D Volume
        vol_3d = get_rdkit_volume_with_topology(smi, name)
        
        volumes_3d.append(vol_3d)
        mw_list.append(mw)
        hac_list.append(hac)
        
    df['RDKit_Volume_A3'] = volumes_3d
    df['MolWt'] = mw_list
    df['HeavyAtomCount'] = hac_list
    
    # Analyze by Category
    categories = df['Category'].unique()
    
    # Create subplots for correlation heatmaps
    fig, axes = plt.subplots(1, len(categories), figsize=(6*len(categories), 5))
    if len(categories) == 1:
        axes = [axes]
        
    cols_to_corr = ['Volume_A3', 'RDKit_Volume_A3', 'Polarizability_au', 'MolWt', 'HeavyAtomCount']
    
    print("\n--- Correlation Results (Pearson r) ---")
    for i, cat in enumerate(categories):
        cat_df = df[df['Category'] == cat].copy()
        
        # Drop missing
        cat_df = cat_df.dropna(subset=cols_to_corr)
        if cat_df.empty:
            continue
            
        corr = cat_df[cols_to_corr].corr()
        
        # Print specific key correlations
        alpha_v_xtb = corr.loc['Polarizability_au', 'Volume_A3']
        mw_v_xtb = corr.loc['MolWt', 'Volume_A3']
        rdkit_v_xtb = corr.loc['RDKit_Volume_A3', 'Volume_A3']
        
        print(f"[{cat}]")
        print(f"  α vs xTB Volume       : {alpha_v_xtb:.4f}")
        print(f"  MolWt vs xTB Volume   : {mw_v_xtb:.4f}")
        print(f"  RDKit Vol vs xTB Vol  : {rdkit_v_xtb:.4f}")
        
        # Plot heatmap
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[i], square=True)
        axes[i].set_title(f"{cat} (N={len(cat_df)})")
        
    plt.tight_layout()
    plt.savefig(AUDIT_OUT / "volume_redundancy_analysis.png", dpi=300)
    print(f"\nSaved correlation heatmap to {AUDIT_OUT / 'volume_redundancy_analysis.png'}")
    
    # Save the expanded dataframe
    df.to_csv(AUDIT_OUT / "descriptors_with_sizes.csv", index=False)

if __name__ == "__main__":
    main()
