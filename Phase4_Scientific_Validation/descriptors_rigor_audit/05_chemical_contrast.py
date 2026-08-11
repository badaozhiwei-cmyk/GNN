import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem

ROOT_DIR = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT_DIR / "Phase4_Scientific_Validation"
AUDIT_OUT = Path(__file__).resolve().parent / "audit_outputs"
AUDIT_OUT.mkdir(parents=True, exist_ok=True)

DESC_CSV = WORK_DIR / "xTB_Physics_Descriptors.csv"

def stereo_audit(df):
    records = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        isomeric = Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None
        stereo = [] if mol is None else [str(b.GetStereo()) for b in mol.GetBonds()
                                         if b.GetBondType() == Chem.BondType.DOUBLE]
        records.append({'Molecule': row['Molecule'], 'input_smiles': row['SMILES'],
                        'canonical_isomeric_smiles': isomeric,
                        'double_bond_stereo': ';'.join(stereo) or 'none'})
    audit = pd.DataFrame(records)
    audit.to_csv(AUDIT_OUT / 'chemical_structure_audit.csv', index=False)
    ez = audit[audit.Molecule.isin(['R1336mzz(E)', 'R1336mzz(Z)'])]
    if len(ez) == 2 and (ez.double_bond_stereo == 'none').any():
        raise ValueError('R1336mzz E/Z label is unsupported: stereochemistry is absent from SMILES.')
    if len(ez) == 2 and ez.canonical_isomeric_smiles.nunique() != 2:
        raise ValueError('R1336mzz E/Z entries resolve to the same isomeric structure.')

def plot_chemical_contrast(df, group_name, molecules, ax_mu, ax_alpha, ax_vol, title_prefix=""):
    """Plot physical descriptors for a selected group of molecules."""
    group_df = df[df['Molecule'].isin(molecules)].copy()
    if group_df.empty:
        return
        
    # Sort dataframe in the exact order provided in 'molecules'
    group_df['Molecule'] = pd.Categorical(group_df['Molecule'], categories=molecules, ordered=True)
    group_df = group_df.sort_values('Molecule')
    
    x = group_df['Molecule'].astype(str)
    mu = group_df['Dipole_Debye']
    alpha = group_df['Polarizability_au']
    vol = group_df['Volume_A3']
    
    # Dipole
    ax_mu.bar(x, mu, color='#3498db', alpha=0.8, edgecolor='black')
    ax_mu.set_ylabel("Dipole (μ) / D")
    ax_mu.set_title(f"{title_prefix}\nDipole Moment", fontsize=10)
    for i, v in enumerate(mu):
        ax_mu.text(i, v + 0.1, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
        
    # Alpha
    ax_alpha.plot(x, alpha, marker='o', color='#e74c3c', linewidth=2, markersize=8)
    ax_alpha.set_ylabel("Polarizability (α) / au")
    ax_alpha.set_title("Polarizability", fontsize=10)
    for i, v in enumerate(alpha):
        ax_alpha.text(i, v + max(alpha)*0.02, f"{v:.1f}", ha='center', va='bottom', fontsize=8)
        
    # Volume
    ax_vol.plot(x, vol, marker='s', color='#2ecc71', linewidth=2, markersize=8)
    ax_vol.set_ylabel("Volume (V) / Å³")
    ax_vol.set_title("Volume", fontsize=10)
    for i, v in enumerate(vol):
        ax_vol.text(i, v + max(vol)*0.02, f"{v:.1f}", ha='center', va='bottom', fontsize=8)

def main():
    if not DESC_CSV.is_file():
        print("Descriptor CSV not found.")
        return
        
    df = pd.read_csv(DESC_CSV)
    df = df[df['Category'] == 'Refrigerant']
    stereo_audit(df)
    
    groups = {
        "Group A: Position Isomers": ["R134", "R134a"],
        "Group B: Stereo Isomers": ["R1336mzz(E)", "R1336mzz(Z)"],
        "Group C: Methane Fluorination": ["R41", "R32", "R23", "R14"],
        "Group D: Ethane Fluorination": ["R161", "R152a", "R143a", "R134a", "R125", "R116"],
        "Group E: Symmetric Perfluoro": ["R14", "R116", "R218"]
    }
    
    fig, axes = plt.subplots(len(groups), 3, figsize=(15, 3.5 * len(groups)))
    
    for i, (group_name, molecules) in enumerate(groups.items()):
        plot_chemical_contrast(df, group_name, molecules, axes[i, 0], axes[i, 1], axes[i, 2], title_prefix=group_name)
        
    # Verify symmetric molecules dipole
    print("\n--- Verifying Symmetric Perfluorocarbons ---")
    sym_df = df[df['Molecule'].isin(["R14", "R116", "R218"])]
    for _, row in sym_df.iterrows():
        print(f"{row['Molecule']}: Dipole = {row['Dipole_Debye']:.4f} D (Tolerance < 0.01 D: {row['Dipole_Debye'] < 0.01})")
    # Do not describe R218 as passing the symmetric-zero criterion when it does not.
    if not sym_df.empty:
        sym_df.assign(near_zero=sym_df['Dipole_Debye'].abs() < 0.01).to_csv(
            AUDIT_OUT / 'symmetric_dipole_tolerance.csv', index=False)
        
    plt.tight_layout()
    plt.savefig(AUDIT_OUT / "chemical_contrast_groups.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved {AUDIT_OUT / 'chemical_contrast_groups.png'}")

if __name__ == "__main__":
    main()
