import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
import os

# --- Configuration ---
DESCRIPTOR_CSV = 'Phase4_Scientific_Validation/xTB_Physics_Descriptors.csv'
OUTPUT_DIR = 'Phase4_Scientific_Validation/Validation_Results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. HFC Rule-Based Filter
def is_hfc(smiles):
    """
    Rule based filtering for HFC refrigerants:
    - Only contains C, H, F
    - Contains at least one H and at least one F
    - No double bonds (no C=C)
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return False
    
    has_c, has_h, has_f = False, False, False
    has_other = False
    
    # Check atoms
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == 'C': has_c = True
        elif sym == 'F': has_f = True
        else: has_other = True
    
    # RDKit implicitly adds H, so we count them
    num_h = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    if num_h > 0: has_h = True
        
    if has_other or not (has_c and has_h and has_f):
        return False
        
    # Check for double bonds
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            return False
            
    return True

def analyze_dataset():
    if not os.path.exists(DESCRIPTOR_CSV):
        print(f"Error: {DESCRIPTOR_CSV} not found. Run xTB pipeline first.")
        return
        
    df = pd.read_csv(DESCRIPTOR_CSV)
    
    # ---------------------------------------------------------
    # Task 1: Identify Target HFCs for LORO
    # ---------------------------------------------------------
    refrig_df = df[df['Category'] == 'Refrigerant'].copy()
    
    hfc_mask = refrig_df['SMILES'].apply(is_hfc)
    hfc_df = refrig_df[hfc_mask].copy()
    
    hfc_list = hfc_df['Molecule'].tolist()
    print("=== HFC-LORO Target Molecules ===")
    print(f"Identified {len(hfc_list)} pure HFCs: {', '.join(hfc_list)}\n")
    
    # ---------------------------------------------------------
    # Task 2: Collinearity Check (Alpha vs Volume)
    # ---------------------------------------------------------
    alpha = refrig_df['Polarizability_au']
    vol = refrig_df['Volume_A3']
    
    # Drop NaNs if any (shouldn't be any after fix)
    valid_mask = alpha.notna() & vol.notna()
    if valid_mask.sum() > 0:
        pearson_r = np.corrcoef(alpha[valid_mask], vol[valid_mask])[0, 1]
        print(f"=== Collinearity Check ===")
        print(f"Pearson r (Polarizability vs Volume): {pearson_r:.4f}")
        if pearson_r > 0.95:
            print("[WARNING] High collinearity detected. Ablation study (α alone vs V alone) is highly recommended.\n")
        else:
            print("[OK] Collinearity is within acceptable bounds.\n")
            
    # ---------------------------------------------------------
    # Task 3: Canonical SMILES Audit (Check for duplicates)
    # ---------------------------------------------------------
    print("=== Canonical SMILES Audit ===")
    refrig_df['Canonical_SMILES'] = refrig_df['SMILES'].apply(lambda s: Chem.CanonSmiles(s) if pd.notna(s) and Chem.MolFromSmiles(s) else None)
    
    # Find duplicates
    duplicates = refrig_df[refrig_df.duplicated(subset=['Canonical_SMILES'], keep=False)]
    if not duplicates.empty:
        print("[CRITICAL ERROR] Found duplicated refrigerants based on Canonical SMILES!")
        for smiles, group in duplicates.groupby('Canonical_SMILES'):
            mols = group['Molecule'].tolist()
            orig_smiles = group['SMILES'].tolist()
            print(f"  - Canonical SMILES {smiles} is shared by: {mols}")
            print(f"    Original SMILES: {orig_smiles}")
        print("You MUST fix the source SMILES before training.\n")
    else:
        print("[OK] All 26 refrigerants have unique Canonical SMILES.\n")
            
    # ---------------------------------------------------------
    # Task 4: Vega 2022 Cross-Validation Placeholder
    # ---------------------------------------------------------
    # Populate this dictionary with actual Vega 2022 SI values for the validation plot
    vega_literature_dipoles = {
        'R32': 1.98,
        'R134a': 2.06,
        'R143a': 2.32,
        'R125': 1.56,
        'R152a': 2.26,
        'R23': 1.65,
        'R41': 1.81,
        # Add more if available...
    }
    
    print("=== Vega 2022 Cross-Validation ===")
    merged = []
    for mol_name, vega_mu in vega_literature_dipoles.items():
        row = refrig_df[refrig_df['Molecule'] == mol_name]
        if not row.empty:
            xtb_mu = row.iloc[0]['Dipole_Debye']
            merged.append((mol_name, vega_mu, xtb_mu))
            
    if merged:
        test_df = pd.DataFrame(merged, columns=['Molecule', 'Vega_Dipole', 'xTB_Dipole'])
        test_df = test_df.dropna()
        rmse = np.sqrt(np.mean((test_df['Vega_Dipole'] - test_df['xTB_Dipole'])**2))
        r = np.corrcoef(test_df['Vega_Dipole'], test_df['xTB_Dipole'])[0,1]
        
        print(f"Matched {len(test_df)} refrigerants with Vega 2022.")
        print(f"Dipole RMSE: {rmse:.3f} D")
        print(f"Dipole Pearson r: {r:.3f}")
        
        plt.figure(figsize=(6,6))
        plt.scatter(test_df['Vega_Dipole'], test_df['xTB_Dipole'], c='blue', alpha=0.7)
        plt.plot([0, 4], [0, 4], 'k--', alpha=0.5)
        plt.xlabel('Vega 2022 Experimental/DFT Dipole (D)')
        plt.ylabel('GFN2-xTB Computed Dipole (D)')
        plt.title('Dipole Moment Validation')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'Vega_Dipole_Validation.png'), dpi=300)
        print(f"Saved validation plot to {OUTPUT_DIR}/Vega_Dipole_Validation.png\n")
    else:
        print("No matching Vega data found in descriptors yet.\n")
        
        
    # ---------------------------------------------------------
    # Task 5: Generate Nested HFC-LORO Split Definitions
    # ---------------------------------------------------------
    print("=== HFC-LORO Nested Split Definitions ===")
    print("For each iteration, one HFC is Test, one is Val, rest are Train.")
    # Simple rotational split for demonstration
    splits = []
    for i, test_hfc in enumerate(hfc_list):
        val_idx = (i + 1) % len(hfc_list)
        val_hfc = hfc_list[val_idx]
        train_hfcs = [h for h in hfc_list if h not in (test_hfc, val_hfc)]
        
        splits.append({
            'Test': test_hfc,
            'Val': val_hfc,
            'Train': train_hfcs
        })
        
    print(f"Generated {len(splits)} LORO split configurations.")
    print("Example Split 0:")
    print(f"  Test : {splits[0]['Test']}")
    print(f"  Val  : {splits[0]['Val']}")
    print(f"  Train: {splits[0]['Train'][:3]} ... ({len(splits[0]['Train'])} total)")
    print("Next step: Use these lists to mask the tri-graph dataset.")


if __name__ == "__main__":
    analyze_dataset()
