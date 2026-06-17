import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import random
from rdkit import Chem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'Interpretability_Case_Studies'))

from Explainer_Engine import Explainer_Engine, ELEMENT_SYMBOL

def lookup_smiles_inline(name, smiles_dict):
    name = str(name).strip().upper()
    name_no_bracket = name.replace('[', '').replace(']', '')
    if name in smiles_dict: return smiles_dict[name]
    if name_no_bracket in smiles_dict: return smiles_dict[name_no_bracket]
    name_no_hyphen = name_no_bracket.replace('-', '')
    if name_no_hyphen in smiles_dict: return smiles_dict[name_no_hyphen]
    return None

def main():
    model_path = os.path.join(ROOT, 'GNN_for_property_prediction', 'pretrained_model', 'GAT_300', 'best_model_para.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    explainer = Explainer_Engine(model_path)
    
    # 1. Load Dictionary
    smiles_csv_path = os.path.join(ROOT, 'Original_Data', 'smiles.csv')
    if not os.path.exists(smiles_csv_path):
        smiles_csv_path = os.path.join(ROOT, 'smiles.csv')
        
    if not os.path.exists(smiles_csv_path):
         print(f"Error: {smiles_csv_path} not found")
         return
         
    il_df = pd.read_csv(smiles_csv_path)
    il_df.columns = [c.strip() for c in il_df.columns]
    smiles_dict = {}
    for idx, row in il_df.iterrows():
        abbr = str(row['Abbreviation']).strip().upper()
        abbr_no_bracket = abbr.replace('[', '').replace(']', '')
        smi = str(row['Smiles']).strip()
        smiles_dict[abbr] = smi
        smiles_dict[abbr_no_bracket] = smi
    
    extra_smiles = {
        'Tf2N': 'FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F',
        'R1234YF': 'C(=C(F)F)(C(F)(F)F)F',
        'R32': 'C(F)F',
        'R134A': 'C(C(F)(F)F)F',
        'BF4': '[B-](F)(F)(F)F',
        'PF6': '[P-](F)(F)(F)(F)(F)F',
        'DCA': 'N#C[N-]C#N',
        'SCN': 'N#C[S-]'
    }
    for k, v in extra_smiles.items():
        smiles_dict[k.upper()] = v
        smiles_dict[k.upper().replace('[','').replace(']','')] = v

    # 2. Load Dataset
    excel_path = os.path.join(ROOT, 'ZLJ_DATA.xlsx')
    dfs = []
    for sheet in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']:
        try:
            tmp_df = pd.read_excel(excel_path, sheet_name=sheet, skiprows=2)
            dfs.append(tmp_df)
        except: pass
    df = pd.concat(dfs, ignore_index=True).dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])
    
    # Target Anions to test generalization
    target_anions = ['Tf2N', 'BF4', 'PF6', 'DCA', 'SCN']
    
    print("\n" + "="*75)
    print(" PRIORITY 2: ANCHOR GENERALIZATION TEST")
    print("="*75)
    
    results = []
    
    for target in target_anions:
        # Find rows with this anion (fuzzy match)
        anion_rows = df[df['IL anion'].str.contains(target, case=False, na=False)]
        if len(anion_rows) == 0:
            continue
            
        # Sample up to N inferences
        N = min(50, len(anion_rows))
        sampled = anion_rows.sample(n=N, random_state=42)
        
        top1_atoms = []
        
        print(f"\nEvaluating Anion: {target} (N={N})")
        for idx, row in sampled.iterrows():
            c_smi = lookup_smiles_inline(row['IL cation'], smiles_dict)
            a_smi = lookup_smiles_inline(row['IL anion'], smiles_dict)
            r_smi = lookup_smiles_inline(row['Refrigerant'], smiles_dict)
            if None in (c_smi, a_smi, r_smi):
                continue
                
            T = float(row['T (K)'])
            P = float(row['P (MPa)'])
            
            node_scores, atom_types, mol_type = explainer.get_attention_scores(c_smi, a_smi, r_smi, T, P)
            if node_scores is None: continue
            
            c_len = Chem.MolFromSmiles(c_smi).GetNumAtoms()
            
            anion_scores = []
            for i in range(len(node_scores)):
                if mol_type[i] == 1:
                    at_num = int(atom_types[i])
                    sym = ELEMENT_SYMBOL.get(at_num, f"Z{at_num}")
                    anion_scores.append((sym, float(node_scores[i])))
            
            if anion_scores:
                anion_scores.sort(key=lambda x: x[1], reverse=True)
                top1_atom_sym = anion_scores[0][0]
                top1_atoms.append(top1_atom_sym)
                
        if not top1_atoms:
            continue
            
        freq = pd.Series(top1_atoms).value_counts()
        best_anchor = freq.index[0]
        stability = freq.iloc[0] / len(top1_atoms)
        
        results.append({
            'Anion': target,
            'Evaluated': len(top1_atoms),
            'Top_Anchor': best_anchor,
            'Stability_Score': stability,
            'Top_3_Frequencies': ", ".join([f"{k}({v})" for k, v in freq.head(3).items()])
        })
        
    print("\n" + "="*75)
    print(" ANCHOR STABILITY SCORE TABLE")
    print("="*75)
    print(f"{'Anion':<15} | {'N':<5} | {'Top Anchor':<12} | {'Stability Score':<15} | {'Details (Top 3)'}")
    print("-" * 75)
    for res in results:
        print(f"{res['Anion']:<15} | {res['Evaluated']:<5} | {res['Top_Anchor']:<12} | {res['Stability_Score']:<15.2f} | {res['Top_3_Frequencies']}")

if __name__ == '__main__':
    main()
