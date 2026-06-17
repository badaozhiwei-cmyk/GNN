import os
import sys
import numpy as np
from rdkit import Chem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'Interpretability_Case_Studies'))

from Explainer_Engine import Explainer_Engine, ELEMENT_SYMBOL
from smarts_dict import get_group_matches

def main():
    model_path = os.path.join(ROOT, 'GNN_for_property_prediction', 'pretrained_model', 'GAT_300', 'best_model_para.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    explainer = Explainer_Engine(model_path)
    
    # BMIM-Tf2N + R1234yf
    c_smi = "CCCC[n+]1cccc(C)c1"
    a_smi = "FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F"
    r_smi = "C(=C(F)F)(C(F)(F)F)F" # R1234yf
    T = 298.15
    P = 1.0

    print("Running IG Attribution...")
    node_scores, atom_types, mol_type = explainer.get_attention_scores(c_smi, a_smi, r_smi, T, P)
    if node_scores is None:
        print("Error: Could not build graph")
        return
        
    c_matches = get_group_matches(c_smi)
    a_matches = get_group_matches(a_smi)
    r_matches = get_group_matches(r_smi)
    
    c_len = Chem.MolFromSmiles(c_smi).GetNumAtoms()
    a_len = Chem.MolFromSmiles(a_smi).GetNumAtoms()
    
    group_scores = {}
    atom_details = []
    
    for i in range(len(node_scores)):
        m_type = mol_type[i]
        score = float(node_scores[i])
        at_num = int(atom_types[i])
        sym = ELEMENT_SYMBOL.get(at_num, f"Z{at_num}")
        
        if m_type == 0:
            part = "Cat"
            local_idx = i
            matches = c_matches
        elif m_type == 1:
            part = "Ani"
            local_idx = i - c_len
            matches = a_matches
        else:
            part = "Ref"
            local_idx = i - c_len - a_len
            matches = r_matches
            
        my_group = "Unknown"
        for g_name, g_idx_list in matches.items():
            if local_idx in g_idx_list:
                my_group = f"{g_name}_{part}"
                break
                
        atom_details.append({
            'idx': i,
            'part': part,
            'sym': sym,
            'group': my_group,
            'score': score
        })
        
        if my_group not in group_scores:
            group_scores[my_group] = {'sum': 0.0, 'count': 0}
        group_scores[my_group]['sum'] += score
        group_scores[my_group]['count'] += 1

    print("\n" + "="*75)
    print(" 1. GROUP-LEVEL ABSOLUTE IG & DENSITY")
    print("="*75)
    print(f"{'Group Name':<30} | {'Atom Count':<10} | {'Absolute Sum':<15} | {'Density (Sum/Count)':<15}")
    print("-" * 75)
    
    sorted_groups = sorted(group_scores.items(), key=lambda x: x[1]['sum'], reverse=True)
    for g_name, data in sorted_groups:
        density = data['sum'] / data['count']
        print(f"{g_name:<30} | {data['count']:<10} | {data['sum']:<15.4f} | {density:<15.4f}")

    print("\n" + "="*75)
    print(" 2. ATOM-LEVEL ABSOLUTE IG (Top 15)")
    print("="*75)
    print(f"{'Rank':<5} | {'Part':<5} | {'Element':<8} | {'Functional Group':<30} | {'Absolute Score':<15}")
    print("-" * 75)
    
    sorted_atoms = sorted(atom_details, key=lambda x: x['score'], reverse=True)
    for rank, atom in enumerate(sorted_atoms[:15]):
        print(f"#{rank+1:<4} | {atom['part']:<5} | {atom['sym']:<8} | {atom['group']:<30} | {atom['score']:<15.4f}")

if __name__ == '__main__':
    main()
