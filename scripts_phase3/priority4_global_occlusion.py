import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'Interpretability_Case_Studies'))

from Explainer_Engine import Explainer_Engine
from smarts_dict import get_group_matches
from torch_geometric.data import Batch

def main():
    model_path = os.path.join(ROOT, 'GNN_for_property_prediction', 'checkpoints_v2', 'best_gat_seed_1.pth')
    data_path = os.path.join(ROOT, 'ZLJ_DATA.xlsx')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading Explainer Engine...")
    explainer = Explainer_Engine(model_path)
    model = explainer.model
    device = explainer.device
    model.eval()
    
    print(f"Loading Dataset from {data_path}...")
    df = pd.read_excel(data_path)
    print(f"Total records: {len(df)}")
    
    # We will accumulate the drop in prediction for each group
    group_stats = {} # group_name -> list of drops
    
    # Helper to perform occlusion
    def get_occluded_prediction(G_batch, cond_device, target_indices):
        h_input = explainer._get_embeddings(G_batch).detach()
        h_mod = h_input.clone()
        for idx in target_indices:
            h_mod[idx] = 0.0 # Knock out the embedding
            
        def pre_hook(module, args):
            return (h_mod, args[1])
        handle = model.l1.register_forward_pre_hook(pre_hook)
        
        with torch.no_grad():
            out = model(G_batch, cond_device).item()
            
        handle.remove()
        return out
        
    print("Running Global Occlusion Sweep...")
    # Wrap with tqdm for progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sweeping 4444 Records"):
        c_smi = row['IL Cation SMILES']
        a_smi = row['IL Anion SMILES']
        r_smi = row['Refrigerant SMILES']
        T = row['T (K)']
        P = row['P (bar)']
        
        G, num_bond = explainer._build_strict_graph(c_smi, a_smi, r_smi)
        if G is None: continue
            
        cond = explainer.compute_condition(c_smi, a_smi, r_smi, T, P)
        cond_device = cond.unsqueeze(0).to(device)
        G_batch = Batch.from_data_list([G]).to(device)
        
        # Baseline
        with torch.no_grad():
            y_base = model(G_batch, cond_device).item()
            
        # Graph structures
        try:
            m_c = Chem.MolFromSmiles(c_smi)
            m_a = Chem.MolFromSmiles(a_smi)
            m_r = Chem.MolFromSmiles(r_smi)
            n_c = m_c.GetNumAtoms()
            n_a = m_a.GetNumAtoms()
        except:
            continue
            
        # Match groups
        c_groups = get_group_matches(c_smi)
        a_groups = get_group_matches(a_smi)
        r_groups = get_group_matches(r_smi)
        
        # We will only occlude Cation and Anion groups for the solvent analysis
        # Map indices to the combined graph
        combined_groups = {}
        for g_name, indices in c_groups.items():
            combined_groups.setdefault(f"Cat_{g_name}", []).extend(indices)
            
        for g_name, indices in a_groups.items():
            shifted_indices = [i + n_c for i in indices]
            combined_groups.setdefault(f"Ani_{g_name}", []).extend(shifted_indices)
            
        for g_name, indices in combined_groups.items():
            if 'Other_Atoms' in g_name: continue # Skip miscellaneous atoms
            
            y_occluded = get_occluded_prediction(G_batch, cond_device, indices)
            drop = y_base - y_occluded
            
            if g_name not in group_stats:
                group_stats[g_name] = []
            group_stats[g_name].append(drop)
            
    # Calculate stats
    print("\n" + "="*80)
    print(" GLOBAL OCCLUSION RESULTS (THERMODYNAMIC DRIVING FORCES)")
    print("="*80)
    
    results = []
    for g_name, drops in group_stats.items():
        results.append({
            'Group': g_name,
            'Count': len(drops),
            'Mean_Drop': np.mean(drops),
            'Std_Drop': np.std(drops),
            'Max_Drop': np.max(drops),
        })
        
    res_df = pd.DataFrame(results).sort_values(by='Mean_Drop', ascending=False)
    
    print(f"{'Group':<25} | {'Count':<6} | {'Mean Drop':<10} | {'Std Drop':<10}")
    print("-" * 60)
    for _, r in res_df.iterrows():
        print(f"{r['Group']:<25} | {r['Count']:<6.0f} | {r['Mean_Drop']:<10.4f} | {r['Std_Drop']:<10.4f}")
        
    out_path = os.path.join(ROOT, 'scripts_phase3', 'global_group_importance.csv')
    res_df.to_csv(out_path, index=False)
    print(f"\n✅ Results saved to: {out_path}")

if __name__ == '__main__':
    main()
