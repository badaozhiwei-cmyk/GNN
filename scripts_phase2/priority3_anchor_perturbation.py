import os
import sys
import torch
import pandas as pd
from rdkit import Chem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'Interpretability_Case_Studies'))

from Explainer_Engine import Explainer_Engine, ELEMENT_SYMBOL
from torch_geometric.data import Batch

def main():
    model_path = os.path.join(ROOT, 'GNN_for_property_prediction', 'checkpoints_v2', 'best_gat_seed_1.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    explainer = Explainer_Engine(model_path)
    model = explainer.model
    device = explainer.device
    
    # Define test cases
    test_cases = [
        {
            'name': 'BMIM-Tf2N + R1234yf',
            'c': 'CCCC[n+]1cccc(C)c1',
            'a': 'FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F',
            'r': 'C(=C(F)F)(C(F)(F)F)F',
            'anchor_symbol': 'N',
            'peripheral_symbol': 'F'
        },
        {
            'name': 'BMIM-BF4 + R1234yf',
            'c': 'CCCC[n+]1cccc(C)c1',
            'a': '[B-](F)(F)(F)F',
            'r': 'C(=C(F)F)(C(F)(F)F)F',
            'anchor_symbol': 'B',
            'peripheral_symbol': 'F'
        },
        {
            'name': 'BMIM-PF6 + R1234yf',
            'c': 'CCCC[n+]1cccc(C)c1',
            'a': '[P-](F)(F)(F)(F)(F)F',
            'r': 'C(=C(F)F)(C(F)(F)F)F',
            'anchor_symbol': 'P',
            'peripheral_symbol': 'F'
        }
    ]
    
    T = 298.15
    P = 1.0
    
    print("\n" + "="*80)
    print(" PHASE 2: ANCHOR PERTURBATION (KNOCKOUT) EXPERIMENT")
    print("="*80)
    
    for case in test_cases:
        print(f"\n🧪 Evaluating Case: {case['name']}")
        
        G, num_bond = explainer._build_strict_graph(case['c'], case['a'], case['r'])
        if G is None: continue
            
        cond = explainer.compute_condition(case['c'], case['a'], case['r'], T, P)
        cond_device = cond.unsqueeze(0).to(device)
        G_batch = Batch.from_data_list([G]).to(device)
        
        # 1. Baseline prediction
        model.eval()
        with torch.no_grad():
            y_base = model(G_batch, cond_device).item()
            
        print(f"  [Baseline] Original Prediction (x1): {y_base:.4f}")
        
        # Helper to perform occlusion
        def get_occluded_prediction(target_indices):
            h_input = explainer._get_embeddings(G_batch).detach()
            h_mod = h_input.clone()
            for idx in target_indices:
                h_mod[idx] = 0.0 # Knock out the embedding
                
            # Hook
            h_injected = None
            def pre_hook(module, args):
                return (h_injected, args[1])
            handle = model.l1.register_forward_pre_hook(pre_hook)
            
            with torch.no_grad():
                out = model(G_batch, cond_device).item()
                
            handle.remove()
            return out
            
        # Find indices
        num_real_atom = G.x.shape[0] - 1
        atom_types = G.x[:num_real_atom, 0].cpu().numpy()
        mol_type = G.mol_type[:num_real_atom].cpu().numpy()
        
        anchor_indices = []
        peripheral_indices = []
        
        for i in range(num_real_atom):
            if mol_type[i] == 1: # Anion
                at_num = int(atom_types[i])
                sym = ELEMENT_SYMBOL.get(at_num, f"Z{at_num}")
                if sym == case['anchor_symbol']:
                    anchor_indices.append(i)
                elif sym == case['peripheral_symbol']:
                    peripheral_indices.append(i)
                    
        # 2. Occlude Peripheral
        if peripheral_indices:
            y_peri = get_occluded_prediction(peripheral_indices)
            drop_peri = y_base - y_peri
            print(f"  [Perturb Periphery] Knocked out {len(peripheral_indices)}x {case['peripheral_symbol']} atoms.")
            print(f"                      New Pred: {y_peri:.4f} (Drop: {drop_peri:+.4f})")
        else:
            print(f"  [Perturb Periphery] No {case['peripheral_symbol']} atoms found in anion.")
            
        # 3. Occlude Anchor
        if anchor_indices:
            y_anchor = get_occluded_prediction(anchor_indices)
            drop_anchor = y_base - y_anchor
            print(f"  [Perturb Anchor]    Knocked out {len(anchor_indices)}x {case['anchor_symbol']} atom(s).")
            print(f"                      New Pred: {y_anchor:.4f} (Drop: {drop_anchor:+.4f})")
            
            # Comparison
            if peripheral_indices:
                impact_ratio = abs(drop_anchor) / (abs(drop_peri) + 1e-6)
                print(f"  --> Knocking out {case['anchor_symbol']} is {impact_ratio:.1f}x more impactful than knocking out ALL {case['peripheral_symbol']} atoms!")
        else:
            print(f"  [Perturb Anchor] No {case['anchor_symbol']} atom found in anion.")
            
if __name__ == '__main__':
    main()
