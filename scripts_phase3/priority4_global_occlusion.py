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

# ----------------- Helper to Load SMILES Dictionary -----------------
def load_smiles_dict():
    smiles_csv_path = os.path.join(ROOT, 'Original_Data', 'smiles.csv')
    if not os.path.exists(smiles_csv_path):
        smiles_csv_path = os.path.join(ROOT, 'smiles.csv')
    
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
        'R32':        'C(F)F',                           
        'R134A':      'C(C(F)(F)F)F',                    
        'R143A':      'CC(F)(F)F',                       
        'R125':       'C(F)(F)(C(F)(F)F)',               
        'R114':       'C(C(F)(F)Cl)(F)(F)Cl',           
        'R1234YF':    'C(=C(F)F)(C(F)(F)F)F',           
        'R1234ZE(E)': 'F/C=C/C(F)(F)F',                 
        'R152A':      'CC(F)F',                          
        'R23':        'C(F)(F)F',                        
        'R41':        'CF',                              
        'AC':         'CC(=O)[O-]',                      
        'Tf2N':       'FC(S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F)(F)F', 
        'R22':        'ClC(F)F',                         
        'R22B1':      'BrC(F)F',                         
        'R14':        'FC(F)(F)F',                       
        'R116':       'FC(F)(F)C(F)(F)F',               
        'R124':       'FC(F)(F)C(Cl)F',                 
        'R124A':      'ClC(F)C(F)(F)F',                 
        'R114A':      'ClC(Cl)(F)C(F)(F)F',             
        'R134':       'FC(F)C(F)F',                      
        'R161':       'CCF',                             
        'R218':       'FC(F)(F)C(F)(F)C(F)(F)F',        
        'R227EA':     'FC(F)(F)C(F)C(F)(F)F',           
        'R236FA':     'FC(F)(F)CC(F)(F)F',              
        'R245FA':     'FC(F)(F)CC(F)F',                 
        'R1233ZD(E)': 'FC(F)(F)/C=C/Cl',               
        'R1336MZZ(E)':'FC(F)(F)/C=C/C(F)(F)F',         
        'R1336MZZ(Z)':'FC(F)(F)/C=C\\C(F)(F)F', 
        'P4442':      'CCCC[P+](CCCC)(CCCC)CC',                  
        'P66614':     'CCCCCC[P+](CCCCCC)(CCCCCC)CCCCCCCCCCCCCC',
        'DOIM':       'CCCCCCCCn1cc[n+](CCCCCCCC)c1',            
        'P44414':     'CCCC[P+](CCCC)(CCCC)CCCCCCCCCCCCCC',      
        'EMPY':       'CC[n+]1cccc(C)c1',                        
        'BMPY':       'CCCC[n+]1cccc(C)c1',                      
        'DMPIM':      'CCCn1cc[n+](C)c1C',                       
        'P4441':      'CCCC[P+](CCCC)(CCCC)C',                   
        'C8H4F13C1IM':'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CCn1cc[n+](C)c1', 
        'ET2PO4':     'CCOP(=O)([O-])OCC',                               
        'BEI':        'FC(F)(F)C(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)C(F)(F)F', 
        'TTES':       'FC(F)(F)OC(F)C(F)(F)S(=O)(=O)[O-]',               
        'HFPS':       'FC(F)(F)C(F)C(F)(F)S(=O)(=O)[O-]',                
        'PFBS':       'FC(F)(F)C(F)(F)C(F)(F)S(=O)(=O)[O-]',      
        'TMPP':       'CC(C)(C)CC(C)CCP(=O)([O-])CC(C)CC(C)(C)C',        
        'FS':         'FC(F)(F)C(F)OC(F)(F)C(F)(F)S(=O)(=O)[O-]',        
        'FEP':        'F[P-](F)(F)(C(F)(F)C(F)(F)F)(C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F', 
        'PR':         'CCC(=O)[O-]',                                     
        'OTF':        'FC(F)(F)S(=O)(=O)[O-]',                           
        'TPES':       'FC(F)(F)C(F)(F)OC(F)C(F)(F)S(=O)(=O)[O-]',        
        'I':          '[I-]',                                            
        'TFES':       'FC(F)C(F)(F)S(=O)(=O)[O-]',                       
        'PFP':        'FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)[O-]',          
        'PE':         'CCCCC(=O)[O-]',                                   
        'TMEM':       'FC(F)(F)S(=O)(=O)[C-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F', 
    }
    for k, v in extra_smiles.items():
        smiles_dict[k.upper()] = v
        smiles_dict[k.upper().replace('[','').replace(']','')] = v
        
    return smiles_dict

def lookup_smiles(name, smiles_dict):
    name = str(name).strip().upper()
    name_no_bracket = name.replace('[', '').replace(']', '')
    if name in smiles_dict: return smiles_dict[name]
    if name_no_bracket in smiles_dict: return smiles_dict[name_no_bracket]
    name_no_hyphen = name_no_bracket.replace('-', '')
    if name_no_hyphen in smiles_dict: return smiles_dict[name_no_hyphen]
    return None

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
    
    print(f"Loading SMILES Dictionary...")
    smiles_dict = load_smiles_dict()
    
    print(f"Loading Dataset from {data_path}...")
    dfs = []
    for sheet in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']:
        try:
            tmp_df = pd.read_excel(data_path, sheet_name=sheet, skiprows=2)
            dfs.append(tmp_df)
        except Exception as e:
            pass
    df_vle = pd.concat(dfs, ignore_index=True)
    df_vle = df_vle.dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])
    print(f"Total valid records: {len(df_vle)}")
    
    group_stats = {}
    
    def get_occluded_prediction(G_batch, cond_device, target_indices):
        h_input = explainer._get_embeddings(G_batch).detach()
        h_mod = h_input.clone()
        for idx in target_indices:
            h_mod[idx] = 0.0
            
        def pre_hook(module, args):
            return (h_mod, args[1])
        handle = model.l1.register_forward_pre_hook(pre_hook)
        
        with torch.no_grad():
            out = model(G_batch, cond_device).item()
            
        handle.remove()
        return out
        
    print("Running Global Occlusion Sweep...")
    for idx, row in tqdm(df_vle.iterrows(), total=len(df_vle), desc="Sweeping 4444 Records"):
        c_smi = lookup_smiles(row['IL cation'], smiles_dict)
        a_smi = lookup_smiles(row['IL anion'], smiles_dict)
        r_smi = lookup_smiles(row['Refrigerant'], smiles_dict)
        
        if None in (c_smi, a_smi, r_smi): continue
        
        T = float(row['T (K)'])
        P = float(row['P (MPa)'])
        
        G, num_bond = explainer._build_strict_graph(c_smi, a_smi, r_smi)
        if G is None: continue
            
        cond = explainer.compute_condition(c_smi, a_smi, r_smi, T, P)
        cond_device = cond.unsqueeze(0).to(device)
        G_batch = Batch.from_data_list([G]).to(device)
        
        with torch.no_grad():
            y_base = model(G_batch, cond_device).item()
            
        try:
            m_c = Chem.MolFromSmiles(c_smi)
            m_a = Chem.MolFromSmiles(a_smi)
            n_c = m_c.GetNumAtoms()
        except:
            continue
            
        c_groups = get_group_matches(c_smi)
        a_groups = get_group_matches(a_smi)
        
        combined_groups = {}
        for g_name, indices in c_groups.items():
            combined_groups.setdefault(f"Cat_{g_name}", []).extend(indices)
        for g_name, indices in a_groups.items():
            shifted_indices = [i + n_c for i in indices]
            combined_groups.setdefault(f"Ani_{g_name}", []).extend(shifted_indices)
            
        for g_name, indices in combined_groups.items():
            if 'Other_Atoms' in g_name: continue
            
            y_occluded = get_occluded_prediction(G_batch, cond_device, indices)
            drop = y_base - y_occluded
            
            if g_name not in group_stats:
                group_stats[g_name] = []
            group_stats[g_name].append(drop)
            
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
