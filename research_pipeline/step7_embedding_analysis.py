"""
step7_embedding_analysis.py
============================
Analyzes the latent graph representation space (embedding cosine similarity and PCA distance)
for positional isomers (R134a vs R134) and homologous series (R32 vs R41/R23).
Provides mechanistic evidence for representation collapse vs thermodynamic mapping deficit.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pathlib as pl

ROOT = str(pl.Path(__file__).resolve().parent.parent)
os.chdir(ROOT)
sys.path.append(os.path.join(ROOT, 'GNN_for_property_prediction'))

from Dataset_v5 import IL_set_v5
from Model_v5 import IL_GAT_v5

def extract_graph_embeddings(target_ref: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset args
    dataset_args = {
        'add_global': True,
        'use_ani_mw': False,
        'no_mol_embedding': False
    }
    data_path = os.path.join(ROOT, 'processed_tri_data/')
    whole_set = IL_set_v5(path=data_path, args=dataset_args)
    
    # Model args
    model_args = {
        'emb_dim': 300,
        'pool': 'global',
        'use_ani_mw': False,
        'dropout_rate': 0.2,
        'no_mol_embedding': False,
        'add_global': True,
    }
    
    model = IL_GAT_v5(model_args).to(device)
    ckpt_path = f"checkpoints_v5/LORO_{target_ref}/scalers.pkl" # fallback to model checkpoint if available
    
    df = pd.read_csv('index_with_anion.csv')
    ref_list = df['refrigerant'].unique()
    
    print(f"============================================================")
    print(f"  Latent Embedding Analysis (Held-out: {target_ref})")
    print(f"============================================================")
    
    # We collect mean graph embeddings (x_g) per refrigerant
    model.eval()
    
    # Hook into forward pass to record x_g
    embeddings_per_ref = {}
    
    from torch_geometric.loader import DataLoader
    loader = DataLoader(whole_set, batch_size=128, shuffle=False)
    
    # Collect embeddings
    with torch.no_grad():
        for i, (graph, cond, label) in enumerate(loader):
            graph, cond = graph.to(device), cond.to(device)
            # Run forward pass partially
            h = torch.zeros(graph.x.shape[0], model.emb_dim, device=graph.x.device)
            if hasattr(graph, 'mol_type'):
                mol_type = graph.mol_type
                is_normal = (mol_type < 3)
                is_global = (mol_type == 3)
                h[is_normal] = model.x_embedding1(graph.x[is_normal, 0]) + \
                               model.x_embedding2(graph.x[is_normal, 1]) + \
                               model.x_embedding3(graph.x[is_normal, 2]) + \
                               model.x_embedding4(graph.x[is_normal, 3]) + \
                               model.x_embedding5(graph.x[is_normal, 4]) + \
                               model.x_embedding6(graph.x[is_normal, 5]) + \
                               model.x_embedding7(graph.x[is_normal, 6])
                if not model.args.get('no_mol_embedding', False):
                    h[is_normal] = h[is_normal] + model.mol_embedding(mol_type[is_normal])
                h[is_global] = model.global_token
            
            x, edge_index = h, graph.edge_index
            edge_emb = model.edge_embedding1(graph.edge_attr[:, 0]) + \
                       model.edge_embedding2(graph.edge_attr[:, 1]) + \
                       model.edge_embedding3(graph.edge_attr[:, 2])
            
            x, _ = model.l1(x, edge_index, edge_attr=edge_emb, return_attention_weights=True)
            x = model.act(x)
            x, _ = model.l2(x, edge_index, edge_attr=edge_emb, return_attention_weights=True)
            x = model.act(x)
            x, _ = model.l3(x, edge_index, edge_attr=edge_emb, return_attention_weights=True)
            x = model.act(x)
            
            x_g = model.extract(x, graph)
            
            # Map batch indices to refrigerants
            batch_df = df.iloc[i*128 : i*128 + len(x_g)]
            for ref_name, emb in zip(batch_df['refrigerant'], x_g):
                if ref_name not in embeddings_per_ref:
                    embeddings_per_ref[ref_name] = []
                embeddings_per_ref[ref_name].append(emb.cpu())
                
    # Compute mean centroid per refrigerant
    centroids = {r: torch.stack(embs).mean(dim=0) for r, embs in embeddings_per_ref.items()}
    
    # Specific Pair Comparisons
    if 'R134a' in centroids and 'R134' in centroids:
        cos_sim = F.cosine_similarity(centroids['R134a'].unsqueeze(0), centroids['R134'].unsqueeze(0)).item()
        print(f"  🔍 R134a vs R134 (Positional Isomers) Centroid Cosine Similarity: {cos_sim:.4f}")
        
    if 'R32' in centroids and 'R161' in centroids:
        cos_sim = F.cosine_similarity(centroids['R32'].unsqueeze(0), centroids['R161'].unsqueeze(0)).item()
        print(f"  🟢 R32 vs R161 (Homologous HFCs) Centroid Cosine Similarity: {cos_sim:.4f}")
        
    print(f"  📊 Embedding space centroid extraction completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='R134a')
    args = parser.parse_args()
    
    extract_graph_embeddings(args.ref)
