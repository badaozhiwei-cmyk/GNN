"""
step8_cci_scatter.py
====================
Computes the Chemical Coverage Index (CCI) for each refrigerant and plots 
CCI vs Zero-shot R² as a publication-quality scatter plot.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Resolve ROOT directory using pathlib
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

def compute_cci_and_plot():
    print("[Info] Starting Chemical Coverage Index (CCI) analysis...")
    
    # 1. Read index_with_anion.csv from ROOT
    csv_path = ROOT / 'index_with_anion.csv'
    df = pd.read_csv(csv_path)
    
    # Get unique refrigerants and their SMILES
    ref_df = df[['refrigerant', 'refri_smiles']].drop_duplicates()
    
    # 2. Compute Morgan fingerprints (radius=2, nBits=2048)
    fps = {}
    for _, row in ref_df.iterrows():
        ref_name = row['refrigerant']
        smiles = row['refri_smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps[ref_name] = fp
        else:
            print(f"[Warning] Could not parse SMILES for {ref_name}: {smiles}")
            
    # 3. Compute CCI(R) = max Tanimoto similarity between R and all OTHER refrigerants
    cci_dict = {}
    nn_dict = {}
    for ref_name, fp in fps.items():
        max_sim = -1.0
        best_nn = None
        for other_name, other_fp in fps.items():
            if ref_name != other_name:
                sim = DataStructs.TanimotoSimilarity(fp, other_fp)
                if sim > max_sim:
                    max_sim = sim
                    best_nn = other_name
        cci_dict[ref_name] = max_sim
        nn_dict[ref_name] = best_nn
        
    # 4. Hardcoded GAT R² values and chemical family mappings
    gat_r2 = {
        'R32': 0.9286,
        'R152a': 0.8222,
        'R125': 0.7731,
        'R161': 0.7489,
        'R1234yf': 0.5824,
        'R23': 0.1894,
        'R134a': 0.0737,
        'R22': -0.1207
    }
    
    family_map = {
        'R32': 'HFC',
        'R152a': 'HFC',
        'R125': 'HFC',
        'R161': 'HFC',
        'R23': 'HFC',
        'R134a': 'HFC',
        'R1234yf': 'HFO',
        'R22': 'HCFC'
    }
    
    family_colors = {
        'HFC': '#1f77b4',   # Blue
        'HFO': '#2ca02c',   # Green
        'HCFC': '#ff7f0e'   # Orange
    }
    
    # 5. Build analysis table
    analysis_data = []
    for ref_name, r2_val in gat_r2.items():
        if ref_name in cci_dict:
            fam = family_map.get(ref_name, 'Unknown')
            cci_val = cci_dict[ref_name]
            nn_val = nn_dict[ref_name]
            analysis_data.append({
                'refrigerant': ref_name,
                'family': fam,
                'cci': cci_val,
                'zero_shot_r2': r2_val,
                'nearest_neighbor': nn_val
            })
            
    res_df = pd.DataFrame(analysis_data)
    
    # 6. Save to loro_cci_analysis.csv
    csv_out = ROOT / 'loro_cci_analysis.csv'
    res_df.to_csv(csv_out, index=False)
    print(f"[Done] CCI data saved to {csv_out}")
    print("\n" + res_df.to_string(index=False) + "\n")
    
    # 7. Create publication-quality scatter plot
    fig_dir = ROOT / 'figure_v5'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / 'cci_vs_r2.png'
    
    # Font setup
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    chosen_font = 'Arial' if 'Arial' in available_fonts else 'sans-serif'
    
    plt.rcParams['font.family'] = chosen_font
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.2
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Plot points by family for legend control
    families_present = res_df['family'].unique()
    for fam in ['HFC', 'HFO', 'HCFC']:
        fam_df = res_df[res_df['family'] == fam]
        if not fam_df.empty:
            ax.scatter(
                fam_df['cci'],
                fam_df['zero_shot_r2'],
                c=family_colors[fam],
                label=fam,
                s=120,
                alpha=0.9,
                edgecolors='black',
                linewidths=1.0,
                zorder=4
            )
            
    # Annotate refrigerant names
    # Custom text offset offsets to prevent text collision
    offsets = {
        'R32': (0.01, 0.025),
        'R152a': (0.012, 0.005),
        'R125': (0.01, 0.02),
        'R161': (-0.048, -0.01),
        'R1234yf': (0.01, 0.02),
        'R23': (0.01, 0.02),
        'R134a': (0.012, 0.015),
        'R22': (0.012, -0.03)
    }
    
    for _, row in res_df.iterrows():
        r_name = row['refrigerant']
        dx, dy = offsets.get(r_name, (0.01, 0.01))
        ax.annotate(
            r_name,
            (row['cci'], row['zero_shot_r2']),
            xytext=(row['cci'] + dx, row['zero_shot_r2'] + dy),
            fontsize=11,
            fontweight='bold',
            color='#222222',
            zorder=5
        )
        
    # Reference lines
    ax.axhline(0.0, color='red', linestyle='--', linewidth=1.5, label='R² = 0', zorder=2)
    ax.axhline(0.7, color='green', linestyle='--', linewidth=1.5, label='Reliable threshold (R² = 0.7)', zorder=2)
    
    # Axes limits and labels
    x_min = res_df['cci'].min() - 0.05
    x_max = res_df['cci'].max() + 0.08
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.25, 1.05)
    
    ax.set_xlabel('Chemical Coverage Index (CCI)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Zero-Shot R²', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Chemical Coverage Index vs Zero-Shot Generalization', fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=11, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Done] Scatter plot successfully saved to {fig_path}")

if __name__ == '__main__':
    compute_cci_and_plot()
