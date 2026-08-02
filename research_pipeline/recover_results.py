"""
recover_results.py
==================
Disaster recovery script. If Kaggle instance is cleared or reset, 
runs this script to reconstruct `loro_gnn_results.csv` and `loro_gnn_seed_details.csv`
from the printed logs of the first 7 completed refrigerants.
"""

import os
import pandas as pd
from pathlib import Path

# Resolve ROOT directory using pathlib
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# Hardcoded data from printed terminal logs for the 7 completed refrigerants
COMPLETED_DATA = {
    'R32': {
        'n_test': 672,
        'seeds': [
            {'seed': 0, 'r2': 0.0742, 'mae': 0.1552},
            {'seed': 1, 'r2': 0.7318, 'mae': 0.0785},
            {'seed': 2, 'r2': 0.8422, 'mae': 0.0569},
            {'seed': 3, 'r2': -0.4398, 'mae': 0.1921},
            {'seed': 4, 'r2': 0.9260, 'mae': 0.0336}
        ],
        'mean_r2': 0.4269,
        'std_r2': 0.5272,
        'mean_mae': 0.1033  # calculated mean mae from details
    },
    'R1234yf': {
        'n_test': 685,
        'seeds': [
            {'seed': 0, 'r2': 0.2680, 'mae': 0.0571},
            {'seed': 1, 'r2': 0.6958, 'mae': 0.0476},
            {'seed': 2, 'r2': -0.1029, 'mae': 0.1037},
            {'seed': 3, 'r2': 0.5396, 'mae': 0.0524},
            {'seed': 4, 'r2': 0.6318, 'mae': 0.0429}
        ],
        'mean_r2': 0.4064,
        'std_r2': 0.2936,
        'mean_mae': 0.0607
    },
    'R134a': {
        'n_test': 649,
        'seeds': [
            {'seed': 0, 'r2': -0.5368, 'mae': 0.1854},
            {'seed': 1, 'r2': -0.3870, 'mae': 0.1766},
            {'seed': 2, 'r2': 0.0464, 'mae': 0.1392},
            {'seed': 3, 'r2': -0.2187, 'mae': 0.1589},
            {'seed': 4, 'r2': -0.4644, 'mae': 0.1788}
        ],
        'mean_r2': -0.3121,
        'std_r2': 0.2081,
        'mean_mae': 0.1678
    },
    'R22': {
        'n_test': 184,
        'seeds': [
            {'seed': 0, 'r2': 0.1447, 'mae': 0.1905},
            {'seed': 1, 'r2': 0.9520, 'mae': 0.0389},
            {'seed': 2, 'r2': 0.7444, 'mae': 0.0933},
            {'seed': 3, 'r2': 0.9485, 'mae': 0.0326},
            {'seed': 4, 'r2': 0.9414, 'mae': 0.0383}
        ],
        'mean_r2': 0.7462,
        'std_r2': 0.3108,
        'mean_mae': 0.0787
    },
    'R152a': {
        'n_test': 217,
        'seeds': [
            {'seed': 0, 'r2': 0.7549, 'mae': 0.0616},
            {'seed': 1, 'r2': 0.6911, 'mae': 0.0693},
            {'seed': 2, 'r2': 0.8911, 'mae': 0.0387},
            {'seed': 3, 'r2': 0.6232, 'mae': 0.0764},
            {'seed': 4, 'r2': 0.8165, 'mae': 0.0468}
        ],
        'mean_r2': 0.7554,
        'std_r2': 0.0936,
        'mean_mae': 0.0586
    },
    'R125': {
        'n_test': 260,
        'seeds': [
            {'seed': 0, 'r2': 0.6731, 'mae': 0.0706},
            {'seed': 1, 'r2': 0.5209, 'mae': 0.0758},
            {'seed': 2, 'r2': 0.4173, 'mae': 0.0909},
            {'seed': 3, 'r2': 0.4501, 'mae': 0.0872},
            {'seed': 4, 'r2': 0.7438, 'mae': 0.0526}
        ],
        'mean_r2': 0.5610,
        'std_r2': 0.1269,
        'mean_mae': 0.0754
    },
    'R161': {
        'n_test': 155,
        'seeds': [
            {'seed': 0, 'r2': 0.7025, 'mae': 0.0575},
            {'seed': 1, 'r2': 0.7677, 'mae': 0.0489},
            {'seed': 2, 'r2': 0.5694, 'mae': 0.0685},
            {'seed': 3, 'r2': 0.4713, 'mae': 0.0848},
            {'seed': 4, 'r2': 0.7147, 'mae': 0.0573}
        ],
        'mean_r2': 0.6451,
        'std_r2': 0.1087,
        'mean_mae': 0.0634
    }
}

def recover():
    print("====================================================")
    print("  Disaster Recovery: Reconstructing GNN CSV files")
    print("====================================================")

    res_path = ROOT / 'loro_gnn_results.csv'
    seed_path = ROOT / 'loro_gnn_seed_details.csv'

    # Load existing if available to avoid overwriting newly run data (e.g. R23)
    if res_path.exists():
        res_df = pd.read_csv(res_path)
    else:
        res_df = pd.DataFrame(columns=['model', 'refrigerant', 'r2_mean', 'r2_std', 'mae_mean', 'n_seeds', 'n_test'])

    if seed_path.exists():
        seed_df = pd.read_csv(seed_path)
    else:
        seed_df = pd.DataFrame(columns=['model', 'refrigerant', 'seed', 'r2', 'mae'])

    # Reconstruct the 7 completed refrigerants
    new_res_rows = []
    new_seed_rows = []

    for ref, data in COMPLETED_DATA.items():
        # Remove old rows if they exist to prevent duplicates
        if not res_df.empty:
            res_df = res_df[~((res_df['refrigerant'] == ref) & (res_df['model'] == 'GAT_v5'))]
        if not seed_df.empty:
            seed_df = seed_df[~((seed_df['refrigerant'] == ref) & (seed_df['model'] == 'GAT_v5'))]

        # Append summary row
        new_res_rows.append({
            'model': 'GAT_v5',
            'refrigerant': ref,
            'r2_mean': data['mean_r2'],
            'r2_std': data['std_r2'],
            'mae_mean': data['mean_mae'],
            'n_seeds': 5,
            'n_test': data['n_test']
        })

        # Append seed-level rows
        for s_info in data['seeds']:
            new_seed_rows.append({
                'model': 'GAT_v5',
                'refrigerant': ref,
                'seed': s_info['seed'],
                'r2': s_info['r2'],
                'mae': s_info['mae']
            })

    # Combine and save
    added_res = pd.DataFrame(new_res_rows)
    added_seed = pd.DataFrame(new_seed_rows)

    res_df = pd.concat([res_df, added_res], ignore_index=True)
    seed_df = pd.concat([seed_df, added_seed], ignore_index=True)

    res_df.to_csv(res_path, index=False)
    seed_df.to_csv(seed_path, index=False)

    print(f"[Done] Reconstructed: {len(COMPLETED_DATA)} refrigerants successfully written back.")
    print(f"       loro_gnn_results.csv -> {res_path}")
    print(f"       loro_gnn_seed_details.csv -> {seed_path}")
    print("====================================================\n")

if __name__ == '__main__':
    recover()
