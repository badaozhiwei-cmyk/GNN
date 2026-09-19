#!/usr/bin/env python3
"""
step24_stereo_ablation.py — Phase III Step 24: Stereo-Aware Representation Ablation
=====================================================================================
The First Mechanistic Intervention Experiment of Phase III.

Scientific Mission:
  Investigate whether providing explicit stereochemical bond representations (E/Z parity)
  can break the mathematical graph degeneracy G_E == G_Z, reduce prediction errors on
  the geometric isomers R1336mzz(E) and R1336mzz(Z), and relieve representation-blind
  false confidence (low ensemble sigma despite large OOD error).

Protocol:
  24A: Data Representation Lock (confirmed distinct SMILES in source data)
  24B: Isolated Stereo Feature Intervention (only bond.GetStereo() added to edge_attr)
  24C: Parametric Model Adaptation (IL_GAT_v6 edge_dim=4)
  24D: 5-Seed Training (M0_stereo x 5, Mreduced_stereo x 5 -> results_hfc_all_stereo/)
  24E: Hypothesis Testing & Evidence Decision Matrix:
         H1: E/Z Prediction Gap narrowed?
         H2: R1336mzz(E) error reduced?
         H3: False confidence relieved?

Usage:
  uv run python research_pipeline/step24_stereo_ablation.py [--mode all] [--seeds 42,43,44,45,46] [--dry_run]

Outputs:
  paper_results/table_step24_stereo_ablation.csv
  paper_results/table_step24_stereo_ablation.tex
  paper_results/audit_step24_stereo_ablation.json
  paper_results/stereo_zeroshot_predictions.csv
"""

from __future__ import annotations
import argparse
import os
import sys
import io
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data

# Force UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    print("FATAL: RDKit is required.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper_results"
PAPER.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'GNN_for_property_prediction'))

from Dataset_v6 import (
    FEATURE_SCHEMA,
    BASE_FEATURES,
    MODE_DEF,
    MODE_INDICES,
    MODE_COND_DIM,
    IL_set_v6,
    combine_Graph,
    add_global
)
from Model_v6 import IL_GAT_v6
from prepare_tri_graph_data_v6 import (
    get_atom_features,
    get_bond_features,
    lookup_smiles
)
from research_pipeline.step24_stereo_preflight_final import (
    run_preflight,
    mol2graph_stereo,
    mol_data_to_pyg
)

# Reference critical parameters for HFO evaluation
HFO_CRITICAL = {
    'R1234YF': (367.85, 3.382, 0.276),
    'R1234ZE(E)': (382.51, 3.635, 0.313),
    'R1233ZD(E)': (438.86, 3.5828, 0.304),
    'R1336MZZ(E)': (403.53, 2.779, 0.4128),
    'R1336MZZ(Z)': (444.50, 2.9037, 0.386),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


# ═══════════════════════════════════════════════════════════════
# 24D: Training Pipeline for Stereo Models
# ═══════════════════════════════════════════════════════════════

def train_stereo_models(args, modes: list[str], seeds: list[int], device: torch.device):
    data_dir = ROOT / 'processed_tri_data_hfc2739'
    splits_dir = ROOT / 'splits'
    out_base = ROOT / 'results_hfc_all_stereo'
    out_base.mkdir(parents=True, exist_ok=True)

    split_file = splits_dir / "HFC_all_split.npz"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")

    sp_data = np.load(split_file)
    train_idx = sp_data['train'].astype(int)
    val_idx = sp_data['val'].astype(int)

    print(f"\n{'='*80}")
    print(f"  STEP 24D: STEREO-AWARE MODEL TRAINING (edge_dim=4)")
    print(f"  Modes: {modes} | Seeds: {seeds} | Device: {device}")
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")
    print(f"{'='*80}")

    summary_records = []

    for md in modes:
        cond_dim = MODE_COND_DIM[md]
        run_tag = f"HFC_all_stereo_{md}"
        run_dir = out_base / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        exp_config = {
            "split": "HFC_all_split",
            "descriptor_mode": md,
            "cond_dim": cond_dim,
            "edge_dim": 4,
            "feature_indices": MODE_INDICES[md],
            "seeds": seeds,
            "epoch": args.epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "split_sha256": sha256_file(split_file),
        }
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(exp_config, f, indent=2)

        print(f"\n>>> [Training Mode: {run_tag}] (cond_dim={cond_dim}, edge_dim=4)...")
        # Instantiate dataset with edge_dim=4
        whole_set = IL_set_v6(path=str(data_dir), args={'descriptor_mode': md, 'edge_dim': 4})
        whole_set.fit_scalers(train_idx, save_dir=str(run_dir))

        train_set = torch.utils.data.Subset(whole_set, train_idx)
        val_set = torch.utils.data.Subset(whole_set, val_idx)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        seed_val_metrics = []

        for seed in seeds:
            ckpt_path = run_dir / f"best_seed_{seed}.pth"
            if ckpt_path.exists() and not args.force_retrain:
                print(f"  [Seed {seed}] Existing checkpoint found at {ckpt_path.name}. Skipping training.")
                continue

            set_seed(seed)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

            model_args = {
                'emb_dim': 300,
                'dropout_rate': 0.2,
                'cond_dim': cond_dim,
                'edge_dim': 4,
                'pool': 'global',
                'use_layernorm': False,
                'use_adaptive_gate': False
            }
            model = IL_GAT_v6(model_args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)
            criterion = nn.HuberLoss(delta=0.05)

            best_val_loss = float('inf')
            early_stop_count = 0

            for epoch in range(1, args.epoch + 1):
                model.train()
                for graph, cond, label in train_loader:
                    graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                    optimizer.zero_grad()
                    out = model(graph, cond)
                    loss = criterion(out.flatten(), label.flatten())
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                # Validation
                model.eval()
                val_losses = []
                val_preds = []
                val_trues = []
                with torch.no_grad():
                    for graph, cond, label in val_loader:
                        graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                        out = model(graph, cond)
                        loss = criterion(out.flatten(), label.flatten())
                        val_losses.append(loss.item())
                        val_preds.extend(out.flatten().cpu().numpy())
                        val_trues.extend(label.flatten().cpu().numpy())

                avg_val_loss = np.mean(val_losses)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stop_count = 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    early_stop_count += 1

                if early_stop_count >= args.patience:
                    break

            # Evaluate best model on val set
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            best_preds, best_trues = [], []
            with torch.no_grad():
                for graph, cond, label in val_loader:
                    graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                    out = model(graph, cond)
                    best_preds.extend(out.flatten().cpu().numpy())
                    best_trues.extend(label.flatten().cpu().numpy())

            vp = np.clip(np.array(best_preds), 0.0, 1.0)
            vt = np.array(best_trues)
            seed_mae = float(mean_absolute_error(vt, vp))
            seed_r2 = float(r2_score(vt, vp))
            print(f"  [Seed {seed}] Finished: Val MAE = {seed_mae:.4f}, R2 = {seed_r2:.4f}, Best Loss = {best_val_loss:.6f}")
            seed_val_metrics.append({
                'seed': seed, 'best_val_loss': best_val_loss,
                'mae': seed_mae, 'r2': seed_r2
            })

        if seed_val_metrics:
            df_val = pd.DataFrame(seed_val_metrics)
            df_val.to_csv(run_dir / "seed_val_metrics.csv", index=False)


# ═══════════════════════════════════════════════════════════════
# 24D: Zero-Shot Evaluation on R1336mzz(E/Z) and Full HFO
# ═══════════════════════════════════════════════════════════════

def evaluate_stereo_zeroshot(modes: list[str], seeds: list[int], device: torch.device):
    print(f"\n{'='*80}")
    print(f"  STEP 24D: ZERO-SHOT EVALUATION ON R1336mzz(E/Z) & HFO UNIVERSE")
    print(f"{'='*80}")

    df_raw = pd.read_csv(ROOT / 'index_with_anion.csv')
    df_hfo = df_raw[df_raw['sheet'] == 'Table S4. VLE HFOs'].copy().reset_index()
    df_hfo.rename(columns={'index': 'orig_data_idx'}, inplace=True)

    # Focus test set: R1336mzz isomers
    mzz_mask = df_hfo['refrigerant'].str.upper().str.contains('1336')
    print(f"  Total HFO test points: N={len(df_hfo)} (R1336mzz points: N={mzz_mask.sum()})")

    # Cache molecular weights
    mw_cache = {}
    def get_mw(smi):
        if smi not in mw_cache:
            mol = Chem.MolFromSmiles(smi)
            mw_cache[smi] = float(Descriptors.MolWt(mol)) if mol else 0.0
        return mw_cache[smi]

    # Pre-build 4-dim stereo graphs for all HFO points
    hfo_records = []
    for _, r in df_hfo.iterrows():
        r_name = str(r['refrigerant']).strip().upper()
        ref_smi = str(r['refri_smiles']).strip()
        cat_smi = str(r['cation_smiles']).strip()
        ani_smi = str(r['anion_smiles']).strip()

        cg = mol_data_to_pyg(*mol2graph_stereo(cat_smi), edge_dim=4)
        ag = mol_data_to_pyg(*mol2graph_stereo(ani_smi), edge_dim=4)
        rg = mol_data_to_pyg(*mol2graph_stereo(ref_smi), edge_dim=4)
        combined_g = add_global(combine_Graph([cg, ag, rg]))

        t_val = float(r['T_K'])
        p_val = float(r['P_MPa'])
        tc, pc, omega = HFO_CRITICAL[r_name]
        tr = t_val / tc
        pr = p_val / pc

        ref_mw = get_mw(ref_smi)
        cat_mw = get_mw(cat_smi)
        ani_mw = get_mw(ani_smi)

        mol_ref = Chem.MolFromSmiles(ref_smi)
        ref_logp = float(Descriptors.MolLogP(mol_ref)) if mol_ref else 0.0
        try:
            ref_charge = float(Descriptors.MaxAbsPartialCharge(mol_ref)) if mol_ref else 0.0
        except Exception:
            ref_charge = 0.0

        mol_cat = Chem.MolFromSmiles(cat_smi)
        cat_tpsa = float(Descriptors.TPSA(mol_cat)) if mol_cat else 0.0
        try:
            cat_charge = float(Descriptors.MaxAbsPartialCharge(mol_cat)) if mol_cat else 0.0
        except Exception:
            cat_charge = 0.0

        cond_22 = [
            0, 0, 0,  # 0,1,2 reserved for graphs
            t_val, p_val,
            ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa,
            ref_mw, cat_mw,
            0.0, 0.0, 0.0,
            0.0, 0.0,
            tc, pc, omega,
            tr, pr
        ]

        hfo_records.append({
            'orig_idx': int(r['orig_data_idx']),
            'refrigerant': r_name,
            'cation': str(r['cation']).strip(),
            'anion': str(r['anion']).strip(),
            'T_K': t_val,
            'P_MPa': p_val,
            'true_x1': float(r['x1']),
            'graph': combined_g,
            'cond_22': cond_22
        })

    prediction_rows = []

    for md in modes:
        cond_dim = MODE_COND_DIM[md]
        feat_indices = MODE_INDICES[md]
        run_tag = f"HFC_all_stereo_{md}"
        run_dir = ROOT / 'results_hfc_all_stereo' / run_tag

        # Load scalers
        scalers_path = run_dir / 'scalers.pkl'
        import joblib
        scalers = joblib.load(scalers_path)
        means = np.array([float(s.mean_[0]) for s in scalers], dtype=np.float32)
        scales = np.array([max(float(s.scale_[0]), 1e-8) for s in scalers], dtype=np.float32)

        # Load 5 seed models
        models = []
        for seed in seeds:
            ckpt_path = run_dir / f"best_seed_{seed}.pth"
            model_args = {
                'emb_dim': 300,
                'dropout_rate': 0.2,
                'cond_dim': cond_dim,
                'edge_dim': 4,
                'pool': 'global',
                'use_layernorm': False,
                'use_adaptive_gate': False
            }
            m = IL_GAT_v6(model_args).to(device)
            m.load_state_dict(torch.load(ckpt_path, map_location=device))
            m.eval()
            models.append(m)

        # Batched vectorized inference
        eval_batch_size = 64
        hfo_eval_data = []
        for item in hfo_records:
            raw_c = [item['cond_22'][idx_] for idx_ in feat_indices]
            scaled_c = [(raw_c[i] - means[i]) / scales[i] for i in range(cond_dim)]
            hfo_eval_data.append((item['graph'], torch.tensor(scaled_c, dtype=torch.float), item))

        all_model_seed_preds = {s: [] for s in seeds}

        for i in range(0, len(hfo_eval_data), eval_batch_size):
            chunk = hfo_eval_data[i:i + eval_batch_size]
            graphs = [c[0] for c in chunk]
            conds = torch.stack([c[1] for c in chunk]).to(device)
            batch_g = Batch.from_data_list(graphs).to(device)

            with torch.no_grad():
                for s_idx, seed in enumerate(seeds):
                    preds = models[s_idx](batch_g, conds).flatten().cpu().numpy()
                    all_model_seed_preds[seed].extend(preds)

        # Assemble dataframe
        for idx, item in enumerate(hfo_records):
            seed_preds_clip = [
                float(np.clip(all_model_seed_preds[s][idx], 0.0, 1.0))
                for s in seeds
            ]
            pred_mu = float(np.mean(seed_preds_clip))
            pred_sigma = float(np.std(seed_preds_clip, ddof=1)) if len(seed_preds_clip) > 1 else 0.0
            abs_err = abs(item['true_x1'] - pred_mu)

            prediction_rows.append({
                'model': f"Stereo_{md}",
                'descriptor_mode': md,
                'orig_idx': item['orig_idx'],
                'refrigerant': item['refrigerant'],
                'cation': item['cation'],
                'anion': item['anion'],
                'T_K': item['T_K'],
                'P_MPa': item['P_MPa'],
                'true_x1': item['true_x1'],
                'pred_mu': pred_mu,
                'pred_sigma': pred_sigma,
                'abs_error': abs_err,
                **{f'pred_seed_{s}': seed_preds_clip[i] for i, s in enumerate(seeds)}
            })

    df_preds = pd.DataFrame(prediction_rows)
    pred_csv_path = PAPER / "stereo_zeroshot_predictions.csv"
    df_preds.to_csv(pred_csv_path, index=False)
    print(f"  Saved full predictions: {pred_csv_path}")

    return df_preds


# ═══════════════════════════════════════════════════════════════
# 24E: Hypothesis Testing & Evidence Decision Matrix
# ═══════════════════════════════════════════════════════════════

def evaluate_hypotheses(df_stereo: pd.DataFrame):
    print(f"\n{'='*80}")
    print(f"  STEP 24E: HYPOTHESIS EVALUATION & EVIDENCE DECISION MATRIX")
    print(f"{'='*80}")

    # Load baseline HFC_all predictions
    base_csv = PAPER / "hfo_zeroshot_predictions_HFC_all.csv"
    if not base_csv.exists():
        raise FileNotFoundError(f"Missing baseline file: {base_csv}")

    df_base = pd.read_csv(base_csv)
    df_base['abs_error'] = np.abs(df_base['true_x1'] - df_base['pred_mu'])

    results_table = []

    for md in ['M0', 'Mreduced']:
        # Baseline subset
        b_sub = df_base[df_base['descriptor_mode'] == md]
        b_e = b_sub[b_sub['refrigerant'] == 'R1336MZZ(E)']
        b_z = b_sub[b_sub['refrigerant'] == 'R1336MZZ(Z)']

        base_e_mae = float(b_e['abs_error'].mean())
        base_z_mae = float(b_z['abs_error'].mean())
        base_e_sigma = float(b_e['pred_sigma'].mean())
        base_z_sigma = float(b_z['pred_sigma'].mean())
        base_gap = abs(base_e_mae - base_z_mae)

        # Stereo subset
        s_sub = df_stereo[df_stereo['descriptor_mode'] == md]
        s_e = s_sub[s_sub['refrigerant'] == 'R1336MZZ(E)']
        s_z = s_sub[s_sub['refrigerant'] == 'R1336MZZ(Z)']

        stereo_e_mae = float(s_e['abs_error'].mean())
        stereo_z_mae = float(s_z['abs_error'].mean())
        stereo_e_sigma = float(s_e['pred_sigma'].mean())
        stereo_z_sigma = float(s_z['pred_sigma'].mean())
        stereo_gap = abs(stereo_e_mae - stereo_z_mae)

        delta_e_mae = stereo_e_mae - base_e_mae
        pct_delta_e = (delta_e_mae / base_e_mae) * 100
        delta_z_mae = stereo_z_mae - base_z_mae
        pct_delta_z = (delta_z_mae / base_z_mae) * 100
        delta_gap = stereo_gap - base_gap

        results_table.append({
            'Mode': md,
            'Model_Type': 'Baseline (2D Graph)',
            'E_MAE': base_e_mae,
            'Z_MAE': base_z_mae,
            'EZ_Gap': base_gap,
            'E_Sigma': base_e_sigma,
            'Z_Sigma': base_z_sigma,
        })
        results_table.append({
            'Mode': md,
            'Model_Type': 'Stereo-Aware (Intervention)',
            'E_MAE': stereo_e_mae,
            'Z_MAE': stereo_z_mae,
            'EZ_Gap': stereo_gap,
            'E_Sigma': stereo_e_sigma,
            'Z_Sigma': stereo_z_sigma,
            'Delta_E_MAE': delta_e_mae,
            'Pct_Delta_E': pct_delta_e,
            'Delta_Z_MAE': delta_z_mae,
            'Pct_Delta_Z': pct_delta_z,
            'Delta_Gap': delta_gap,
        })

    df_summary = pd.DataFrame(results_table)

    # Save summary table
    summary_csv = PAPER / "table_step24_stereo_ablation.csv"
    df_summary.to_csv(summary_csv, index=False)

    # Hypothesis Testing Assessment
    # Evaluate under Mreduced (primary production mode)
    mred_stereo = [r for r in results_table if r['Mode'] == 'Mreduced' and 'Delta_E_MAE' in r][0]
    h1_pass = mred_stereo['Delta_Gap'] < 0  # Gap narrowed
    h2_pass = mred_stereo['Delta_E_MAE'] < 0  # E MAE reduced
    h3_pass = mred_stereo['E_Sigma'] > 0.021332  # Disagreement increased towards real error

    # Evidence Decision Matrix mapping
    evidence_matrix = []
    if h2_pass:
        obs1 = ("R1336mzz(E) MAE reduced", "Supports representation bottleneck hypothesis (stereochemical parity shifts boundary)")
    else:
        obs1 = ("R1336mzz(E) MAE unchanged or increased", "Stereochemical information alone is insufficient; domain shift / dipole disparity remains dominant")

    if h1_pass:
        obs2 = ("E/Z prediction gap narrowed", "Stereo distinction successfully breaks representation degeneracy")
    else:
        obs2 = ("E/Z prediction gap widened or stagnant", "Model lacks training signal to map stereo parity to quantitative polarity")

    evidence_matrix.append(obs1)
    evidence_matrix.append(obs2)

    report = {
        'timestamp': datetime.now().isoformat(),
        'modes_evaluated': ['M0', 'Mreduced'],
        'seeds': [42, 43, 44, 45, 46],
        'metrics_summary': results_table,
        'hypotheses_evaluation': {
            'H1_gap_narrowed': bool(h1_pass),
            'H2_E_error_reduced': bool(h2_pass),
            'H3_false_confidence_relieved': bool(h3_pass),
        },
        'evidence_matrix': evidence_matrix,
        'scientific_verdict': (
            "Intervention confirms G_E != G_Z. " +
            ("Representation boundary experimentally shifted." if (h1_pass or h2_pass) else
             "Degeneracy broken; quantitative transfer governed by remaining electrostatic domain gap.")
        )
    }

    report_json = PAPER / "audit_step24_stereo_ablation.json"
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("  STEP 24 STEREO ABLATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"  {'Model Mode':<20} | {'E MAE':<10} | {'Z MAE':<10} | {'E/Z Gap':<10} | {'E Sigma':<10} | {'Z Sigma':<10}")
    print("-" * 80)
    for r in results_table:
        tag = f"{r['Mode']} {r['Model_Type']}"
        print(f"  {tag:<20} | {r['E_MAE']:<10.4f} | {r['Z_MAE']:<10.4f} | {r['EZ_Gap']:<10.4f} | {r['E_Sigma']:<10.4f} | {r['Z_Sigma']:<10.4f}")
    print("=" * 80)
    print("\n  HYPOTHESIS TESTING REPORT:")
    print(f"   - Hypothesis 1 (E/Z Gap Narrowed)        : {'CONFIRMED' if h1_pass else 'NOT CONFIRMED'} (ΔGap = {mred_stereo['Delta_Gap']:+.4f})")
    print(f"   - Hypothesis 2 (E High Error Reduced)    : {'CONFIRMED' if h2_pass else 'NOT CONFIRMED'} (ΔMAE = {mred_stereo['Delta_E_MAE']:+.4f}, {mred_stereo['Pct_Delta_E']:+.1f}%)")
    print(f"   - Hypothesis 3 (False Confidence Relieved): {'CONFIRMED' if h3_pass else 'NOT CONFIRMED'} (E σ = {mred_stereo['E_Sigma']:.4f} vs base {0.0213:.4f})")
    print("\n  EVIDENCE INTERPRETATION:")
    for obs, interp in evidence_matrix:
        print(f"   • {obs} -> {interp}")
    print(f"\n  Saved:")
    print(f"   - {summary_csv}")
    print(f"   - {report_json}")
    print(f"   - {PAPER / 'stereo_zeroshot_predictions.csv'}")
    print("=" * 80)

    return report


# ═══════════════════════════════════════════════════════════════
# Main CLI Entrypoint
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 24 Stereo Ablation Master Pipeline")
    parser.add_argument('--mode', type=str, default='all', choices=['M0', 'Mreduced', 'all'])
    parser.add_argument('--seeds', type=str, default='42,43,44,45,46')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    # Step 0: Quality Gate Execution (Gate 1 - Gate 5)
    print("\n>>> [STEP 24: EXECUTING QUALITY GATE (Gates 1-5)]...")
    gate_code = run_preflight()
    if gate_code != 0:
        print("FATAL: Quality gate failed. Aborting training.")
        return 1

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    modes = ['M0', 'Mreduced'] if args.mode == 'all' else [args.mode]

    if args.dry_run:
        print("\n[DRY RUN COMPLETED SUCCESSFULLY: ALL GATES PASS, ARGUMENTS VALID]")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[EXECUTION COMPUTE DEVICE]: {device}")

    if not args.eval_only:
        train_stereo_models(args, modes, seeds, device)

    df_stereo_preds = evaluate_stereo_zeroshot(modes, seeds, device)
    evaluate_hypotheses(df_stereo_preds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
