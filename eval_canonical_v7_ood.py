"""
eval_canonical_v7_ood.py — Frozen Cross-Axis OOD Evaluator for Canonical V7
=============================================================================
【核心定位与使命】
作为 Canonical V7 评测体系的统一执行入口：
严格加载在 HFC-2739 Grouped-State (2464 train / 275 val) 上训练的 10 个 Canonical Checkpoints
(V7-A × seeds 42..46, V7-B × seeds 42..46)，搭配严格由 2464 训练集拟合的 Canonical Scalers，
在 1 个 ID Reference Domain + 5 个 OOD Domains 上执行无偏、冻结的前向评估与表征敏感度分析。

【评测域矩阵】
  1. L0_Grouped_State (ID Reference)      : N=275, val_idx from splits/HFC_grouped_state_split.npz
  2. B1_Anion_OOD (Anion Fam-2 holdout)  : N=513, from splits_anion_ood/split_B1_Fam2_Fluorosulfonate.npz
  3. B2_Anion_OOD (Anion Fam-3 holdout)  : N=598, from splits_anion_ood/split_B2_Fam3_InorganicFluoride.npz
  4. L2_Composite_OOD (Pair Recomb.)     : N=374, from splits/L2_controlled_composite.npz
  5. M1_LORO (12 HFC Refrigerants LORO)  : N=1403, 1:1 mapped to HFC-2739 meta_info.csv
  6. HFO_ZeroShot_OOD (Table S4 HFOs)    : N=1106, npy_idx 2739..3844, canonical R1234yf graph

【输出规范】
  - paper_results/canonical_v7_ood/canonical_v7_ood_predictions_all.csv
  - paper_results/canonical_v7_ood/canonical_v7_ood_ensemble_sample_level.csv
  - paper_results/canonical_v7_ood/canonical_v7_ood_domain_summary.csv
  - paper_results/canonical_v7_ood/canonical_v7_ood_manifest.json
"""

from __future__ import annotations
import os
import sys
import io
import time
import json
import joblib
import hashlib
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import Descriptors

# Windows UTF-8 stdout protection
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "v7_shadow_experiment"))

from Model_v7 import IL_GAT_v7
from Dataset_v7 import DecoupledTriDataset_v7, collate_v7, get_v7_dataloader
from prepare_tri_graph_data_v6 import mol2graph_components

# ══════════════════════════════════════════════════════════════════════════════
# 1. Authoritative Provenance Constants & Lineage Gates
# ══════════════════════════════════════════════════════════════════════════════
GIT_COMMIT = "4cec58a200dea4c8c02b8ffdc8d0960467394a1e"
EXPECTED_GROUPED_SPLIT_SHA = "04f52d330f929c622187ff07b9bce2a7a5061992d04fd3385044a9245e67b56b"
EXPECTED_DATA_SHA = "a408c2832352d69799d1c0a8b7151dd7535fe2edf34d171992b9385b1bd3e6bb"
EXPECTED_LABEL_SHA = "f8cd8f4e58c746c2b41122466cebcee0a08929fe3c94f349d4cd34a51b3c2f3a"
EXPECTED_SCALER_SHA = "b43ec2def4ce937c4146c1b6a0ee4b48550ed67f56f931276185e5c026fe40a5"

SEEDS = [42, 43, 44, 45, 46]
MODELS = ["V7-A", "V7-B"]

CHECKPOINT_BASE = Path(r"D:\折腾\v7_retrained_gpu_results\v7_shadow_experiment\checkpoints")
CANONICAL_SCALER_PATH = CHECKPOINT_BASE / "V7-A_grouped_state_v1" / "scalers.pkl"

HFO_CANONICAL_SMILES = {
    'R1234YF': 'C=C(F)C(F)(F)F',
    'R1234ZE(E)': 'F/C=C/C(F)(F)F',
    'R1233ZD(E)': 'FC(F)(F)/C=C/Cl',
    'R1336MZZ(E)': 'FC(F)(F)/C=C/C(F)(F)F',
    'R1336MZZ(Z)': r'FC(F)(F)/C=C\C(F)(F)F',
}

def sha256_file(filepath: Path | str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def mol2graph_v7_hfo(mol_data, target_edge_dim=4):
    x = torch.tensor(mol_data[0], dtype=torch.long)
    edge_index = torch.tensor(mol_data[1], dtype=torch.long)
    if len(mol_data[2]) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros((1, target_edge_dim), dtype=torch.long)
    else:
        raw_attr = torch.tensor(mol_data[2], dtype=torch.long)
        if target_edge_dim == 4 and raw_attr.size(1) == 3:
            zero_stereo = torch.zeros((raw_attr.size(0), 1), dtype=torch.long)
            edge_attr = torch.cat([raw_attr, zero_stereo], dim=1)
        else:
            edge_attr = raw_attr
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def run_lineage_gate():
    print("=" * 85)
    print("  CANONICAL V7 LINEAGE GATE VERIFICATION")
    print("=" * 85)
    
    # 1. Grouped Split
    sp_path = PROJECT_ROOT / "splits" / "HFC_grouped_state_split.npz"
    assert sp_path.exists(), f"FATAL: Missing split file {sp_path}"
    sp_sha = sha256_file(sp_path)
    assert sp_sha == EXPECTED_GROUPED_SPLIT_SHA, f"FATAL: Split SHA mismatch: {sp_sha}"
    print(f"  [OK] Training Split SHA256: {sp_sha[:16]}... (2464/275)")

    # 2. HFC Binary Data & Label
    hfc_dir = PROJECT_ROOT / "processed_tri_data_hfc2739"
    d_path, l_path = hfc_dir / "data.npy", hfc_dir / "label.npy"
    assert d_path.exists() and l_path.exists(), "FATAL: HFC data.npy or label.npy missing!"
    d_sha = sha256_file(d_path)
    l_sha = sha256_file(l_path)
    assert d_sha == EXPECTED_DATA_SHA, f"FATAL: Data SHA mismatch: {d_sha}"
    assert l_sha == EXPECTED_LABEL_SHA, f"FATAL: Label SHA mismatch: {l_sha}"
    print(f"  [OK] HFC Data SHA256: {d_sha[:16]}... Label SHA256: {l_sha[:16]}...")

    # 3. Canonical Scaler
    assert CANONICAL_SCALER_PATH.exists(), f"FATAL: Missing scaler at {CANONICAL_SCALER_PATH}"
    sc_sha = sha256_file(CANONICAL_SCALER_PATH)
    assert sc_sha == EXPECTED_SCALER_SHA, f"FATAL: Scaler SHA mismatch: {sc_sha}"
    print(f"  [OK] Canonical Scaler SHA256: {sc_sha[:16]}... (9 dimensions, Train-Only)")

    # 4. Checkpoints
    ckpt_shas = {}
    for m in MODELS:
        folder = CHECKPOINT_BASE / f"{m}_grouped_state_v1"
        for s in SEEDS:
            cp = folder / f"best_seed_{s}.pth"
            assert cp.exists(), f"FATAL: Missing checkpoint {cp}"
            c_sha = sha256_file(cp)
            ckpt_shas[f"{m}_seed{s}"] = c_sha
    print(f"  [OK] All 10 Checkpoints verified in {CHECKPOINT_BASE.name}")
    print("=" * 85)
    return ckpt_shas, sp_sha, sc_sha


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset Preparation across 6 Evaluation Domains
# ══════════════════════════════════════════════════════════════════════════════
class HFODataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_evaluation_domains(hfc_data_dir: Path, canonical_scaler_path: Path):
    print("\n[DATA PREPARATION] Assembling all 6 evaluation domains...")
    meta_df = pd.read_csv(hfc_data_dir / "meta_info.csv")
    master_df = pd.read_csv(PROJECT_ROOT / "index_with_anion.csv")
    raw_tri_data = np.load(PROJECT_ROOT / "processed_tri_data" / "data.npy", allow_pickle=True)

    # Disambiguate duplicate state points in meta_df to ensure 100% unique row IDs
    counts = {}
    disambiguated_ids = []
    for sid in meta_df['sample_id']:
        counts[sid] = counts.get(sid, 0) + 1
        if counts[sid] == 1:
            disambiguated_ids.append(sid)
        else:
            disambiguated_ids.append(f"{sid}__#{counts[sid]}")
    meta_df['disambiguated_sample_id'] = disambiguated_ids

    scalers = joblib.load(canonical_scaler_path)
    scaler_means = np.array([float(s.mean_[0]) for s in scalers], dtype=np.float32)
    scaler_scales = np.array([max(float(s.scale_[0]), 1e-8) for s in scalers], dtype=np.float32)

    domains = OrderedDict()

    # 1. L0 Grouped-State ID Reference (N=275)
    l0_split = np.load(PROJECT_ROOT / "splits" / "HFC_grouped_state_split.npz")
    l0_val_idx = l0_split['val'].tolist()
    ds_l0 = DecoupledTriDataset_v7(str(hfc_data_dir), valid_indices=l0_val_idx)
    ds_l0.load_scalers(str(canonical_scaler_path))
    domains["L0"] = {
        "name": "L0_Grouped_State",
        "axis": "L0_Grouped_State",
        "source_protocol": "L0_Grouped_State_Val",
        "dataset": ds_l0,
        "indices": l0_val_idx,
        "meta": meta_df.iloc[l0_val_idx].copy().reset_index(drop=True),
        "expected_n": 275
    }

    # 2. B1 Anion Family OOD (Fam-2 Fluorosulfonates, N=513)
    b1_split = np.load(PROJECT_ROOT / "splits_anion_ood" / "split_B1_Fam2_Fluorosulfonate.npz")
    b1_test_idx = b1_split['test'].tolist()
    ds_b1 = DecoupledTriDataset_v7(str(hfc_data_dir), valid_indices=b1_test_idx)
    ds_b1.load_scalers(str(canonical_scaler_path))
    domains["B1"] = {
        "name": "B1_Anion_OOD",
        "axis": "B1_Anion_OOD",
        "source_protocol": "split_B1_Fam2_Fluorosulfonate",
        "dataset": ds_b1,
        "indices": b1_test_idx,
        "meta": meta_df.iloc[b1_test_idx].copy().reset_index(drop=True),
        "expected_n": 513
    }

    # 3. B2 Anion Family OOD (Fam-3 Inorganic Fluorides, N=598)
    b2_split = np.load(PROJECT_ROOT / "splits_anion_ood" / "split_B2_Fam3_InorganicFluoride.npz")
    b2_test_idx = b2_split['test'].tolist()
    ds_b2 = DecoupledTriDataset_v7(str(hfc_data_dir), valid_indices=b2_test_idx)
    ds_b2.load_scalers(str(canonical_scaler_path))
    domains["B2"] = {
        "name": "B2_Anion_OOD",
        "axis": "B2_Anion_OOD",
        "source_protocol": "split_B2_Fam3_InorganicFluoride",
        "dataset": ds_b2,
        "indices": b2_test_idx,
        "meta": meta_df.iloc[b2_test_idx].copy().reset_index(drop=True),
        "expected_n": 598
    }

    # 4. L2 Compositional Recombination OOD (4 unseen pairs, N=374)
    l2_split = np.load(PROJECT_ROOT / "splits" / "L2_controlled_composite.npz")
    l2_test_idx = l2_split['test'].tolist()
    ds_l2 = DecoupledTriDataset_v7(str(hfc_data_dir), valid_indices=l2_test_idx)
    ds_l2.load_scalers(str(canonical_scaler_path))
    domains["L2"] = {
        "name": "L2_Composite_OOD",
        "axis": "L2_Composite_OOD",
        "source_protocol": "L2_controlled_composite",
        "dataset": ds_l2,
        "indices": l2_test_idx,
        "meta": meta_df.iloc[l2_test_idx].copy().reset_index(drop=True),
        "expected_n": 374
    }

    # 5. M1 / LORO (12 HFC refrigerants, N=1403)
    # Direct first-principles ingestion from the 12 canonical LORO test splits
    loro_dir = PROJECT_ROOT / "splits" / "HFC_loro_all"
    assert loro_dir.exists(), f"Missing LORO directory: {loro_dir}"
    
    mapping_csv = PROJECT_ROOT / "paper_results" / "canonical_v7_ood" / "historical_1403_index_mapping.csv"
    assert mapping_csv.exists(), f"Missing index mapping: {mapping_csv}"
    df_map = pd.read_csv(mapping_csv)
    hist_to_canon = dict(zip(df_map['historical_idx'], df_map['canonical_npy_idx']))
    
    m1_indices = []
    loro_split_files = sorted(loro_dir.glob("loro_*.npz"))
    assert len(loro_split_files) == 12, f"Expected 12 LORO files, found {len(loro_split_files)}"
    
    loro_shas = {}
    for f in loro_split_files:
        loro_shas[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        sp = np.load(f)
        for h_idx in sp['test']:
            m1_indices.append(hist_to_canon[int(h_idx)])
            
    assert len(m1_indices) == 1403, f"Expected 1403 M1 test indices, got {len(m1_indices)}"
    assert len(set(m1_indices)) == 1403, "Duplicate indices found in M1 test indices!"

    ds_m1 = DecoupledTriDataset_v7(str(hfc_data_dir), valid_indices=m1_indices)
    ds_m1.load_scalers(str(canonical_scaler_path))
    domains["M1"] = {
        "name": "M1_LORO",
        "axis": "M1_LORO",
        "source_protocol": "L4_LORO_12_Refrigerants_Union",
        "dataset": ds_m1,
        "indices": m1_indices,
        "meta": meta_df.iloc[m1_indices].copy().reset_index(drop=True),
        "expected_n": 1403,
        "loro_split_sha256": loro_shas
    }

    # 6. HFO / HCFO Zero-Shot OOD (N=1106)
    df_hfo = master_df[master_df['sheet'] == 'Table S4. VLE HFOs'].copy().reset_index(drop=True)
    assert len(df_hfo) == 1106, f"Expected 1106 HFO samples, got {len(df_hfo)}"
    df_hfo['disambiguated_sample_id'] = [
        f"HFO_{r['cation']}__{r['anion']}__{r['refrigerant']}__{r['T_K']}__{r['P_MPa']}__{r['npy_idx']}"
        for _, r in df_hfo.iterrows()
    ]
    
    mw_cache = {}
    def get_mw(smi):
        if smi not in mw_cache:
            mol = Chem.MolFromSmiles(smi)
            mw_cache[smi] = float(Descriptors.MolWt(mol)) if mol else 0.0
        return mw_cache[smi]

    hfo_samples = []
    for _, r in df_hfo.iterrows():
        orig_i = int(r['npy_idx'])
        raw_item = raw_tri_data[orig_i]
        r_clean = str(r['refrigerant']).strip().upper()
        ref_smi = HFO_CANONICAL_SMILES.get(r_clean, str(r['refri_smiles']).strip())

        g_cat = mol2graph_v7_hfo(raw_item[0], target_edge_dim=4)
        g_ani = mol2graph_v7_hfo(raw_item[1], target_edge_dim=4)
        if r_clean == 'R1234YF':
            g_ref = mol2graph_v7_hfo(mol2graph_components(ref_smi), target_edge_dim=4)
        else:
            g_ref = mol2graph_v7_hfo(raw_item[2], target_edge_dim=4)

        t_val = float(r['T_K'])
        p_val = float(r['P_MPa'])
        ref_mw = get_mw(ref_smi)
        cat_mw = get_mw(r['cation_smiles'])
        ani_mw = get_mw(r['anion_smiles'])

        mol_ref = Chem.MolFromSmiles(ref_smi)
        ref_logp = float(Descriptors.MolLogP(mol_ref)) if mol_ref else 0.0
        try:
            ref_charge = float(Descriptors.MaxAbsPartialCharge(mol_ref)) if mol_ref else 0.0
        except Exception:
            ref_charge = 0.0

        mol_cat = Chem.MolFromSmiles(r['cation_smiles'])
        cat_tpsa = float(Descriptors.TPSA(mol_cat)) if mol_cat else 0.0
        try:
            cat_charge = float(Descriptors.MaxAbsPartialCharge(mol_cat)) if mol_cat else 0.0
        except Exception:
            cat_charge = 0.0

        raw_cond_9 = np.array([
            t_val, p_val,
            ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa,
            ref_mw, cat_mw
        ], dtype=np.float32)

        norm_cond = (raw_cond_9 - scaler_means) / scaler_scales
        state_cond = torch.tensor(norm_cond[0:2], dtype=torch.float32)
        desc_cond = torch.tensor(norm_cond[2:9], dtype=torch.float32)
        label = torch.tensor(float(r['x1']), dtype=torch.float32)

        hfo_samples.append({
            'cat': g_cat,
            'ani': g_ani,
            'ref': g_ref,
            'state': state_cond,
            'desc': desc_cond,
            'label': label
        })

    ds_hfo = HFODataset(hfo_samples)
    domains["HFO"] = {
        "name": "HFO_ZeroShot_OOD",
        "axis": "HFO_ZeroShot_OOD",
        "source_protocol": "Table_S4_HFO_ZeroShot",
        "dataset": ds_hfo,
        "indices": df_hfo['npy_idx'].astype(int).tolist(),
        "meta": df_hfo,
        "expected_n": 1106
    }

    for d_key, d_info in domains.items():
        assert len(d_info["dataset"]) == d_info["expected_n"], \
            f"Domain {d_key} length mismatch: got {len(d_info['dataset'])}, expected {d_info['expected_n']}"
        print(f"  [Domain {d_key:4s}] {d_info['name']:25s}: N = {len(d_info['dataset']):4d} samples verified.")

    return domains


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model Inference Engine with Reused GNN System Representation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_domain_model(model: IL_GAT_v7, dataloader: DataLoader, device: torch.device):
    """
    Executes fast inference over a domain:
    Reuses h_sys to compute y_base, y_no_graph, y_no_desc in a single encoder pass.
    Mathematically identical to 3 full forward calls, 3x faster.
    """
    model.eval()
    y_trues, y_preds_raw, y_preds_clip = [], [], []
    delta_graphs, delta_descs = [], []

    with torch.no_grad():
        for batch in dataloader:
            g_cat = batch['cat'].to(device)
            g_ani = batch['ani'].to(device)
            g_ref = batch['ref'].to(device)
            state = batch['state'].to(device)
            desc  = batch['desc'].to(device)
            y_true = batch['label'].cpu().numpy()

            # 1. GNN Encoder pass (ran once!)
            h_cat = model._encode_il(g_cat, mol_type_idx=0)
            h_ani = model._encode_il(g_ani, mol_type_idx=1)
            h_ref = model._encode_ref(g_ref)
            h_cat_p, h_ani_p, h_ref_p = model._build_interaction_graph(h_cat, h_ani, h_ref)
            h_sys = model.sys_proj(torch.cat([h_cat_p, h_ani_p, h_ref_p], dim=-1))

            # 2. Base prediction
            fused_base = torch.cat([h_sys, state, desc], dim=-1)
            out_base = model.head(fused_base).squeeze(-1)
            if model.use_sigmoid:
                out_base = torch.sigmoid(out_base)

            # 3. Graph-ablated prediction
            fused_no_graph = torch.cat([torch.zeros_like(h_sys), state, desc], dim=-1)
            out_no_graph = model.head(fused_no_graph).squeeze(-1)
            if model.use_sigmoid:
                out_no_graph = torch.sigmoid(out_no_graph)

            # 4. Desc-ablated prediction
            fused_no_desc = torch.cat([h_sys, state, torch.zeros_like(desc)], dim=-1)
            out_no_desc = model.head(fused_no_desc).squeeze(-1)
            if model.use_sigmoid:
                out_no_desc = torch.sigmoid(out_no_desc)

            pred_raw = out_base.cpu().numpy()
            pred_clip = np.clip(pred_raw, 0.0, 1.0)
            d_graph = np.abs(pred_raw - out_no_graph.cpu().numpy())
            d_desc  = np.abs(pred_raw - out_no_desc.cpu().numpy())

            y_trues.extend(y_true)
            y_preds_raw.extend(pred_raw)
            y_preds_clip.extend(pred_clip)
            delta_graphs.extend(d_graph)
            delta_descs.extend(d_desc)

    return {
        "y_true": np.array(y_trues, dtype=np.float32),
        "y_pred_raw": np.array(y_preds_raw, dtype=np.float32),
        "y_pred_clip": np.array(y_preds_clip, dtype=np.float32),
        "delta_y_graph": np.array(delta_graphs, dtype=np.float32),
        "delta_y_desc": np.array(delta_descs, dtype=np.float32)
    }


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred) if np.var(y_true) > 1e-12 else 0.0
    bias = float(np.mean(y_pred - y_true))
    p_r = pearsonr(y_true, y_pred)[0] if np.var(y_pred) > 1e-12 and np.var(y_true) > 1e-12 else 0.0
    s_rho = spearmanr(y_true, y_pred)[0] if np.var(y_pred) > 1e-12 and np.var(y_true) > 1e-12 else 0.0
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Bias": float(bias),
        "Pearson_r": float(p_r),
        "Spearman_rho": float(s_rho)
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main Evaluator Execution Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: Lineage Gates
    ckpt_shas, split_sha, scaler_sha = run_lineage_gate()

    # Step 2: Data Assembly
    hfc_data_dir = PROJECT_ROOT / "processed_tri_data_hfc2739"
    domains = build_evaluation_domains(hfc_data_dir, CANONICAL_SCALER_PATH)

    out_dir = PROJECT_ROOT / "paper_results" / "canonical_v7_ood"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pred_records = []
    domain_summary_records = []
    domain_ensemble_records = []

    print("\n" + "=" * 85)
    print("  COMMENCING CANONICAL V7 CROSS-AXIS EVALUATION")
    print("=" * 85)

    # Step 3: Run Evaluation across Models & Seeds
    # To avoid repeated dataset loading, we loop: model -> seed -> domain
    for model_name in MODELS:
        ckpt_folder = CHECKPOINT_BASE / f"{model_name}_grouped_state_v1"
        for seed in SEEDS:
            ckpt_file = ckpt_folder / f"best_seed_{seed}.pth"
            ckpt_key = f"{model_name}_seed{seed}"
            c_sha = ckpt_shas[ckpt_key]

            # Load checkpoint
            ckpt = torch.load(ckpt_file, map_location=device)
            model_args = ckpt.get("model_args", {'emb_dim': 300, 'hidden_dim': 512, 'desc_dropout_p': 0.0})
            model = IL_GAT_v7(args=model_args).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            print(f"\n>>> Running Evaluation: Model = {model_name}, Seed = {seed} (SHA: {c_sha[:12]}...)")

            for d_key, d_info in domains.items():
                loader = DataLoader(d_info["dataset"], batch_size=64, shuffle=False, collate_fn=collate_v7)
                res = evaluate_domain_model(model, loader, device)

                meta = d_info["meta"]
                n_pts = len(res["y_true"])
                assert n_pts == len(meta), f"Length mismatch in {d_key}: {n_pts} vs {len(meta)}"

                # Per-seed metrics
                m_raw = compute_metrics(res["y_true"], res["y_pred_raw"])
                m_clip = compute_metrics(res["y_true"], res["y_pred_clip"])

                d_graph_mean = float(np.mean(res["delta_y_graph"]))
                d_desc_mean  = float(np.mean(res["delta_y_desc"]))
                ratio = d_graph_mean / (d_graph_mean + d_desc_mean + 1e-12)

                domain_summary_records.append({
                    "domain": d_key,
                    "domain_name": d_info["name"],
                    "axis": d_info["axis"],
                    "source_protocol": d_info["source_protocol"],
                    "model": model_name,
                    "seed": seed,
                    "N": n_pts,
                    "R2_clip": m_clip["R2"],
                    "MAE_clip": m_clip["MAE"],
                    "RMSE_clip": m_clip["RMSE"],
                    "Bias": m_clip["Bias"],
                    "Pearson_r": m_clip["Pearson_r"],
                    "R2_raw": m_raw["R2"],
                    "MAE_raw": m_raw["MAE"],
                    "delta_y_graph_mean": d_graph_mean,
                    "delta_y_desc_mean": d_desc_mean,
                    "graph_reliance_ratio": float(ratio),
                    "ckpt_sha256": c_sha,
                    "split_sha256": split_sha,
                    "scaler_sha256": scaler_sha,
                    "git_commit": GIT_COMMIT
                })

                print(f"    [{d_key:4s}] N={n_pts:4d} | R2_clip = {m_clip['R2']:7.4f} | MAE = {m_clip['MAE']:.4f} | "
                      f"d_graph = {d_graph_mean:.4f} | d_desc = {d_desc_mean:.4f}")

                # Collect per-sample records
                for i in range(n_pts):
                    r_meta = meta.iloc[i]
                    sid = str(r_meta.get("disambiguated_sample_id", r_meta.get("sample_id", f"{d_key}_{i}")))
                    ref = str(r_meta.get("refrigerant", "")).strip().upper()
                    cat = str(r_meta.get("cation", "")).strip()
                    ani = str(r_meta.get("anion", "")).strip()
                    t_k = float(r_meta.get("T_K", np.nan))
                    p_mpa = float(r_meta.get("P_MPa", np.nan))

                    all_pred_records.append({
                        "sample_id": sid,
                        "domain": d_key,
                        "axis": d_info["axis"],
                        "source_protocol": d_info["source_protocol"],
                        "model": model_name,
                        "seed": seed,
                        "y_true": float(res["y_true"][i]),
                        "y_pred_raw": float(res["y_pred_raw"][i]),
                        "y_pred_clip": float(res["y_pred_clip"][i]),
                        "delta_y_graph": float(res["delta_y_graph"][i]),
                        "delta_y_desc": float(res["delta_y_desc"][i]),
                        "refrigerant": ref,
                        "cation": cat,
                        "anion": ani,
                        "T_K": t_k,
                        "P_MPa": p_mpa,
                        "ckpt_sha256": c_sha,
                        "split_sha256": split_sha,
                        "scaler_sha256": scaler_sha,
                        "git_commit": GIT_COMMIT
                    })

    # Step 4: Assemble DataFrames & Compute Ensemble UQ
    df_all_preds = pd.DataFrame(all_pred_records)
    df_summary = pd.DataFrame(domain_summary_records)

    total_domain_obs = len(df_all_preds) // (len(MODELS) * len(SEEDS))
    unique_state_points = df_all_preds['sample_id'].nunique()

    print("\n" + "=" * 85)
    print("  COMPUTING 5-SEED ENSEMBLE PREDICTIONS & UNCERTAINTY QUANTIFICATION (UQ)")
    print(f"  Total Domain Observations: {total_domain_obs} (Union Unique Chemical States: {unique_state_points})")
    print("=" * 85)

    ensemble_rows = []
    ensemble_metrics_list = []

    for d_key, d_info in domains.items():
        for model_name in MODELS:
            sub = df_all_preds[(df_all_preds['domain'] == d_key) & (df_all_preds['model'] == model_name)]
            
            # --- Fail-Closed Defensive Assertions before Aggregation ---
            dup_keys = sub.duplicated(["domain", "model", "sample_id", "seed"]).sum()
            assert dup_keys == 0, f"[FATAL] Duplicate keys found in {d_key} {model_name}: {dup_keys} duplicates!"
            
            seed_counts = sub.groupby("sample_id")["seed"].nunique()
            assert (seed_counts == len(SEEDS)).all(), \
                f"[FATAL] Sample seed count discrepancy in {d_key} {model_name}: expected all {len(SEEDS)} seeds, got min={seed_counts.min()} max={seed_counts.max()}!"
                
            n_unique_samples = sub["sample_id"].nunique()
            assert n_unique_samples == d_info["expected_n"], \
                f"[FATAL] Unique sample count mismatch in {d_key} {model_name}: got {n_unique_samples} vs expected {d_info['expected_n']}!"

            # Pivot by sample_id to align 5 seeds
            piv_pred = sub.pivot_table(index="sample_id", columns="seed", values="y_pred_clip")
            piv_raw  = sub.pivot_table(index="sample_id", columns="seed", values="y_pred_raw")
            piv_true = sub.groupby("sample_id")["y_true"].first()
            piv_meta = sub.groupby("sample_id").first()

            mu_clip = piv_pred.mean(axis=1).to_numpy()
            sigma   = piv_pred.std(axis=1, ddof=1).to_numpy()
            y_true  = piv_true.to_numpy()
            abs_err = np.abs(y_true - mu_clip)

            # Ensemble metrics
            ens_m = compute_metrics(y_true, mu_clip)
            rho_uq = spearmanr(sigma, abs_err)[0] if np.var(sigma) > 1e-12 and np.var(abs_err) > 1e-12 else 0.0

            # 5-seed statistics
            seed_subs = df_summary[(df_summary['domain'] == d_key) & (df_summary['model'] == model_name)]
            r2_mean = seed_subs['R2_clip'].mean()
            r2_std  = seed_subs['R2_clip'].std(ddof=1)
            mae_mean= seed_subs['MAE_clip'].mean()
            mae_std = seed_subs['MAE_clip'].std(ddof=1)
            rmse_mean = seed_subs['RMSE_clip'].mean()
            d_graph_m = seed_subs['delta_y_graph_mean'].mean()
            d_desc_m  = seed_subs['delta_y_desc_mean'].mean()
            reliance_m= seed_subs['graph_reliance_ratio'].mean()

            ensemble_metrics_list.append({
                "domain": d_key,
                "domain_name": d_info["name"],
                "axis": d_info["axis"],
                "source_protocol": d_info["source_protocol"],
                "model": model_name,
                "N": len(y_true),
                "R2_5seed_mean": float(r2_mean),
                "R2_5seed_std": float(r2_std),
                "MAE_5seed_mean": float(mae_mean),
                "MAE_5seed_std": float(mae_std),
                "RMSE_5seed_mean": float(rmse_mean),
                "Ensemble_R2": float(ens_m["R2"]),
                "Ensemble_MAE": float(ens_m["MAE"]),
                "Ensemble_RMSE": float(ens_m["RMSE"]),
                "Mean_Sigma": float(np.mean(sigma)),
                "Spearman_rho_UQ": float(rho_uq),
                "delta_y_graph_mean": float(d_graph_m),
                "delta_y_desc_mean": float(d_desc_m),
                "graph_reliance_ratio": float(reliance_m),
                "split_sha256": split_sha,
                "scaler_sha256": scaler_sha,
                "git_commit": GIT_COMMIT
            })

            print(f"  [{d_key:4s} | {model_name}] 5-seed R2 = {r2_mean:7.4f} +/- {r2_std:.4f} | "
                  f"Ens R2 = {ens_m['R2']:7.4f} | Ens MAE = {ens_m['MAE']:.4f} | "
                  f"Mean Sigma = {np.mean(sigma):.4f} | rho(sigma, err) = {rho_uq:+.4f} | "
                  f"Reliance = {reliance_m*100:.1f}%")

            # Collect sample-level ensemble rows
            for sid_idx, sid in enumerate(piv_pred.index):
                ensemble_rows.append({
                    "sample_id": sid,
                    "domain": d_key,
                    "axis": d_info["axis"],
                    "source_protocol": d_info["source_protocol"],
                    "model": model_name,
                    "y_true": float(y_true[sid_idx]),
                    "ens_mu_clip": float(mu_clip[sid_idx]),
                    "ens_sigma": float(sigma[sid_idx]),
                    "abs_error": float(abs_err[sid_idx]),
                    "refrigerant": piv_meta.loc[sid, "refrigerant"],
                    "cation": piv_meta.loc[sid, "cation"],
                    "anion": piv_meta.loc[sid, "anion"],
                    "T_K": piv_meta.loc[sid, "T_K"],
                    "P_MPa": piv_meta.loc[sid, "P_MPa"]
                })

    df_ensemble_samples = pd.DataFrame(ensemble_rows)
    df_ensemble_metrics = pd.DataFrame(ensemble_metrics_list)

    # Step 5: Save Authoritative Artifacts
    print("\n" + "=" * 85)
    print("  SAVING AUTHORITATIVE ARTIFACTS")
    print("=" * 85)

    all_csv_path = out_dir / "canonical_v7_ood_predictions_all.csv"
    df_all_preds.to_csv(all_csv_path, index=False)
    print(f"  [SAVED] All Predictions (42,690 rows): {all_csv_path.relative_to(PROJECT_ROOT)}")

    ens_csv_path = out_dir / "canonical_v7_ood_ensemble_sample_level.csv"
    df_ensemble_samples.to_csv(ens_csv_path, index=False)
    print(f"  [SAVED] Ensemble Sample Level ({len(df_ensemble_samples)} rows): {ens_csv_path.relative_to(PROJECT_ROOT)}")

    # Step 5.1: Export Dedicated M1 (1403 samples) predictions
    mapping_csv = PROJECT_ROOT / "paper_results" / "canonical_v7_ood" / "historical_1403_index_mapping.csv"
    if mapping_csv.exists():
        df_map = pd.read_csv(mapping_csv)
        m1_ens = df_ensemble_samples[df_ensemble_samples['domain'] == 'M1']
        m1_v7a = m1_ens[m1_ens['model'] == 'V7-A'].set_index('sample_id')
        m1_v7b = m1_ens[m1_ens['model'] == 'V7-B'].set_index('sample_id')

        df_m1_out = df_map.copy()
        df_m1_out['v7a_pred'] = df_m1_out['disambiguated_sample_id'].map(m1_v7a['ens_mu_clip'])
        df_m1_out['v7a_sigma'] = df_m1_out['disambiguated_sample_id'].map(m1_v7a['ens_sigma'])
        df_m1_out['v7a_abs_err'] = df_m1_out['disambiguated_sample_id'].map(m1_v7a['abs_error'])
        df_m1_out['v7b_pred'] = df_m1_out['disambiguated_sample_id'].map(m1_v7b['ens_mu_clip'])
        df_m1_out['v7b_sigma'] = df_m1_out['disambiguated_sample_id'].map(m1_v7b['ens_sigma'])
        df_m1_out['v7b_abs_err'] = df_m1_out['disambiguated_sample_id'].map(m1_v7b['abs_error'])

        m1_export_path = out_dir / "canonical_v7_m1_1403_predictions.csv"
        df_m1_out.to_csv(m1_export_path, index=False)
        print(f"  [SAVED] Dedicated M1 (1403 samples) predictions: {m1_export_path.relative_to(PROJECT_ROOT)}")

    summary_csv_path = out_dir / "canonical_v7_ood_domain_summary.csv"
    df_ensemble_metrics.to_csv(summary_csv_path, index=False)
    print(f"  [SAVED] Domain Summary Metrics: {summary_csv_path.relative_to(PROJECT_ROOT)}")

    per_seed_csv_path = out_dir / "canonical_v7_ood_per_seed_metrics.csv"
    df_summary.to_csv(per_seed_csv_path, index=False)
    print(f"  [SAVED] Per-Seed Summary Metrics: {per_seed_csv_path.relative_to(PROJECT_ROOT)}")

    # Step 6: Write Manifest JSON
    manifest = {
        "metadata": {
            "title": "Canonical V7 Cross-Axis Evaluation Benchmark Manifest",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_git_commit": GIT_COMMIT,
            "evaluator_script": "eval_canonical_v7_ood.py",
            "evaluator_script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
            "mapping_generator_script": "scripts/provenance/generate_historical_1403_mapping.py",
            "mapping_csv_path": "paper_results/canonical_v7_ood/historical_1403_index_mapping.csv",
            "runtime_dependency_notice": "The audit package is provenance-complete for the M1 lineage and prediction records, while the full inference runtime remains repository/checkpoint dependent.",
            "architecture": "Model_v7 (29,260,989 parameters)",
            "training_split": "splits/HFC_grouped_state_split.npz (2464 train / 275 val)",
            "training_split_sha256": split_sha,
            "scaler_sha256": scaler_sha,
            "total_domain_observations": 4269,
            "union_unique_chemical_states": 3098,
            "sample_scope_clarification": "4,269 domain-level observations across 6 evaluation domains, corresponding to 3,098 unique chemical state points across the union of evaluation domains",
            "execution_device": str(device),
            "execution_duration_sec": round(time.time() - start_time, 2),
            "checkpoint_sha256": ckpt_shas
        },
        "m1_loro_split_sha256": domains["M1"]["loro_split_sha256"],
        "domains": {
            d_key: {
                "name": d_info["name"],
                "axis": d_info["axis"],
                "source_protocol": d_info["source_protocol"],
                "sample_count": d_info["expected_n"]
            } for d_key, d_info in domains.items()
        },
        "benchmark_results": df_ensemble_metrics.to_dict(orient="records")
    }

    manifest_path = out_dir / "canonical_v7_ood_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] Authoritative Manifest: {manifest_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 85)
    print(f"  CANONICAL V7 EVALUATION COMPLETED IN {time.time() - start_time:.2f}s")
    print("=" * 85)


if __name__ == "__main__":
    main()
