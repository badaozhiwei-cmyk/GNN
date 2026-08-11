"""Extract trained LORO fused-system embeddings and quantify latent distances.

The saved representation is the global-node embedding before condition features
and the prediction MLP are applied. It is therefore a learned representation of
the cation-anion-refrigerant graph, not a refrigerant-only embedding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.append(str(ROOT / "GNN_for_property_prediction"))

from Dataset_v5 import IL_set_v5  # noqa: E402
from Model_v5 import IL_GAT_v5  # noqa: E402


def _load_dataset_and_model(target_ref: str, seed: int, split_mode: str, val_id: int, checkpoint: str | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_args = {"add_global": True, "use_ani_mw": False, "no_mol_embedding": False}
    model_args = {
        "emb_dim": 300,
        "pool": "global",
        "use_ani_mw": False,
        "dropout_rate": 0.2,
        "no_mol_embedding": False,
        "add_global": True,
    }
    dataset = IL_set_v5(path=str(ROOT / "processed_tri_data"), args=dataset_args)

    experiment_id = "full" if split_mode == "full" else f"hfc_only_val{val_id}"
    run_dir = ROOT / "checkpoints_v5" / "loro" / experiment_id / target_ref / "gat"
    checkpoint_path = Path(checkpoint) if checkpoint else run_dir / f"best_model_seed{seed}.pt"
    scaler_path = run_dir / "scalers.pkl"
    split_path = (
        ROOT / "splits_loro" / f"split_L4_{target_ref}.npz"
        if split_mode == "full"
        else ROOT / "Phase4_Scientific_Validation" / "HFC_LORO_Splits"
        / f"split_hfc_loro_{target_ref}_val{val_id}.npz"
    )
    for required in [checkpoint_path, scaler_path, split_path]:
        if not required.exists():
            raise FileNotFoundError(f"Required trained-run artifact is missing: {required}")

    scalers = joblib.load(scaler_path)
    if len(scalers) != 7:
        raise ValueError(f"Expected 7 condition scalers, found {len(scalers)}")
    dataset.scalers = scalers
    dataset.means = np.asarray([s.mean_[0] for s in scalers], dtype=np.float32)
    dataset.scales = np.asarray([s.scale_[0] for s in scalers], dtype=np.float32)

    model = IL_GAT_v5(model_args).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return dataset, model, device, checkpoint_path, scaler_path, split_path, experiment_id


def _global_embeddings(model: IL_GAT_v5, graph) -> torch.Tensor:
    mol_type = graph.mol_type
    normal = mol_type < 3
    global_nodes = mol_type == 3
    h = torch.zeros(graph.x.shape[0], model.emb_dim, device=graph.x.device)
    h[normal] = (
        model.x_embedding1(graph.x[normal, 0])
        + model.x_embedding2(graph.x[normal, 1])
        + model.x_embedding3(graph.x[normal, 2])
        + model.x_embedding4(graph.x[normal, 3])
        + model.x_embedding5(graph.x[normal, 4])
        + model.x_embedding6(graph.x[normal, 5])
        + model.x_embedding7(graph.x[normal, 6])
        + model.mol_embedding(mol_type[normal])
    )
    h[global_nodes] = model.global_token
    edge_emb = (
        model.edge_embedding1(graph.edge_attr[:, 0])
        + model.edge_embedding2(graph.edge_attr[:, 1])
        + model.edge_embedding3(graph.edge_attr[:, 2])
    )
    x, _ = model.l1(h, graph.edge_index, edge_attr=edge_emb, return_attention_weights=True)
    x = model.act(x)
    x, _ = model.l2(x, graph.edge_index, edge_attr=edge_emb, return_attention_weights=True)
    x = model.act(x)
    x, _ = model.l3(x, graph.edge_index, edge_attr=edge_emb, return_attention_weights=True)
    return model.extract(model.act(x), graph)


def analyze(target_ref: str, seed: int, split_mode: str = "full", val_id: int = 1, checkpoint: str | None = None) -> pd.DataFrame:
    dataset, model, device, checkpoint_path, scaler_path, split_path, experiment_id = _load_dataset_and_model(
        target_ref, seed, split_mode, val_id, checkpoint
    )
    df = pd.read_csv(ROOT / "index_with_anion.csv").reset_index(drop=True)
    split = np.load(split_path, allow_pickle=False)
    train_key, test_key = ("train", "test") if split_mode == "full" else ("train_idx", "test_idx")
    selected_indices = np.concatenate([split[train_key], split[test_key]]).astype(int)
    loader = DataLoader(torch.utils.data.Subset(dataset, selected_indices.tolist()), batch_size=128, shuffle=False)

    embeddings = []
    with torch.no_grad():
        cursor = 0
        for graph, _, _ in loader:
            graph = graph.to(device)
            x_g = _global_embeddings(model, graph).cpu().numpy()
            batch_indices = selected_indices[cursor : cursor + len(x_g)]
            embeddings.extend(zip(batch_indices.tolist(), x_g))
            cursor += len(x_g)

    rows = []
    centroids = {}
    for ref_name in sorted(df.loc[selected_indices, "refrigerant"].unique()):
        vectors = [vec for idx, vec in embeddings if df.loc[idx, "refrigerant"] == ref_name]
        if vectors:
            centroids[ref_name] = np.mean(np.stack(vectors), axis=0)

    target_vec = centroids[target_ref]
    target_norm = target_vec / max(np.linalg.norm(target_vec), 1e-12)
    for ref_name, vector in centroids.items():
        if ref_name == target_ref:
            continue
        norm = vector / max(np.linalg.norm(vector), 1e-12)
        rows.append({
            "held_out_refrigerant": target_ref,
            "training_refrigerant": ref_name,
            "cosine_similarity": float(np.dot(target_norm, norm)),
            "euclidean_distance": float(np.linalg.norm(target_vec - vector)),
            "n_target_systems": int((df.loc[selected_indices, "refrigerant"] == target_ref).sum()),
            "n_training_systems": int((df.loc[selected_indices, "refrigerant"] == ref_name).sum()),
        })

    result = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)
    output_dir = ROOT / "results_v5" / "loro" / experiment_id / "embedding_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"LORO_{target_ref}_seed{seed}_latent_distances.csv"
    result.to_csv(output_path, index=False)
    manifest = {
        "representation": "trained fused-system global-node embedding before T/P and MLP",
        "held_out_refrigerant": target_ref,
        "split_mode": split_mode,
        "val_id": val_id if split_mode == "hfc-only" else None,
        "experiment_id": experiment_id,
        "seed": seed,
        "checkpoint": os.path.relpath(checkpoint_path, ROOT),
        "scaler": os.path.relpath(scaler_path, ROOT),
        "split": os.path.relpath(split_path, ROOT),
        "output": os.path.relpath(output_path, ROOT),
        "limitation": "Centroids can be influenced by ionic-liquid coverage; they are not refrigerant-only embeddings.",
    }
    (output_dir / f"LORO_{target_ref}_seed{seed}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(result.head(10).to_string(index=False))
    print(f"Saved {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="R134a")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["full", "hfc-only"], default="full")
    parser.add_argument("--val-id", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    analyze(args.ref, args.seed, args.split_mode, args.val_id, args.checkpoint)
