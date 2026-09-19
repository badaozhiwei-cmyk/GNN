#!/usr/bin/env python3
"""
step24_preflight_stereo.py -- Phase III-B Step 24 Preflight
===========================================================
Rigorous 5-Stage Verification for Stereo-Aware Molecular Graph Featurization.

Checks:
  P1: Input SMILES Stereochemistry Integrity (E/Z identification in RDKit)
  P2: Baseline Featurizer Mathematical Degeneracy Verification (G_E == G_Z)
  P3: Stereo-Aware Featurizer Difference Verification (G_E != G_Z)
  P4: Unique-Source-of-Difference Diff Report (isolated to stereo dimensions)
  P5: Batch-Level Pipeline & Model Input Preservation (DataLoader / Batch check)

Usage:
  uv run python research_pipeline/step24_preflight_stereo.py

Outputs:
  paper_results/audit_step24_preflight_stereo.json
  paper_results/audit_step24_preflight_stereo.csv
"""

import sys
import io
import json
import csv
import pathlib
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    print("FATAL: RDKit is required for stereo preflight.")
    sys.exit(1)

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper_results"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GNN_for_property_prediction"))

from Dataset_v6 import combine_Graph, add_global
from prepare_tri_graph_data_v6 import (
    get_atom_features as baseline_atom_features,
    get_bond_features as baseline_bond_features,
    mol2graph_components as baseline_mol2graph,
    lookup_smiles
)

# Canonical SMILES for R1336mzz isomers
SMILES_E = 'FC(F)(F)/C=C/C(F)(F)F'     # trans-1,1,1,4,4,4-hexafluoro-2-butene
SMILES_Z = r'FC(F)(F)/C=C\C(F)(F)F'    # cis-1,1,1,4,4,4-hexafluoro-2-butene


# ═══════════════════════════════════════════════════════════════
# Stereo-Aware Featurizer Implementation
# ═══════════════════════════════════════════════════════════════

def stereo_atom_features(atom):
    """Extends baseline 7-dim atom features with chiral tag as 8th dimension."""
    base_f = baseline_atom_features(atom)  # 7 dims
    chiral_tag = int(atom.GetChiralTag())  # 0: None, 1: CW, 2: CCW, 3: Other
    return base_f + [chiral_tag]           # 8 dims


def stereo_bond_features(bond):
    """Extends baseline 3-dim bond features with stereo parity as 4th dimension."""
    base_f = baseline_bond_features(bond)  # 3 dims
    # bond.GetStereo(): 0: None, 1: Any, 2: Z, 3: E, 4: Cis, 5: Trans
    stereo_parity = int(bond.GetStereo())
    return base_f + [stereo_parity]        # 4 dims


def stereo_mol2graph_components(smiles_string):
    """Constructs stereo-aware molecular graph representation."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    node_f = [stereo_atom_features(atom) for atom in mol.GetAtoms()]
    edge_index = [[], []]
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        f = stereo_bond_features(bond)
        edge_index[0].extend([i, j])
        edge_index[1].extend([j, i])
        edge_attr.extend([f, f])

    if len(edge_attr) == 0:
        edge_index = [[], []]
        edge_attr = []

    return [node_f, edge_index, edge_attr]


def mol_data_to_pyg(mol_data):
    x = torch.tensor(mol_data[0], dtype=torch.long)
    edge_index = torch.tensor(mol_data[1], dtype=torch.long)
    if len(mol_data[2]) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros((1, len(mol_data[2][0]) if mol_data[2] else 3), dtype=torch.long)
    else:
        edge_attr = torch.tensor(mol_data[2], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def stereo_add_global(graph):
    """Dynamic version of add_global supporting 8-dim atom and 4-dim bond tensors."""
    atom_dim = graph.x.size(1)
    edge_dim = graph.edge_attr.size(1)

    node = torch.zeros((1, atom_dim), dtype=torch.long)
    x = torch.cat([graph.x, node], dim=0)
    num_node = x.shape[0] - 1
    new_node = x.shape[0] - 1

    start = []
    end = []
    attr = []
    zero_edge = [0] * edge_dim

    for i in range(num_node):
        start.append(i)
        end.append(new_node)
        attr.append(zero_edge)
        start.append(new_node)
        end.append(i)
        attr.append(zero_edge)

    start = torch.tensor(start, dtype=torch.long).reshape(1, -1)
    end = torch.tensor(end, dtype=torch.long).reshape(1, -1)
    new_edge = torch.cat([start, end], dim=0)
    edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
    attr = torch.tensor(attr, dtype=torch.long)
    edge_attr = torch.cat([graph.edge_attr, attr], dim=0)

    if hasattr(graph, 'mol_type'):
        global_mol_type = torch.tensor([3], dtype=torch.long)
        new_mol_type = torch.cat([graph.mol_type, global_mol_type], dim=0)
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mol_type=new_mol_type)
    else:
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return g


class MockStereoGNN(torch.nn.Module):
    """Minimal end-to-end forward module with stereo embeddings and 2-layer GATv2 message passing."""
    def __init__(self, emb_dim=32):
        super().__init__()
        self.emb_dim = emb_dim
        # Base embeddings
        self.atom_embs = torch.nn.ModuleList([torch.nn.Embedding(120, emb_dim) for _ in range(7)])
        self.chiral_emb = torch.nn.Embedding(5, emb_dim)  # 8th atom dim: chiral tag

        self.bond_embs = torch.nn.ModuleList([torch.nn.Embedding(10, emb_dim) for _ in range(3)])
        self.stereo_emb = torch.nn.Embedding(10, emb_dim)  # 4th bond dim: stereo parity

        self.global_token = torch.nn.Parameter(torch.zeros(1, emb_dim))
        
        # 2-layer message passing: Layer 1 passes bond stereo to nodes; Layer 2 passes to global token
        self.conv1 = GATv2Conv(emb_dim, emb_dim, heads=1, concat=False, edge_dim=emb_dim)
        self.conv2 = GATv2Conv(emb_dim, emb_dim, heads=1, concat=False, edge_dim=emb_dim)
        self.act = torch.nn.ReLU()
        self.lin_out = torch.nn.Linear(emb_dim, 1)

    def forward(self, batch):
        h = torch.zeros(batch.x.shape[0], self.emb_dim, device=batch.x.device)
        is_normal = (batch.mol_type < 3)
        is_global = (batch.mol_type == 3)

        # Sum atom embeddings
        for i in range(7):
            h[is_normal] += self.atom_embs[i](batch.x[is_normal, i])
        h[is_normal] += self.chiral_emb(batch.x[is_normal, 7])
        h[is_global] = self.global_token

        # Sum bond embeddings (including 4th stereo parity dimension)
        edge_emb = torch.zeros(batch.edge_attr.shape[0], self.emb_dim, device=batch.edge_attr.device)
        for i in range(3):
            edge_emb += self.bond_embs[i](batch.edge_attr[:, i])
        edge_emb += self.stereo_emb(batch.edge_attr[:, 3])

        # Message Passing Layer 1
        x = self.act(self.conv1(h, batch.edge_index, edge_attr=edge_emb))
        # Message Passing Layer 2
        x = self.act(self.conv2(x, batch.edge_index, edge_attr=edge_emb))

        # Global node representation for prediction
        global_indices = torch.where(is_global)[0]
        out = self.lin_out(x[global_indices])
        return out, x, edge_emb


# ═══════════════════════════════════════════════════════════════
# Preflight Execution
# ═══════════════════════════════════════════════════════════════

def run_preflight():
    print("=" * 80)
    print("  STEP 24 PREFLIGHT: STEREOCHEMICAL FEATURIZATION AUDIT (R1336mzz E/Z)")
    print("=" * 80)

    stages = []

    # ─────────────────────────────────────────────────────────────
    # Stage P1: Input SMILES Stereochemistry Check
    # ─────────────────────────────────────────────────────────────
    print("\n>>> [P1] Verifying Stereochemical Information in Input SMILES...")
    mol_e = Chem.MolFromSmiles(SMILES_E)
    mol_z = Chem.MolFromSmiles(SMILES_Z)
    Chem.AssignStereochemistry(mol_e, cleanIt=True, force=True)
    Chem.AssignStereochemistry(mol_z, cleanIt=True, force=True)

    stereo_e_bonds = [int(b.GetStereo()) for b in mol_e.GetBonds() if b.GetBondTypeAsDouble() == 2.0]
    stereo_z_bonds = [int(b.GetStereo()) for b in mol_z.GetBonds() if b.GetBondTypeAsDouble() == 2.0]

    p1_pass = (stereo_e_bonds == [3] and stereo_z_bonds == [2])
    print(f"  R1336mzz(E) Double Bond Stereo Parity : {stereo_e_bonds} (expected [3] -> STEREOE)")
    print(f"  R1336mzz(Z) Double Bond Stereo Parity : {stereo_z_bonds} (expected [2] -> STEREOZ)")
    print(f"  [P1 Status] : {'PASS' if p1_pass else 'FAIL'}")

    stages.append({
        'stage': 'P1_SMILES_STEREO_INTEGRITY',
        'status': 'PASS' if p1_pass else 'FAIL',
        'detail': f"RDKit detected STEREOE (3) for trans and STEREOZ (2) for cis.",
        'metrics': {'E_stereo': stereo_e_bonds, 'Z_stereo': stereo_z_bonds}
    })

    # ─────────────────────────────────────────────────────────────
    # Stage P2: Baseline Featurizer Mathematical Degeneracy Check
    # ─────────────────────────────────────────────────────────────
    print("\n>>> [P2] Verifying Mathematical Degeneracy under Baseline Featurizer...")
    base_g_e = baseline_mol2graph(SMILES_E)
    base_g_z = baseline_mol2graph(SMILES_Z)

    x_e = np.array(base_g_e[0])
    x_z = np.array(base_g_z[0])
    e_e = np.array(base_g_e[1])
    e_z = np.array(base_g_z[1])
    a_e = np.array(base_g_e[2])
    a_z = np.array(base_g_z[2])

    diff_x = np.max(np.abs(x_e - x_z))
    diff_e = np.max(np.abs(e_e - e_z))
    diff_a = np.max(np.abs(a_e - a_z))

    p2_is_degenerate = (diff_x == 0 and diff_e == 0 and diff_a == 0)
    print(f"  Baseline Atom Matrix Max Diff  : {diff_x} (Shape: {x_e.shape})")
    print(f"  Baseline Edge Index Max Diff   : {diff_e} (Shape: {e_e.shape})")
    print(f"  Baseline Bond Feature Max Diff : {diff_a} (Shape: {a_e.shape})")
    print(f"  Mathematical Equivalence: G_E === G_Z -> {p2_is_degenerate}")
    print(f"  [P2 Status] : {'PASS' if p2_is_degenerate else 'FAIL'} (Confirms baseline representation blindness)")

    stages.append({
        'stage': 'P2_BASELINE_DEGENERACY_VERIFIED',
        'status': 'PASS' if p2_is_degenerate else 'FAIL',
        'detail': "Baseline featurizer produces mathematically identical graphs for R1336mzz(E) and R1336mzz(Z).",
        'metrics': {'diff_x': float(diff_x), 'diff_e': float(diff_e), 'diff_a': float(diff_a)}
    })

    # ─────────────────────────────────────────────────────────────
    # Stage P3: Stereo-Aware Featurizer Difference Check
    # ─────────────────────────────────────────────────────────────
    print("\n>>> [P3] Verifying Distinguishability under Stereo-Aware Featurizer...")
    stereo_g_e = stereo_mol2graph_components(SMILES_E)
    stereo_g_z = stereo_mol2graph_components(SMILES_Z)

    sx_e = np.array(stereo_g_e[0])
    sx_z = np.array(stereo_g_z[0])
    se_e = np.array(stereo_g_e[1])
    se_z = np.array(stereo_g_z[1])
    sa_e = np.array(stereo_g_e[2])
    sa_z = np.array(stereo_g_z[2])

    diff_sa = np.max(np.abs(sa_e - sa_z))
    p3_distinguishable = (diff_sa > 0)

    print(f"  Stereo-Aware Bond Feature Max Diff : {diff_sa} (sa_e[:, 3] vs sa_z[:, 3])")
    print(f"  Stereo Bond Parities (E) : {sa_e[:, 3].tolist()}")
    print(f"  Stereo Bond Parities (Z) : {sa_z[:, 3].tolist()}")
    print(f"  Mathematical Distinction: G_E_stereo !== G_Z_stereo -> {p3_distinguishable}")
    print(f"  [P3 Status] : {'PASS' if p3_distinguishable else 'FAIL'} (Confirms stereo-aware resolution)")

    stages.append({
        'stage': 'P3_STEREO_DISTINCTION_VERIFIED',
        'status': 'PASS' if p3_distinguishable else 'FAIL',
        'detail': f"Stereo-aware featurizer resolves E/Z via bond stereo dimension with max diff {diff_sa}.",
        'metrics': {'max_bond_diff': float(diff_sa), 'E_bonds': sa_e[:, 3].tolist(), 'Z_bonds': sa_z[:, 3].tolist()}
    })

    # ─────────────────────────────────────────────────────────────
    # Stage P4: Unique-Source-of-Difference Diff Report
    # ─────────────────────────────────────────────────────────────
    print("\n>>> [P4] Isolated Diff Report: Verifying Single-Source Stereo Difference...")
    # Check 1: Non-stereo atom dimensions (indices 0..6)
    non_stereo_atom_diff = np.max(np.abs(sx_e[:, :7] - sx_z[:, :7]))
    # Check 2: Non-stereo bond dimensions (indices 0..2)
    non_stereo_bond_diff = np.max(np.abs(sa_e[:, :3] - sa_z[:, :3]))
    # Check 3: Edge index identical
    edge_index_diff = np.max(np.abs(se_e - se_z))
    # Check 4: Graph topology shapes identical
    shapes_identical = (sx_e.shape == sx_z.shape and sa_e.shape == sa_z.shape and se_e.shape == se_z.shape)

    # Check 5: Only bond stereo dimension (index 3) differs
    bond_stereo_diff_locations = np.where(sa_e[:, 3] != sa_z[:, 3])[0]
    only_stereo_differs = (
        non_stereo_atom_diff == 0 and
        non_stereo_bond_diff == 0 and
        edge_index_diff == 0 and
        shapes_identical and
        len(bond_stereo_diff_locations) == 2  # The 2 directed edges corresponding to the C=C double bond
    )

    print(f"  Diff in non-stereo atom features (0..6) : {non_stereo_atom_diff}")
    print(f"  Diff in non-stereo bond features (0..2) : {non_stereo_bond_diff}")
    print(f"  Diff in edge indices (topology walk)    : {edge_index_diff}")
    print(f"  Number of differing bond entries       : {len(bond_stereo_diff_locations)} / {len(sa_e)} (directed C=C edges)")
    print(f"  Single-Source Difference Confirmed      : {only_stereo_differs}")
    print(f"  [P4 Status] : {'PASS' if only_stereo_differs else 'FAIL'}")

    stages.append({
        'stage': 'P4_ISOLATED_DIFF_REPORT',
        'status': 'PASS' if only_stereo_differs else 'FAIL',
        'detail': "Except for the 4th bond stereo dimension, all atom/bond features and graph topologies are 100% identical.",
        'metrics': {
            'non_stereo_atom_diff': float(non_stereo_atom_diff),
            'non_stereo_bond_diff': float(non_stereo_bond_diff),
            'edge_index_diff': float(edge_index_diff),
            'differing_edges': int(len(bond_stereo_diff_locations))
        }
    })

    # ─────────────────────────────────────────────────────────────
    # Stage P5: Batch-Level Pipeline & Model Input Preservation
    # ─────────────────────────────────────────────────────────────
    print("\n>>> [P5] Verifying Stereo Survival Through Full Graph Pipeline (Tri-graph + Global + Batch + Forward)...")
    c_smi = lookup_smiles('emim')
    a_smi = lookup_smiles('Tf2N')
    cg = mol_data_to_pyg(stereo_mol2graph_components(c_smi))
    ag = mol_data_to_pyg(stereo_mol2graph_components(a_smi))
    rg_e = mol_data_to_pyg(stereo_g_e)
    rg_z = mol_data_to_pyg(stereo_g_z)

    comp_e = stereo_add_global(combine_Graph([cg, ag, rg_e]))
    comp_z = stereo_add_global(combine_Graph([cg, ag, rg_z]))

    batch_e = Batch.from_data_list([comp_e])
    batch_z = Batch.from_data_list([comp_z])

    diff_batch_edge_attr = torch.max(torch.abs(batch_e.edge_attr - batch_z.edge_attr)).item()

    # Pass through MockStereoGNN to verify distinct forward pass activations
    torch.manual_seed(42)
    mock_model = MockStereoGNN(emb_dim=32)
    mock_model.eval()
    with torch.no_grad():
        out_e, h_e, edge_emb_e = mock_model(batch_e)
        out_z, h_z, edge_emb_z = mock_model(batch_z)

    diff_out = torch.max(torch.abs(out_e - out_z)).item()
    diff_h = torch.max(torch.abs(h_e - h_z)).item()
    diff_edge_emb = torch.max(torch.abs(edge_emb_e - edge_emb_z)).item()

    p5_pass = (diff_batch_edge_attr > 0 and diff_edge_emb > 0 and diff_out > 0)
    print(f"  Batch edge_attr max diff           : {diff_batch_edge_attr}")
    print(f"  Model edge embeddings max diff     : {diff_edge_emb}")
    print(f"  Model node embeddings max diff     : {diff_h}")
    print(f"  Model output prediction max diff   : {diff_out}")
    print(f"  Stereo Information Preservation    : {p5_pass}")
    print(f"  [P5 Status] : {'PASS' if p5_pass else 'FAIL'}")

    stages.append({
        'stage': 'P5_BATCH_PIPELINE_PRESERVATION',
        'status': 'PASS' if p5_pass else 'FAIL',
        'detail': "Stereo information survives tri-graph combination, global-node addition, PyG Batch collation, and yields distinct forward model activations.",
        'metrics': {
            'batch_edge_attr_diff': float(diff_batch_edge_attr),
            'edge_emb_diff': float(diff_edge_emb),
            'model_output_diff': float(diff_out),
            'tri_graph_nodes': comp_e.x.shape[0],
            'tri_graph_edges': comp_e.edge_attr.shape[0]
        }
    })

    # ─────────────────────────────────────────────────────────────
    # Save Reports
    # ─────────────────────────────────────────────────────────────
    all_passed = all(s['status'] == 'PASS' for s in stages)
    ts = datetime.now().isoformat()

    json_path = PAPER / "audit_step24_preflight_stereo.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            'audit_timestamp': ts,
            'all_stages_passed': all_passed,
            'summary': {s['stage']: s['status'] for s in stages},
            'stages': stages
        }, f, indent=2, ensure_ascii=False)

    csv_path = PAPER / "audit_step24_preflight_stereo.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Stage ID", "Check Description", "Status", "Detail"])
        for s in stages:
            w.writerow([s['stage'], s['stage'].replace("_", " "), s['status'], s['detail']])

    print("\n" + "=" * 80)
    print("  STEP 24 PREFLIGHT SUMMARY REPORT")
    print("=" * 80)
    for s in stages:
        mark = "✅ PASS" if s['status'] == 'PASS' else "❌ FAIL"
        print(f"  {mark:<8} | {s['stage']:<35} | {s['detail']}")
    print("-" * 80)
    print(f"  Overall Preflight Verdict: {'ALL 5 STAGES PASSED! READY FOR STEREO ABLATION.' if all_passed else 'PREFLIGHT FAILED'}")
    print(f"  Saved:")
    print(f"   - {json_path}")
    print(f"   - {csv_path}")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_preflight())
