"""
Model_v7_concat_mlp.py — Capacity-Matched Concat+MLP Baseline (Model B for P1)
==============================================================================
Rigorous controlled ablation baseline for Canonical V7 (Model A: 3-Node Interaction GAT).

Architectural Hypothesis & Control Design:
1. Replaces the 3-Node molecular interaction graph (inter_edge_embed, inter_conv, inter_norm)
   and system projection (sys_proj) [together 3,942,912 parameters] with an explicit 3-layer MLP.
2. Direct concatenation of pooled component representations: [h_cat, h_ani, h_ref] in R^1536.
3. Capacity matching:
   - Linear(1536, 1378) + ReLU + Dropout(p=0.1) [2,117,986 params]
   - Linear(1378, 960)  + ReLU + Dropout(p=0.1) [1,323,840 params]
   - Linear(960, 512)   + LayerNorm(512) + ReLU + Dropout(p=0.1) [493,056 params]
   - Total replacement parameters = 3,934,882 (-0.204% vs 3,942,912)
   - Total model trainable parameters = 29,252,959 (-0.0274% vs 29,260,989)
4. Frozen regularization:
   - Activation: nn.ReLU()
   - Layer normalization: LayerNorm(512) at the 512-dim bottleneck
   - Dropout probability: p=0.1 (strictly aligned with sys_proj)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

num_atom_type = 119
num_Hbrid = 8
num_Aro = 2
num_degree = 7
num_charge = 3
num_eneg = 8
num_radius = 8

num_bond_type = 5
num_bond_isInRing = 2
num_bond_isAromatic = 2
num_bond_stereo = 6


class IL_ConcatMLP_v7(nn.Module):
    def __init__(self, args=None):
        super(IL_ConcatMLP_v7, self).__init__()
        self.args = args or {}
        self.emb_dim = self.args.get('emb_dim', 300)
        self.hidden_dim = self.args.get('hidden_dim', 512)
        self.desc_dropout_p = self.args.get('desc_dropout_p', 0.0)  # Phase 1 controlled at 0.0
        self.use_sigmoid = self.args.get('use_sigmoid', False)

        # ── 1. Shared Atom & Bond Embeddings (Identical to Model A) ──
        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_Hbrid, self.emb_dim)
        self.x_embedding3 = nn.Embedding(num_Aro, self.emb_dim)
        self.x_embedding4 = nn.Embedding(num_degree, self.emb_dim)
        self.x_embedding5 = nn.Embedding(num_charge, self.emb_dim)
        self.x_embedding6 = nn.Embedding(num_eneg, self.emb_dim)
        self.x_embedding7 = nn.Embedding(num_radius, self.emb_dim)

        self.edge_embedding1 = nn.Embedding(num_bond_type, self.emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_isInRing, self.emb_dim)
        self.edge_embedding3 = nn.Embedding(num_bond_isAromatic, self.emb_dim)
        self.edge_embedding4 = nn.Embedding(num_bond_stereo, self.emb_dim)

        # Component Type Identity: 0: Cation, 1: Anion, 2: Refrigerant
        self.mol_embedding = nn.Embedding(3, self.emb_dim)

        # ── 2. Intramolecular Local GNN Encoders (Identical to Model A) ──
        # (A) Shared IL Encoder (Cation & Anion)
        self.il_l1 = GATv2Conv(self.emb_dim, 512, heads=4, concat=False, edge_dim=self.emb_dim)
        self.il_l2 = GATv2Conv(512, 1024, heads=4, concat=False, edge_dim=self.emb_dim)
        self.il_l3 = GATv2Conv(1024, 512, heads=4, concat=False, edge_dim=self.emb_dim)

        # (B) Separate Solute Encoder (Refrigerant)
        self.ref_l1 = GATv2Conv(self.emb_dim, 512, heads=4, concat=False, edge_dim=self.emb_dim)
        self.ref_l2 = GATv2Conv(512, 1024, heads=4, concat=False, edge_dim=self.emb_dim)
        self.ref_l3 = GATv2Conv(1024, 512, heads=4, concat=False, edge_dim=self.emb_dim)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=self.args.get('dropout_rate', 0.2))

        # ── 3. Capacity-Matched Fusion MLP (Replacing 3-Node Interaction GAT + sys_proj) ──
        # Input: [h_cat, h_ani, h_ref] concatenated = 512 * 3 = 1536 -> 512
        # Exact parameter count: 3,934,882 (matches 3,942,912 within -0.204%)
        self.sys_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, 1378),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1378, 960),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(960, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        # ── 4. Prediction Head (Identical to Model A) ──
        head_in_dim = self.hidden_dim + 2 + 7
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 1)
        )

    def _embed_graph(self, data, mol_type_idx):
        h = self.x_embedding1(data.x[:, 0]) + \
            self.x_embedding2(data.x[:, 1]) + \
            self.x_embedding3(data.x[:, 2]) + \
            self.x_embedding4(data.x[:, 3]) + \
            self.x_embedding5(data.x[:, 4]) + \
            self.x_embedding6(data.x[:, 5]) + \
            self.x_embedding7(data.x[:, 6])

        h = h + self.mol_embedding(torch.tensor(mol_type_idx, device=data.x.device))

        e = self.edge_embedding1(data.edge_attr[:, 0]) + \
            self.edge_embedding2(data.edge_attr[:, 1]) + \
            self.edge_embedding3(data.edge_attr[:, 2])
        if data.edge_attr.size(1) >= 4:
            e = e + self.edge_embedding4(data.edge_attr[:, 3])

        return h, e

    def _encode_il(self, data, mol_type_idx):
        h, e = self._embed_graph(data, mol_type_idx)
        x = self.il_l1(h, data.edge_index, edge_attr=e)
        x = self.dropout(self.act(x))
        x = self.il_l2(x, data.edge_index, edge_attr=e)
        x = self.dropout(self.act(x))
        x = self.il_l3(x, data.edge_index, edge_attr=e)
        x = self.dropout(self.act(x))
        return global_mean_pool(x, data.batch)

    def _encode_ref(self, data):
        h, e = self._embed_graph(data, mol_type_idx=2)
        x = self.ref_l1(h, data.edge_index, edge_attr=e)
        x = self.dropout(self.act(x))
        x = self.ref_l2(x, data.edge_index, edge_attr=e)
        x = self.dropout(self.act(x))
        x = self.ref_l3(x, data.edge_index, edge_attr=e)
        x = self.dropout(self.act(x))
        return global_mean_pool(x, data.batch)

    def forward(self, batch_data, override_mask_desc=None, override_zero_graph=False):
        g_cat = batch_data['cat']
        g_ani = batch_data['ani']
        g_ref = batch_data['ref']
        state = batch_data['state']  # [B, 2] (T, P)
        desc  = batch_data['desc']   # [B, 7]

        # 1. Molecular Encodings
        h_cat = self._encode_il(g_cat, mol_type_idx=0)
        h_ani = self._encode_il(g_ani, mol_type_idx=1)
        h_ref = self._encode_ref(g_ref)

        # 2. Direct Component Concatenation [B, 1536]
        h_concat = torch.cat([h_cat, h_ani, h_ref], dim=-1)

        # 3. System Projection via Capacity-Matched MLP [B, 512]
        h_sys = self.sys_mlp(h_concat)
        if override_zero_graph:
            h_sys = torch.zeros_like(h_sys)

        # 4. Modality-Level Descriptor Dropout
        if self.training and self.desc_dropout_p > 0.0:
            B = desc.size(0)
            mask = (torch.rand(B, 1, device=desc.device) >= self.desc_dropout_p).float()
            desc = desc * mask
        elif override_mask_desc is not None:
            desc = desc * override_mask_desc

        # 5. Fusion
        fused = torch.cat([h_sys, state, desc], dim=-1)

        # 6. Prediction Head
        out = self.head(fused).squeeze(-1)
        if self.use_sigmoid:
            out = torch.sigmoid(out)

        return out
