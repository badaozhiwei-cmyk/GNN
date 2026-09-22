"""
Model_v7.py — SolvGNN-Aligned Three-Component Interaction Architecture
======================================================================
Scientific Hypothesis:
Replacing V6's single global-token fusion bottleneck (RANGE, Nat. Commun. 2026)
with an explicit 3-node molecular interaction graph (SolvGNN, Digit. Discov. 2023)
coupled with shared ionic-liquid encoding (Rittig et al., arXiv:2206.11776)
and descriptor modality dropout (ModDrop, arXiv:1501.00102; MEGNet, Chem. Mater. 2019)
tests whether preserving component identity and intermolecular message passing
improves graph representation reliance and alleviates scalar bypass.
"""

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

num_bond_type = 5        # 0: global, 1: single, 2: double, 3: triple, 4: aromatic
num_bond_isInRing = 2    # 0: no, 1: yes
num_bond_isAromatic = 2  # 0: no, 1: yes
num_bond_stereo = 6      # 0: None, 1: Any, 2: Z, 3: E, 4: Cis, 5: Trans


class IL_GAT_v7(nn.Module):
    def __init__(self, args=None):
        super(IL_GAT_v7, self).__init__()
        self.args = args or {}
        self.emb_dim = self.args.get('emb_dim', 300)
        self.hidden_dim = self.args.get('hidden_dim', 512)
        self.desc_dropout_p = self.args.get('desc_dropout_p', 0.0)  # 0.0 for V7-A, 0.3 for V7-B
        self.use_sigmoid = self.args.get('use_sigmoid', False)

        # ── 1. Shared Atom & Bond Embeddings ──
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

        # ── 2. Intramolecular Local GNN Encoders ──
        # (A) Shared IL Encoder (Cation & Anion, Rittig et al. 2023)
        self.il_l1 = GATv2Conv(self.emb_dim, 512, heads=4, concat=False, edge_dim=self.emb_dim)
        self.il_l2 = GATv2Conv(512, 1024, heads=4, concat=False, edge_dim=self.emb_dim)
        self.il_l3 = GATv2Conv(1024, 512, heads=4, concat=False, edge_dim=self.emb_dim)

        # (B) Separate Solute Encoder (Refrigerant)
        self.ref_l1 = GATv2Conv(self.emb_dim, 512, heads=4, concat=False, edge_dim=self.emb_dim)
        self.ref_l2 = GATv2Conv(512, 1024, heads=4, concat=False, edge_dim=self.emb_dim)
        self.ref_l3 = GATv2Conv(1024, 512, heads=4, concat=False, edge_dim=self.emb_dim)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=self.args.get('dropout_rate', 0.2))

        # ── 3. SolvGNN-Style 3-Node Molecular Interaction Graph ──
        # Nodes: 0: Cation, 1: Anion, 2: Refri
        # Edges: 3 pair types: 0: C-A, 1: C-R, 2: A-R
        self.inter_edge_embed = nn.Embedding(3, self.hidden_dim)
        self.inter_conv = GATv2Conv(self.hidden_dim, self.hidden_dim, heads=4, concat=False, edge_dim=self.hidden_dim)
        self.inter_norm = nn.LayerNorm(self.hidden_dim)

        # ── 4. System Readout Projection ──
        # Concatenation of updated [h'_cat, h'_ani, h'_ref] = 512 * 3 = 1536 -> 512
        self.sys_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        # ── 5. Prediction Head ──
        # Input: h_sys (512) + State (2: T, P) + Descriptors (7) = 521
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
        
        # Inject component identity bias
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

    def _build_interaction_graph(self, h_cat, h_ani, h_ref):
        """
        Constructs a batched 3-node molecular interaction graph (SolvGNN style).
        Each sample has 3 nodes: Cat (0), Ani (1), Ref (2).
        Bidirectional edges:
          0 <-> 1: C-A (type 0)
          0 <-> 2: C-R (type 1)
          1 <-> 2: A-R (type 2)
        """
        B = h_cat.size(0)
        device = h_cat.device

        # Stack into (B, 3, 512) -> flatten to (3B, 512)
        h_nodes = torch.stack([h_cat, h_ani, h_ref], dim=1).view(B * 3, self.hidden_dim)

        # Base 6 directed edges per sample
        base_src = torch.tensor([0, 1, 0, 2, 1, 2], dtype=torch.long, device=device)
        base_dst = torch.tensor([1, 0, 2, 0, 2, 1], dtype=torch.long, device=device)
        base_type = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)

        offsets = (torch.arange(B, device=device) * 3).unsqueeze(1)  # (B, 1)
        src = (base_src.unsqueeze(0) + offsets).view(-1)
        dst = (base_dst.unsqueeze(0) + offsets).view(-1)
        edge_types = base_type.repeat(B)

        inter_edge_index = torch.stack([src, dst], dim=0)
        inter_edge_attr = self.inter_edge_embed(edge_types)

        # Message passing over molecular interaction graph
        h_updated = self.inter_conv(h_nodes, inter_edge_index, edge_attr=inter_edge_attr)
        h_updated = self.inter_norm(h_nodes + self.act(h_updated))

        # Unflatten back to (B, 3, 512)
        h_mol_updated = h_updated.view(B, 3, self.hidden_dim)
        h_cat_prime = h_mol_updated[:, 0, :]
        h_ani_prime = h_mol_updated[:, 1, :]
        h_ref_prime = h_mol_updated[:, 2, :]

        return h_cat_prime, h_ani_prime, h_ref_prime

    def forward(self, batch_data, override_mask_desc=None, override_zero_graph=False):
        """
        batch_data: dict with 'cat', 'ani', 'ref', 'state', 'desc'
        override_mask_desc: float/tensor for descriptor ablation (e.g. 0.0)
        override_zero_graph: bool, if True zeros out h_sys to measure delta_y_graph
        """
        g_cat = batch_data['cat']
        g_ani = batch_data['ani']
        g_ref = batch_data['ref']
        state = batch_data['state']  # [B, 2] (T, P)
        desc  = batch_data['desc']   # [B, 7] (MW, Charge, LogP...)

        # 1. Molecular Encodings
        h_cat = self._encode_il(g_cat, mol_type_idx=0)
        h_ani = self._encode_il(g_ani, mol_type_idx=1)
        h_ref = self._encode_ref(g_ref)

        # 2. Intermolecular Interaction MPNN
        h_cat_p, h_ani_p, h_ref_p = self._build_interaction_graph(h_cat, h_ani, h_ref)

        # 3. System Readout
        h_concat = torch.cat([h_cat_p, h_ani_p, h_ref_p], dim=-1)
        h_sys = self.sys_proj(h_concat)
        if override_zero_graph:
            h_sys = torch.zeros_like(h_sys)

        # 4. Modality-Level Descriptor Dropout (ModDrop)
        if self.training and self.desc_dropout_p > 0.0:
            B = desc.size(0)
            mask = (torch.rand(B, 1, device=desc.device) >= self.desc_dropout_p).float()
            desc = desc * mask
        elif override_mask_desc is not None:
            # Used for diagnostic ablation / perturbation tests
            desc = desc * override_mask_desc

        # 5. Fusion: System Graph Embedding + State (100% Retained) + Descriptors
        fused = torch.cat([h_sys, state, desc], dim=-1)

        # 6. Prediction Head
        out = self.head(fused).squeeze(-1)
        if self.use_sigmoid:
            out = torch.sigmoid(out)

        return out
