import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool, GlobalAttention

num_atom_type = 119 
num_Hbrid = 8
num_Aro = 2
num_degree = 7
num_charge = 3
num_eneg   = 8
num_radius = 8

# [Round 2 物理增强] 化学键特征 Embedding 维度定义
num_bond_type = 5        # 0: global, 1: single, 2: double, 3: triple, 4: aromatic
num_bond_isInRing = 2    # 0: no, 1: yes
num_bond_isAromatic = 2  # 0: no, 1: yes

class IL_GAT_v6(torch.nn.Module):
    def __init__(self, args):
        super(IL_GAT_v6, self).__init__()
        self.args = args
        self.emb_dim = args.get('emb_dim', 300)
        self.pool_type = args.get('pool', 'global')
        
        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_Hbrid, self.emb_dim)
        self.x_embedding3 = nn.Embedding(num_Aro, self.emb_dim)
        self.x_embedding4 = nn.Embedding(num_degree, self.emb_dim)
        self.x_embedding5 = nn.Embedding(num_charge, self.emb_dim)
        self.x_embedding6 = nn.Embedding(num_eneg,   self.emb_dim)
        self.x_embedding7 = nn.Embedding(num_radius, self.emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        nn.init.xavier_uniform_(self.x_embedding7.weight.data)

        # [Round 2 物理增强] 化学键 Embedding 初始化
        self.edge_embedding1 = nn.Embedding(num_bond_type, self.emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_isInRing, self.emb_dim)
        self.edge_embedding3 = nn.Embedding(num_bond_isAromatic, self.emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)

        # [修复 1] 增加 Molecule Type Embedding (0: Cation, 1: Anion, 2: Refrigerant)
        self.mol_embedding = nn.Embedding(3, self.emb_dim)
        nn.init.xavier_uniform_(self.mol_embedding.weight.data)
        
        # [修复 1 - 续] Global Token 独立于普通分子的 Embedding 空间
        self.global_token = nn.Parameter(torch.zeros(1, self.emb_dim))

        # [Round 2 物理增强] GATv2Conv 开启 edge_dim
        self.l1 = GATv2Conv(self.emb_dim, 512, heads=4, concat=False, edge_dim=self.emb_dim)
        self.l2 = GATv2Conv(512, 1024, heads=4, concat=False, edge_dim=self.emb_dim)
        self.l3 = GATv2Conv(1024, 512, heads=4, concat=False, edge_dim=self.emb_dim)

        # [消融控制] 如果用 attention pool，需要初始化 GlobalAttention
        if self.pool_type == 'attention':
            self.att_pool = GlobalAttention(nn.Sequential(nn.Linear(512, 1)))

        # v6 改动：cond_dim 不再硬编码，而是由 Dataset_v6 根据 descriptor_mode 动态计算
        # 支持 M0(7维), Msize(9维), Mmu(8维), Mphys(10维) 四种消融模式
        cond_dim = args['cond_dim']  # Dynamically set by Dataset_v6 based on descriptor_mode
        
        # [FIX v2] 物理特征提权：先把 cond_dim 映射到 64 维，防止被 512 维图特征淹没
        # 同时不过度放大（128→64），避免物理噪声主导
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(),
        )

        # [FIX v2] 恢复 MLP 容量（1024→512→1），不缩减网络宽度
        self.l5 = nn.Sequential(
            nn.Linear(512 + 64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.15),

            nn.Linear(512, 1)
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=args['dropout_rate'])

    def extract(self, x, data_i):
        """提取全局节点（每个子图的最后一个节点）"""
        batch = data_i.batch
        output, count = torch.unique(batch, return_counts=True)
        count = count.tolist()

        l = []
        cur = 0
        for i in count:
            cur += i
            l.append(cur)
        re = []
        for j in l:
            # [安全断言] 确保取到的图最后一个节点确实是 Global Token (mol_type == 3)
            if hasattr(data_i, 'mol_type'):
                assert data_i.mol_type[j - 1] == 3, f"Critical Error: Last node is not global node, got mol_type {data_i.mol_type[j - 1]}"
            re.append(x[j - 1].reshape(1, -1))
        return torch.cat(re, dim=0)

    def forward(self, data_i, cond):
        h = torch.zeros(data_i.x.shape[0], self.emb_dim, device=data_i.x.device)
        
        # ====================================================
        # 【核心修正】 Heterogeneous Component-Aware Embedding
        # 1. 真实原子：物理 Embedding + Component Identity (0,1,2)
        # 2. 全局节点：完全跳过物理 Embedding，独享 Global Token
        # ====================================================
        if hasattr(data_i, 'mol_type'):
            mol_type = data_i.mol_type
            is_normal = (mol_type < 3)
            is_global = (mol_type == 3)
            
            # 真实原子的特征映射
            h[is_normal] = self.x_embedding1(data_i.x[is_normal, 0]) + \
                           self.x_embedding2(data_i.x[is_normal, 1]) + \
                           self.x_embedding3(data_i.x[is_normal, 2]) + \
                           self.x_embedding4(data_i.x[is_normal, 3]) + \
                           self.x_embedding5(data_i.x[is_normal, 4]) + \
                           self.x_embedding6(data_i.x[is_normal, 5]) + \
                           self.x_embedding7(data_i.x[is_normal, 6])
                           
            # 为阴阳离子和气体加入专门的身份标识
            if not self.args.get('no_mol_embedding', False):
                h[is_normal] = h[is_normal] + self.mol_embedding(mol_type[is_normal])
                
            # 为全局节点加入独立的高维标识 Token
            h[is_global] = self.global_token
        else:
            # 兼容非 v5 数据集的 fallback 逻辑
            h = self.x_embedding1(data_i.x[:, 0]) + \
                self.x_embedding2(data_i.x[:, 1]) + \
                self.x_embedding3(data_i.x[:, 2]) + \
                self.x_embedding4(data_i.x[:, 3]) + \
                self.x_embedding5(data_i.x[:, 4]) + \
                self.x_embedding6(data_i.x[:, 5]) + \
                self.x_embedding7(data_i.x[:, 6])
        # ====================================================

        x, edge_index = h, data_i.edge_index

        # [Round 2 物理增强] 嵌入化学键离散特征 (单键/双键/三键/芳香键 + 环 + 芳香)
        edge_emb = self.edge_embedding1(data_i.edge_attr[:, 0]) + \
                   self.edge_embedding2(data_i.edge_attr[:, 1]) + \
                   self.edge_embedding3(data_i.edge_attr[:, 2])

        x, _ = self.l1(x, edge_index, edge_attr=edge_emb, return_attention_weights=True)
        x = self.act(x)
        x = self.dropout(x)

        x, _ = self.l2(x, edge_index, edge_attr=edge_emb, return_attention_weights=True)
        x = self.act(x)
        x = self.dropout(x)

        x, _ = self.l3(x, edge_index, edge_attr=edge_emb, return_attention_weights=True)
        x = self.act(x)
        x = self.dropout(x)

        # ====================================================
        # 【池化消融】支持 Global Node / Mean / Attention Pool
        # ====================================================
        if self.pool_type == 'global':
            x_g = self.extract(x, data_i)
        elif self.pool_type == 'mean':
            # Mean pool 时我们需要移除全局节点，否则会被污染
            # 但简单实现可以先直接 mean_pool 整体
            if hasattr(data_i, 'mol_type'):
                normal_mask = (data_i.mol_type < 3)
                x_g = global_mean_pool(x[normal_mask], data_i.batch[normal_mask])
            else:
                x_g = global_mean_pool(x, data_i.batch)
        elif self.pool_type == 'attention':
            if hasattr(data_i, 'mol_type'):
                normal_mask = (data_i.mol_type < 3)
                x_g = self.att_pool(x[normal_mask], data_i.batch[normal_mask])
            else:
                x_g = self.att_pool(x, data_i.batch)
        else:
            raise ValueError(f"Unknown pool type {self.pool_type}")

        cond_emb = self.cond_mlp(cond)
        x_concat = torch.cat([x_g, cond_emb], dim=1)
        x_out = self.l5(x_concat)

        return x_out


# ============================================================
# [Ablation] GCN Baseline — Same architecture, no attention, no edge features
# Purpose: Isolate the contribution of graph attention + edge features
# ============================================================
class IL_GCN_v6(torch.nn.Module):
    """
    Fair ablation baseline for IL_GAT_v6.
    Identical: atom embeddings, mol_embedding, global token, pooling, MLP head.
    Different: GCNConv (no attention mechanism, no edge features).
    """
    def __init__(self, args):
        super(IL_GCN_v6, self).__init__()
        self.args = args
        self.emb_dim = args.get('emb_dim', 300)
        self.pool_type = args.get('pool', 'global')

        # Atom Embeddings (identical to GAT)
        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_Hbrid, self.emb_dim)
        self.x_embedding3 = nn.Embedding(num_Aro, self.emb_dim)
        self.x_embedding4 = nn.Embedding(num_degree, self.emb_dim)
        self.x_embedding5 = nn.Embedding(num_charge, self.emb_dim)
        self.x_embedding6 = nn.Embedding(num_eneg,   self.emb_dim)
        self.x_embedding7 = nn.Embedding(num_radius, self.emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        nn.init.xavier_uniform_(self.x_embedding7.weight.data)

        # Molecule Type Embedding (identical to GAT)
        self.mol_embedding = nn.Embedding(3, self.emb_dim)
        nn.init.xavier_uniform_(self.mol_embedding.weight.data)

        # Global Token (identical to GAT)
        self.global_token = nn.Parameter(torch.zeros(1, self.emb_dim))

        # GCNConv layers (NO edge features, NO attention)
        self.l1 = GCNConv(self.emb_dim, 512)
        self.l2 = GCNConv(512, 1024)
        self.l3 = GCNConv(1024, 512)

        # Condition dimension (identical to GAT)
        # v6 改动：cond_dim 不再硬编码，而是由 Dataset_v6 根据 descriptor_mode 动态计算
        # 支持 M0(7维), Msize(9维), Mmu(8维), Mphys(10维) 四种消融模式
        cond_dim = args['cond_dim']  # Dynamically set by Dataset_v6 based on descriptor_mode

        # MLP head (identical to GAT)
        # [FIX v2] 物理特征提权：先把 cond_dim 映射到 64 维
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(),
        )

        # [FIX v2] 恢复 MLP 容量（1024→512→1）
        self.l5 = nn.Sequential(
            nn.Linear(512 + 64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.15),

            nn.Linear(512, 1)
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=args['dropout_rate'])

    def extract(self, x, data_i):
        """Extract global node embeddings (identical to GAT)"""
        batch = data_i.batch
        output, count = torch.unique(batch, return_counts=True)
        count = count.tolist()

        l = []
        cur = 0
        for i in count:
            cur += i
            l.append(cur)
        re = []
        for j in l:
            if hasattr(data_i, 'mol_type'):
                assert data_i.mol_type[j - 1] == 3, \
                    f"Critical Error: Last node is not global node, got mol_type {data_i.mol_type[j - 1]}"
            re.append(x[j - 1].reshape(1, -1))
        return torch.cat(re, dim=0)

    def forward(self, data_i, cond):
        h = torch.zeros(data_i.x.shape[0], self.emb_dim, device=data_i.x.device)

        # Heterogeneous Embedding (identical to GAT)
        if hasattr(data_i, 'mol_type'):
            mol_type = data_i.mol_type
            is_normal = (mol_type < 3)
            is_global = (mol_type == 3)

            h[is_normal] = self.x_embedding1(data_i.x[is_normal, 0]) + \
                           self.x_embedding2(data_i.x[is_normal, 1]) + \
                           self.x_embedding3(data_i.x[is_normal, 2]) + \
                           self.x_embedding4(data_i.x[is_normal, 3]) + \
                           self.x_embedding5(data_i.x[is_normal, 4]) + \
                           self.x_embedding6(data_i.x[is_normal, 5]) + \
                           self.x_embedding7(data_i.x[is_normal, 6])

            if not self.args.get('no_mol_embedding', False):
                h[is_normal] = h[is_normal] + self.mol_embedding(mol_type[is_normal])

            h[is_global] = self.global_token
        else:
            h = self.x_embedding1(data_i.x[:, 0]) + \
                self.x_embedding2(data_i.x[:, 1]) + \
                self.x_embedding3(data_i.x[:, 2]) + \
                self.x_embedding4(data_i.x[:, 3]) + \
                self.x_embedding5(data_i.x[:, 4]) + \
                self.x_embedding6(data_i.x[:, 5]) + \
                self.x_embedding7(data_i.x[:, 6])

        x, edge_index = h, data_i.edge_index

        # GCNConv message passing (NO edge features, NO attention)
        x = self.l1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)

        x = self.l2(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)

        x = self.l3(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)

        # Pooling (identical to GAT)
        if self.pool_type == 'global':
            x_g = self.extract(x, data_i)
        elif self.pool_type == 'mean':
            if hasattr(data_i, 'mol_type'):
                normal_mask = (data_i.mol_type < 3)
                x_g = global_mean_pool(x[normal_mask], data_i.batch[normal_mask])
            else:
                x_g = global_mean_pool(x, data_i.batch)
        else:
            raise ValueError(f"Unknown pool type {self.pool_type}")

        cond_emb = self.cond_mlp(cond)
        x_concat = torch.cat([x_g, cond_emb], dim=1)
        x_out = self.l5(x_concat)

        return x_out
