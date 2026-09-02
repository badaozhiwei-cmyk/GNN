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

        # v6 动态 cond_dim（支持 M0:7, Msize:9, Mmu:8, Mphys:10）
        cond_dim = args['cond_dim']

        # ══════════════════════════════════════════════════════
        # 【第一招】组件级交互池化开关（默认关闭，向下兼容祖宗之法）
        # ══════════════════════════════════════════════════════
        self.use_interaction = args.get('use_interaction', False)

        # ══════════════════════════════════════════════════════
        # 【第二招】低成本修复三件套开关（默认全部关闭）
        # ══════════════════════════════════════════════════════
        self.use_sigmoid = args.get('use_sigmoid', False)
        self.use_cond_dropout = args.get('use_cond_dropout', False)
        self.cond_dropout_p = args.get('cond_dropout_p', 0.3)
        self.use_layernorm = args.get('use_layernorm', False)

        # ══════════════════════════════════════════════════════
        # 【第三招】自适应门控物理描述符注入
        # (Adaptive Gated Physicochemical Descriptor Injection)
        # 科学动机：LORO 实验发现物理描述符的有效性具有强烈的
        # 制冷剂依赖性（R23: 巨幅改善, R32/R41: 明显恶化），
        # 简单拼接（static concatenation）不是最优策略。
        # 门控让模型学习根据分子图嵌入，对物理描述符进行
        # 自适应加权：g = σ(W · x_g + b)
        # ══════════════════════════════════════════════════════
        self.use_adaptive_gate = args.get('use_adaptive_gate', False)
        if self.use_adaptive_gate:
            self.n_base_features = args.get('n_base_features', 9)
            n_phys = cond_dim - self.n_base_features
            assert n_phys > 0, (
                f"Adaptive gate requires physics features! "
                f"cond_dim={cond_dim}, n_base={self.n_base_features}"
            )
            self.n_phys_features = n_phys
            # 极简门控：单层线性 + Sigmoid
            # 3 个独立标量门分别控制 μ, α, V（或更多物理描述符）
            self.gate_linear = nn.Linear(512, n_phys)
            nn.init.xavier_uniform_(self.gate_linear.weight)
            # 偏置初始化为 -2.0 → σ(-2)≈0.12，默认弱使用物理描述符
            gate_bias = args.get('gate_init_bias', -2.0)
            nn.init.constant_(self.gate_linear.bias, gate_bias)
            # 推理时存储门控值，用于可解释性分析
            self._gate_values = None

        # Condition Dropout：训练时随机屏蔽条件标量，破坏捷径学习
        if self.use_cond_dropout:
            self.cond_drop = nn.Dropout(p=self.cond_dropout_p)

        # MLP Head 输入维度：取决于是否启用交互池化
        if self.use_interaction:
            # x_g(512) + h_il⊙h_ref(512) + h_il−h_ref(512) + cond
            head_input_dim = 512 * 3 + cond_dim
        else:
            # 祖宗之法原版：x_g(512) + cond
            head_input_dim = 512 + cond_dim

        # 选择归一化层（LayerNorm 不受 batch 大小影响，更稳健）
        NormLayer = nn.LayerNorm if self.use_layernorm else nn.BatchNorm1d

        # 经典 MLP Head (head_input_dim -> 1024 -> 512 -> 1)
        self.l5 = nn.Sequential(
            nn.Linear(head_input_dim, 1024),
            NormLayer(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 512),
            NormLayer(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

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

        # ── 第二招：Condition Dropout（训练时随机屏蔽，eval 时自动跳过）──
        if self.use_cond_dropout:
            cond = self.cond_drop(cond)

        # ── 第三招：自适应门控物理描述符注入 ──
        # 门控只看图嵌入 x_g，不看条件变量（更干净的因果解释）
        if self.use_adaptive_gate:
            cond_base = cond[:, :self.n_base_features]   # (batch, 7)
            cond_phys = cond[:, self.n_base_features:]   # (batch, n_phys)
            gate = torch.sigmoid(self.gate_linear(x_g))  # (batch, n_phys)
            self._gate_values = gate.detach()             # 存储用于分析
            cond = torch.cat([cond_base, gate * cond_phys], dim=1)

        # ── 第一招：组件级交互池化 ──
        if self.use_interaction and hasattr(data_i, 'mol_type'):
            mol_type = data_i.mol_type
            batch = data_i.batch

            # 按组件类型分别做均值池化，得到各组件的独立表征
            h_cat = global_mean_pool(x[mol_type == 0], batch[mol_type == 0])  # 阳离子 (512)
            h_ani = global_mean_pool(x[mol_type == 1], batch[mol_type == 1])  # 阴离子 (512)
            h_ref = global_mean_pool(x[mol_type == 2], batch[mol_type == 2])  # 制冷剂 (512)

            h_il = h_cat + h_ani  # 离子液体环境表征

            x_concat = torch.cat([
                x_g,             # 全局图表征 (512)
                h_il * h_ref,    # 乘积交互：捕捉 IL-Ref 相互作用强度 (512)
                h_il - h_ref,    # 差分交互：捕捉 IL-Ref 环境不对称性 (512)
                cond             # 条件标量 (cond_dim)
            ], dim=1)
        else:
            # 祖宗之法原版：直接拼接
            x_concat = torch.cat([x_g, cond], dim=1)

        x_out = self.l5(x_concat)

        # ── 第二招：Sigmoid 物理约束，输出严格限制在 [0, 1] ──
        if self.use_sigmoid:
            x_out = torch.sigmoid(x_out)

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
        # v6 动态 cond_dim（支持 M0:7, Msize:9, Mmu:8, Mphys:10）
        cond_dim = args['cond_dim']

        # ── 增强开关（与 GAT 完全一致）──
        self.use_interaction = args.get('use_interaction', False)
        self.use_sigmoid = args.get('use_sigmoid', False)
        self.use_cond_dropout = args.get('use_cond_dropout', False)
        self.cond_dropout_p = args.get('cond_dropout_p', 0.3)
        self.use_layernorm = args.get('use_layernorm', False)

        if self.use_cond_dropout:
            self.cond_drop = nn.Dropout(p=self.cond_dropout_p)

        if self.use_interaction:
            head_input_dim = 512 * 3 + cond_dim
        else:
            head_input_dim = 512 + cond_dim

        NormLayer = nn.LayerNorm if self.use_layernorm else nn.BatchNorm1d

        self.l5 = nn.Sequential(
            nn.Linear(head_input_dim, 1024),
            NormLayer(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 512),
            NormLayer(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

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
        elif self.pool_type == 'attention':
            if hasattr(data_i, 'mol_type'):
                normal_mask = (data_i.mol_type < 3)
                x_g = self.att_pool(x[normal_mask], data_i.batch[normal_mask])
            else:
                x_g = self.att_pool(x, data_i.batch)
        else:
            raise ValueError(f"Unknown pool type {self.pool_type}")

        # ── Condition Dropout ──
        if self.use_cond_dropout:
            cond = self.cond_drop(cond)

        # ── 组件级交互池化 ──
        if self.use_interaction and hasattr(data_i, 'mol_type'):
            mol_type = data_i.mol_type
            batch = data_i.batch

            h_cat = global_mean_pool(x[mol_type == 0], batch[mol_type == 0])
            h_ani = global_mean_pool(x[mol_type == 1], batch[mol_type == 1])
            h_ref = global_mean_pool(x[mol_type == 2], batch[mol_type == 2])

            h_il = h_cat + h_ani

            x_concat = torch.cat([
                x_g, h_il * h_ref, h_il - h_ref, cond
            ], dim=1)
        else:
            x_concat = torch.cat([x_g, cond], dim=1)

        x_out = self.l5(x_concat)

        if self.use_sigmoid:
            x_out = torch.sigmoid(x_out)

        return x_out
