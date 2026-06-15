"""
Explainer_v2.py
===================================================
制冷剂三图体系专用 GNNExplainer（回归版）

与原版 Explainer.py (IL_Explainer) 的唯一核心差异：
  原版 __loss__：针对分类任务，使用 NLL Loss + argmax 取类别
  V2  __loss__：针对回归任务，使用 MSE Loss

    原理：GNNExplainer 在优化边掩码 (edge_mask) 和节点特征掩码 (node_feat_mask) 时，
          Loss = 预测Loss(掩码后的图) + 稀疏化正则项
          预测 Loss 决定了"哪个方向让预测值靠近真实值"，方向错了掩码就毫无意义。
          我们的 GIN 输出 shape=[B,1] 的溶解度回归值，需要用 MSE 来度量距离，
          而不是分类模型的 log_softmax + argmax。
"""

import torch
import torch.nn as nn
from math import sqrt
from torch_geometric.nn import MessagePassing

EPS = 1e-15


class IL_Explainer_v2(torch.nn.Module):
    """
    制冷剂体系 GNNExplainer（回归版）。
    继承了原版 IL_Explainer 的掩码机制，仅将 __loss__ 改为回归 MSE。
    """

    coeffs = {
        'edge_size':           0.005,   # 稀疏化惩罚强度（鼓励掩码二值化）
        'edge_reduction':      'sum',
        'node_feat_size':      1.0,
        'node_feat_reduction': 'mean',
        'edge_ent':            1.0,     # 熵正则项（鼓励掩码非0即1，而非模糊值）
        'node_feat_ent':       0.1,
    }

    def __init__(self, model, epochs: int = 100, lr: float = 0.01, log: bool = False):
        """
        Args:
            model  : 已加载好权重的制冷剂三图 GIN 模型（回归输出，shape=[B,1]）
            epochs : GNNExplainer 掩码优化轮数
            lr     : 掩码优化学习率
            log    : 是否打印每轮 loss（默认关闭，避免刷屏）
        """
        super().__init__()
        self.model  = model
        self.epochs = epochs
        self.lr     = lr
        self.log    = log

    # ── 内部工具：设置边掩码和节点特征掩码 ─────────────────────
    def __set_masks__(self, x, edge_index):
        N, F = x.size()
        E    = edge_index.size(1)

        # 节点特征掩码：维度 = 特征数 F = 5（atomic_number, degree, charge, hybrid, aromatic）
        self.node_feat_mask = nn.Parameter(torch.randn(F) * 0.1)

        # 边掩码：维度 = 边数 E（包含全局虚拟边）
        std = sqrt(2.0 / (2 * N)) * sqrt(2.0)   # xavier 初始化
        self.edge_mask  = nn.Parameter(torch.randn(E) * std)
        self.loop_mask  = edge_index[0] != edge_index[1]

        # 将掩码注册到所有 MessagePassing 层
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__   = True
                module.__edge_mask__ = self.edge_mask
                module.__loop_mask__ = self.loop_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__   = False
                module.__edge_mask__ = None
                module.__loop_mask__ = None
        self.node_feat_mask = None
        self.edge_mask      = None

    # ── 核心改动：回归 MSE Loss ──────────────────────────────────
    def __loss__(self, pred, target):
        """
        回归版 GNNExplainer Loss = MSE(pred, target) + 稀疏化正则

        原版分类 Loss：-log_prob[argmax_class]（最大化目标类别的概率）
        回归版   Loss：MSE(pred, target)（最小化预测值与真实溶解度的差距）

        稀疏化正则项（与原版相同）：
          - edge_size 项：惩罚掩码值之和（让大多数边的掩码接近 0）
          - edge_ent  项：信息熵正则（让掩码非 0 即 1，避免模糊中间值）
        """
        # ── 主 Loss：回归 MSE ──
        mse_loss = nn.functional.mse_loss(pred.squeeze(), target.squeeze())

        # ── 边掩码稀疏化正则 ──
        m = self.edge_mask.sigmoid()
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = mse_loss + self.coeffs['edge_size'] * edge_reduce(m)
        ent  = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        # ── 节点特征掩码稀疏化正则 ──
        m_nf = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m_nf)
        ent_nf = -m_nf * torch.log(m_nf + EPS) - (1 - m_nf) * torch.log(1 - m_nf + EPS)
        loss   = loss + self.coeffs['node_feat_ent'] * ent_nf.mean()

        return loss

    # ── 主接口：解释整张图（制冷剂三图体系专用）───────────────────
    def explain_graph(self, graph, cond):
        """
        对一个三图拼接后的分子体系进行 GNNExplainer 解释。

        Args:
            graph : PyG Data，三图拼接 + 全局节点（由 Dataset_explain_v2 生成）
            cond  : FloatTensor [1, 5]，5 维归一化条件向量

        Returns:
            node_feat_mask : FloatTensor [5]，5 种原子特征的重要性（sigmoid后，0~1）
            edge_mask      : FloatTensor [E]，每条边的重要性（sigmoid后，0~1）
        """
        self.model.eval()
        self.__clear_masks__()

        # ── Step 1：先用原始图得到模型的真实预测值（作为 MSE 的 target）──
        with torch.no_grad():
            target = self.model(graph, cond)   # shape=[1,1]

        # ── Step 2：初始化掩码参数 ──
        self.__set_masks__(graph.x, graph.edge_index)
        self.to(graph.x.device)

        optimizer = torch.optim.Adam(
            [self.node_feat_mask, self.edge_mask], lr=self.lr
        )

        # ── Step 3：迭代优化掩码 ──
        # ── Step 3：迭代优化掩码 ──
        # 判断当前模型是否是带有嵌入层（x_embedding1）的 GIN 模型
        use_embedding_patch = hasattr(self.model, 'x_embedding1')
        
        if use_embedding_patch:
            # 动态覆写 forward 方法，让梯度可以通过连续的 float 掩码流回 node_feat_mask
            import torch.nn.functional as F_func
            model_type = type(self.model).__name__
            
            def masked_forward(pair_graph, cond_val):
                m = self.node_feat_mask.sigmoid()
                # 分别对 7 个特征的 Embedding 乘以其对应的可微分特征掩码
                h = self.model.x_embedding1(pair_graph.x[:, 0]) * m[0] + \
                    self.model.x_embedding2(pair_graph.x[:, 1]) * m[1] + \
                    self.model.x_embedding3(pair_graph.x[:, 2]) * m[2] + \
                    self.model.x_embedding4(pair_graph.x[:, 3]) * m[3] + \
                    self.model.x_embedding5(pair_graph.x[:, 4]) * m[4] + \
                    self.model.x_embedding6(pair_graph.x[:, 5]) * m[5] + \
                    self.model.x_embedding7(pair_graph.x[:, 6]) * m[6]
                
                if model_type == 'GIN':
                    for layer in range(self.model.num_layer):
                        h = self.model.gnns[layer](h, pair_graph.edge_index, pair_graph.edge_attr)
                        h = self.model.batch_norms[layer](h)
                        h = F_func.dropout(F_func.relu(h), self.model.drop_ratio, training=self.model.training)
                    h = self.model.feat_lin(h)
                    
                    # [修复] 三头分别池化
                    h_c = self.model.pool(h[pair_graph.mol_type == 0], pair_graph.batch[pair_graph.mol_type == 0])
                    h_a = self.model.pool(h[pair_graph.mol_type == 1], pair_graph.batch[pair_graph.mol_type == 1])
                    h_r = self.model.pool(h[pair_graph.mol_type == 2], pair_graph.batch[pair_graph.mol_type == 2])
                    
                    h_concat = torch.cat([h_c, h_a, h_r, cond_val], dim=1) 
                    return self.model.pred_head(h_concat)
                    
                elif model_type == 'IL_Net_GCN':
                    from torch_geometric.nn import global_mean_pool
                    x = h
                    edge_index = pair_graph.edge_index
                    edge_weight = torch.sum(pair_graph.edge_attr, dim=1).to(torch.float)
                    
                    x = self.model.l1(x, edge_index, edge_weight)
                    x = self.model.act(x)
                    x = self.model.dropout(x)
                    
                    x = self.model.l2(x, edge_index, edge_weight)
                    x = self.model.act(x)
                    x = self.model.dropout(x)
                    
                    x = self.model.l3(x, edge_index, edge_weight)
                    x = self.model.act(x)
                    x = self.model.dropout(x)
                    
                    # [修复] 三头分别池化
                    x_c = global_mean_pool(x[pair_graph.mol_type == 0], pair_graph.batch[pair_graph.mol_type == 0])
                    x_a = global_mean_pool(x[pair_graph.mol_type == 1], pair_graph.batch[pair_graph.mol_type == 1])
                    x_r = global_mean_pool(x[pair_graph.mol_type == 2], pair_graph.batch[pair_graph.mol_type == 2])
                    
                    x_concat = torch.cat([x_c, x_a, x_r, cond_val], dim=1)
                    return self.model.l4(x_concat)
                    
                elif model_type == 'IL_GAT':
                    from torch_geometric.nn import global_mean_pool
                    x = h
                    edge_index = pair_graph.edge_index
                    
                    x, _ = self.model.l1(x, edge_index, return_attention_weights=True)
                    x = self.model.act(x)
                    x = self.model.dropout(x)
                    
                    x, _ = self.model.l2(x, edge_index, return_attention_weights=True)
                    x = self.model.act(x)
                    x = self.model.dropout(x)
                    
                    x, _ = self.model.l3(x, edge_index, return_attention_weights=True)
                    x = self.model.act(x)
                    x = self.model.dropout(x)
                    
                    # [修复] 三头分别池化
                    x_c = global_mean_pool(x[pair_graph.mol_type == 0], pair_graph.batch[pair_graph.mol_type == 0])
                    x_a = global_mean_pool(x[pair_graph.mol_type == 1], pair_graph.batch[pair_graph.mol_type == 1])
                    x_r = global_mean_pool(x[pair_graph.mol_type == 2], pair_graph.batch[pair_graph.mol_type == 2])
                    
                    x_concat = torch.cat([x_c, x_a, x_r, cond_val], dim=1)
                    return self.model.l5(x_concat)
                    
                else:
                    raise ValueError(f"不支持的带有 Embedding 的模型类型: {model_type}")
                
            original_forward = self.model.forward
            self.model.forward = masked_forward
            
        original_x = graph.x.clone()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            if not use_embedding_patch:
                # 非 GIN 传统模型（如GCN/GAT），其点特征本身就是 float，采用常规掩码
                x_masked = original_x * self.node_feat_mask.view(1, -1).sigmoid()
                graph.x = x_masked.round().long()

            pred = self.model(graph, cond)     # shape=[1,1]
            loss = self.__loss__(pred, target)
            loss.backward()
            optimizer.step()

        # ── 恢复现场：还原真实的 forward 函数和特征图 ──
        if use_embedding_patch:
            self.model.forward = original_forward
        graph.x = original_x

        # ── Step 4：取出最终掩码（sigmoid 到 0~1）──
        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask      = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()
        return node_feat_mask, edge_mask

    def __repr__(self):
        return f'IL_Explainer_v2(epochs={self.epochs}, lr={self.lr})'
