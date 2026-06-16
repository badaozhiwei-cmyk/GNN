import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv,MessagePassing
from torch_geometric.nn import global_max_pool,global_mean_pool,global_add_pool
from torch_geometric.utils import add_self_loops, degree, softmax

# global argument
num_atom_type = 119 
num_Hbrid = 8
num_Aro = 2
num_degree = 7
num_charge = 3
num_charge = 3
num_eneg   = 8   # V2 新增：电负性分桶数（0~7）
num_radius = 8   # V2 新增：共价半径分桶数（0~7）

num_bond_type = 5 
num_bond_isAromatic = 2
num_bond_isInRing = 2

# GCN (3-layer version)
# n_features should be 7 (atomic_number, hybridization, aromatic, degree, charge, electronegativity, cov_radius)
# cond is 7-dim (T, P, ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa)
# Graph conv output: 512-dim  →  MLP input: 512 + 7 = 519
class IL_Net_GCN(torch.nn.Module):
    def __init__(self, args):
        super(IL_Net_GCN,self).__init__()
        self.args = args
        self.emb_dim = args.get('emb_dim', 300)
        
        # 添加与 GIN 相同的 Embedding 层，避免直接将分类序号当做连续变量计算
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

        # 3-layer GCN: emb_dim → 512 → 1024 → 512
        self.l1 = GCNConv(self.emb_dim, 512, normalize=True)
        self.l2 = GCNConv(512, 1024, normalize=True)
        self.l3 = GCNConv(1024, 512, normalize=True)

        # MLP head: graph_repr(512*4) + cond(7) = 2055
        self.l4 = nn.Sequential(
            nn.Linear(2055, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 1),
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=args['dropout_rate'])

    def extract(self, x, batch):
        """Extract the global node (last node of each graph in batch) as graph representation."""
        output, count = torch.unique(batch, return_counts=True)
        count = count.tolist()

        l = []
        cur = 0
        for i in count:
            cur += i
            l.append(cur)
        re = []
        for j in l:
            re.append(x[j - 1].reshape(1, -1))

        g = torch.cat(re, dim=0)
        return g

    def forward(self, data_i, cond):
        h = self.x_embedding1(data_i.x[:, 0]) + \
            self.x_embedding2(data_i.x[:, 1]) + \
            self.x_embedding3(data_i.x[:, 2]) + \
            self.x_embedding4(data_i.x[:, 3]) + \
            self.x_embedding5(data_i.x[:, 4]) + \
            self.x_embedding6(data_i.x[:, 5]) + \
            self.x_embedding7(data_i.x[:, 6])
            
        edge_index = data_i.edge_index
        edge_weight = torch.sum(data_i.edge_attr, dim=1).to(torch.float)

        # Layer 1
        x = self.l1(h, edge_index, edge_weight)
        x = self.act(x)
        x = self.dropout(x)

        # Layer 2
        x = self.l2(x, edge_index, edge_weight)
        x = self.act(x)
        x = self.dropout(x)

        # Layer 3
        x = self.l3(x, edge_index, edge_weight)
        x = self.act(x)
        x = self.dropout(x)

        # [修复] 分别对阳离子、阴离子、制冷剂进行池化，同时保留全局节点！
        x_c = global_mean_pool(x[data_i.mol_type == 0], data_i.batch[data_i.mol_type == 0])
        x_a = global_mean_pool(x[data_i.mol_type == 1], data_i.batch[data_i.mol_type == 1])
        x_r = global_mean_pool(x[data_i.mol_type == 2], data_i.batch[data_i.mol_type == 2])
        x_g = global_mean_pool(x[data_i.mol_type == 3], data_i.batch[data_i.mol_type == 3])

        # Concatenate with physical condition vector and predict
        x_concat = torch.cat([x_c, x_a, x_r, x_g, cond], dim=1)
        x_out = self.l4(x_concat)

        return x_out

# GAT
class IL_GAT(torch.nn.Module):
    def __init__(self, args):
        super(IL_GAT,self).__init__()
        self.args = args
        self.emb_dim = args.get('emb_dim', 300)
        
        # 添加与 GIN 相同的 Embedding 层，避免直接将分类序号当做连续变量计算
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

        # 3-layer GAT
        self.l1 = GATv2Conv(self.emb_dim, 512)
        self.l2 = GATv2Conv(512, 1024)
        self.l3 = GATv2Conv(1024, 512)

        # MLP head: graph_repr(512) + cond(7) = 519
        self.l5 = nn.Sequential(
            nn.Linear(519, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 1),
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=args['dropout_rate'])

    def extract(self,x,batch):
        output, count= torch.unique(batch, return_counts=True)
        count = count.tolist()

        l = []
        cur = 0
        for i in count:
            cur += i
            l.append(cur)
        re = []
        for j in l:
            re.append(x[j - 1].reshape(1,-1))

        g = torch.cat(re,dim = 0)

        return g

    def forward(self, data_i, cond):
        h = self.x_embedding1(data_i.x[:, 0]) + \
            self.x_embedding2(data_i.x[:, 1]) + \
            self.x_embedding3(data_i.x[:, 2]) + \
            self.x_embedding4(data_i.x[:, 3]) + \
            self.x_embedding5(data_i.x[:, 4]) + \
            self.x_embedding6(data_i.x[:, 5]) + \
            self.x_embedding7(data_i.x[:, 6])
            
        x, edge_index = h, data_i.edge_index

        x,(edge1,attention1) = self.l1(x, edge_index, return_attention_weights = True )
        x = self.act(x)
        x = self.dropout(x)

        x,(edge2,attention2) = self.l2(x, edge_index,return_attention_weights = True )
        x = self.act(x)
        x = self.dropout(x)

        x,(edge3,attention3) = self.l3(x, edge_index,return_attention_weights = True )
        x = self.act(x)
        x = self.dropout(x)

        # [回退] 统一池化，为了兼容你的旧版权重文件 (519维)
        x_pool = global_mean_pool(x, data_i.batch)

        x_concat = torch.cat([x_pool, cond], dim=1)
        x_out = self.l5(x_concat)

        return x_out


# GIN
class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim),
            nn.ReLU(),
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_isInRing, emb_dim)
        self.edge_embedding3 = nn.Embedding(num_bond_isAromatic, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges (directly on correct device).
        self_loop_attr = torch.zeros(x.size(0), 3, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = num_bond_type - 1      # self-loop bond type
        self_loop_attr[:, 1] = num_bond_isInRing - 1   # self-loop isInRing
        self_loop_attr[:, 2] = num_bond_isAromatic - 1  # self-loop isAromatic
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + \
                          self.edge_embedding2(edge_attr[:,1]) + \
                          self.edge_embedding3(edge_attr[:,2])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GIN(nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        self.num_layer = args['num_gin_layer']
        self.emb_dim = args['emb_dim']
        self.feat_dim = args['feat_dim']
        self.drop_ratio = args['drop_ratio']
        pool = args['pool']

        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_Hbrid, self.emb_dim)
        self.x_embedding3 = nn.Embedding(num_Aro, self.emb_dim)
        self.x_embedding4 = nn.Embedding(num_degree, self.emb_dim)
        self.x_embedding5 = nn.Embedding(num_charge, self.emb_dim)
        self.x_embedding6 = nn.Embedding(num_eneg,   self.emb_dim)  # V2新增：电负性桶
        self.x_embedding7 = nn.Embedding(num_radius, self.emb_dim)  # V2新增：共价半径桶
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        nn.init.xavier_uniform_(self.x_embedding7.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(self.num_layer):
            self.gnns.append(GINEConv(self.emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layer):
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.pred_head = nn.Sequential(
            nn.Linear(self.feat_dim * 4 + 7, self.feat_dim),  # 4 graphs (c, a, r, g) + 7 cond
            nn.Softplus(),
            nn.Linear(self.feat_dim, int(self.feat_dim/2)),
            nn.Softplus(),
            nn.Linear(int(self.feat_dim/2), 1)
        )
    def extract(self,x,batch):
        output, count= torch.unique(batch, return_counts=True)
        count = count.tolist()

        l = []
        cur = 0
        for i in count:
            cur += i
            l.append(cur)
        re = []
        for j in l:
            re.append(x[j - 1].reshape(1,-1))

        g = torch.cat(re,dim = 0)

        return g
    def forward(self, pair_graph, cond):
        # GIN layer
        h = self.x_embedding1(pair_graph.x[:, 0]) + \
            self.x_embedding2(pair_graph.x[:, 1]) + \
            self.x_embedding3(pair_graph.x[:, 2]) + \
            self.x_embedding4(pair_graph.x[:, 3]) + \
            self.x_embedding5(pair_graph.x[:, 4]) + \
            self.x_embedding6(pair_graph.x[:, 5]) + \
            self.x_embedding7(pair_graph.x[:, 6])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, pair_graph.edge_index, pair_graph.edge_attr)
            h = self.batch_norms[layer](h)
            h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.feat_lin(h)
        
        # [修复] 分别对阳离子、阴离子、制冷剂进行池化，同时保留全局节点！
        h_c = self.pool(h[pair_graph.mol_type == 0], pair_graph.batch[pair_graph.mol_type == 0])
        h_a = self.pool(h[pair_graph.mol_type == 1], pair_graph.batch[pair_graph.mol_type == 1])
        h_r = self.pool(h[pair_graph.mol_type == 2], pair_graph.batch[pair_graph.mol_type == 2])
        h_g = self.pool(h[pair_graph.mol_type == 3], pair_graph.batch[pair_graph.mol_type == 3])

        # 【特征融合层】
        # 这一步实现了“物理增强”：模型同时参考了结构和物理常数
        h_concat = torch.cat([h_c, h_a, h_r, h_g, cond], dim=1) 
        
        # 将融合后的总特征送入 MLP 输出头进行最终的溶解度预测
        return self.pred_head(h_concat)