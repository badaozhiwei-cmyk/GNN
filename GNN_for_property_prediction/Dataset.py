import numpy as np
import torch
from torch_geometric.data import Batch, Data, Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

args = {
    'add_global':True,
    'bi_direction':True
}


def combine_Graph(Graph_list):
    """
    merge a Graph with multiple subgraph
    Args:
        Graph_list: list() of torch_geometric.data.Data object

    Returns: torch_geometric.data.Data object

    """
    combined = Batch.from_data_list(Graph_list)
    # [修复] 引入 mol_type 用于解决池化灾难
    # combined.batch 原生自带了 0(阳离子), 1(阴离子), 2(制冷剂) 的区分子图索引
    combined_Graph = Data(x=combined.x, edge_index=combined.edge_index, edge_attr=combined.edge_attr, mol_type=combined.batch)

    return combined_Graph

def add_global(graph):
    """
    add a global point, all the attribute are set to zero
    :param graph: pyg.data
    :return: pyg.data
    """
    # V2: 全局节点的特征维度必须与真实原子一致，即 7 维（原5维 + 电负性桶 + 化价半径桶）
    node = torch.tensor([0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)
    # node.shape
    x = torch.cat([graph.x, node], dim=0)
    num_node = x.shape[0] - 1
    new_node = x.shape[0] - 1
    start = []
    end = []
    attr = []
    for i in range(num_node):
        # print(i)
        start.append(i)
        end.append(new_node)
        attr.append([0, 0, 0])
    if args['bi_direction'] == True:
        for i in range(num_node):
            # print(i)
            start.append(new_node)
            end.append(i)
            attr.append([0, 0, 0])

    start = torch.tensor(start).reshape(1, -1)
    end = torch.tensor(end).reshape(1, -1)
    new_edge = torch.cat([start, end], dim=0)
    edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
    attr = torch.tensor(attr)
    edge_attr = torch.cat([graph.edge_attr, attr], dim=0)
    
    # [修复] 给全局节点分配特殊的 mol_type = 3
    if hasattr(graph, 'mol_type'):
        global_mol_type = torch.tensor([3], dtype=torch.long)
        new_mol_type = torch.cat([graph.mol_type, global_mol_type], dim=0)
        g = Data(x = x,edge_index = edge_index,edge_attr = edge_attr, mol_type=new_mol_type)
    else:
        g = Data(x = x,edge_index = edge_index,edge_attr = edge_attr)

    return g


class IL_set(torch.utils.data.Dataset):
    """
    torch dataset
    """
    def __init__(self,path):
        super(IL_set, self).__init__()
        data_path = path + 'data.npy'
        label_path = path + 'label.npy'

        self.data = np.load(data_path, allow_pickle=True)
        self.label = np.load(label_path, allow_pickle=True)

        self.length = self.label.shape[0]

        # [修复] 消除数据泄露，移除 Dataset 内置的 StandardScaler
        # show basic information
        print("----info----")
        print("data_length",self.length)
        print("------------")


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.data[idx]
        cation = data[0]
        anion  = data[1]
        refri  = data[2]

        cation = self.mol2graph(cation)
        anion  = self.mol2graph(anion)
        refri  = self.mol2graph(refri)
        
        Combine_Graph = combine_Graph([cation, anion, refri])
        if args['add_global'] == True:
            Combine_Graph = add_global(Combine_Graph)

        # 提取 7 位原始数据（T, P, 3个制冷剂/阴离子描述符, 2个阳离子描述符）
        # [修复] 不在这里进行 scaler.transform()，统一交由外部 Runner 或封装的 Dataset 处理
        T, P = data[3], data[4]
        ref_charge   = data[5]
        ref_logp     = data[6]
        ani_mw       = data[7]
        cat_charge   = data[8]
        cat_tpsa     = data[9]
        
        # 将这 7 个物理量打包成一个 condition 向量
        condition = torch.tensor([T, P, ref_charge, ref_logp, ani_mw, cat_charge, cat_tpsa], dtype=torch.float)
        label = torch.tensor(self.label[idx],dtype=torch.float)

        return Combine_Graph,condition,label

    def mol2graph(self, mol):
        x = torch.tensor(mol[0], dtype=torch.long)
        edge_index = torch.tensor(mol[1], dtype=torch.long)

        # 关键修复：单原子离子（如 [I-], [Cl-]）没有化学键，
        # 如果边为空，强行给它加一个指向自己的自环，防止 Message Passing 忽略它
        if len(mol[2]) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.zeros((1, 3), dtype=torch.long)
        else:
            edge_attr = torch.tensor(mol[2], dtype=torch.long)

        Graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return Graph

    # collate_fn 已移除：请直接使用 torch_geometric.data.DataLoader，
    # 它内置对 PyG Data 对象的 batch 处理，无需自定义 collate_fn。


if __name__ == '__main__':
    args = {
        'data_path':"clean/"
    }
    D = IL_set(path = args['data_path'])
    print(len(D))
    for item in D:
        G,c,l = item
        print(c,l)