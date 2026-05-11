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
    combined_Graph = Data(x=combined.x, edge_index=combined.edge_index, edge_attr=combined.edge_attr)

    return combined_Graph

def add_global(graph):
    """
    add a global point, all the attribute are set to zero
    :param graph: pyg.data
    :return: pyg.data
    """
    node = torch.tensor([0, 0, 0, 0, 0]).reshape(1, -1)
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

        # 对 3 个全局特征做 StandardScaler 归一化，统一量级：
        # index[5]=Ref_Charge, index[6]=Ref_LogP, index[7]=Ani_MW
        raw_ref_charge = np.array([self.data[i][5] for i in range(self.length)], dtype=np.float32).reshape(-1, 1)
        raw_ref_logp   = np.array([self.data[i][6] for i in range(self.length)], dtype=np.float32).reshape(-1, 1)
        raw_ani_mw     = np.array([self.data[i][7] for i in range(self.length)], dtype=np.float32).reshape(-1, 1)
        self.scaler_ref_charge = StandardScaler().fit(raw_ref_charge)
        self.scaler_ref_logp   = StandardScaler().fit(raw_ref_logp)
        self.scaler_ani_mw     = StandardScaler().fit(raw_ani_mw)

        # show basic information
        print("----info----")
        print("data_length",self.length)
        print("------------")


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cation = self.data[idx][0]
        anion = self.data[idx][1]
        refri = self.data[idx][2]
        T          = self.data[idx][3]
        P          = self.data[idx][4]
        ref_charge = float(self.scaler_ref_charge.transform([[self.data[idx][5]]])[0][0])
        ref_logp   = float(self.scaler_ref_logp.transform([[self.data[idx][6]]])[0][0])
        ani_mw     = float(self.scaler_ani_mw.transform([[self.data[idx][7]]])[0][0])

        cation = self.mol2graph(cation)
        anion  = self.mol2graph(anion)
        refri  = self.mol2graph(refri)
        
        Combine_Graph = combine_Graph([cation, anion, refri])
        if args['add_global'] == True:
            Combine_Graph = add_global(Combine_Graph)

        # cond 维度：[T, P, Ref_Charge↑, Ref_LogP↑, Ani_MW↑] = 5维
        condition = torch.tensor([T, P, ref_charge, ref_logp, ani_mw], dtype=torch.float)


        label = torch.tensor(self.label[idx],dtype=torch.float)


        return Combine_Graph,condition,label

    def mol2graph(self, mol):
        x = torch.tensor(mol[0], dtype=torch.long)
        edge_index = torch.tensor(mol[1], dtype=torch.long)

        # 关键修复：单原子离子（如 [I-], [Cl-]）没有化学键，
        # mol[2] 为空列表 []，torch.tensor([]) 会产生 shape=[0] 的1D张量，
        # 而后续拼接和Embedding需要 shape=[0, 3] 的2D张量，否则会导致
        # CUDA scatter/gather index out of bounds 崩溃。
        if len(mol[2]) == 0:
            edge_attr = torch.zeros((0, 3), dtype=torch.long)
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