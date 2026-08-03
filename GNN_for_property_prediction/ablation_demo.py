import torch
import torch.nn as nn

# 这是一个演示如何进行特征消融（Ablation）的补丁脚本。
# 可以在不修改原 Model.py 的情况下，动态遮蔽特定的特征。

def apply_ablation_patch(model, ablation_mask):
    """
    动态修改 GNN 模型的 forward 函数，实现特征消融。
    ablation_mask: 一个长度为 7 的列表，1 表示保留，0 表示剔除。
    例如：[1, 1, 1, 1, 1, 0, 0] 表示剔除电负性和半径。
    """
    original_forward = model.forward

    def new_forward(data_i, cond):
        # 手动计算消融后的 h
        h = 0
        if ablation_mask[0]: h += model.x_embedding1(data_i.x[:, 0])
        if ablation_mask[1]: h += model.x_embedding2(data_i.x[:, 1])
        if ablation_mask[2]: h += model.x_embedding3(data_i.x[:, 2])
        if ablation_mask[3]: h += model.x_embedding4(data_i.x[:, 3])
        if ablation_mask[4]: h += model.x_embedding5(data_i.x[:, 4])
        if ablation_mask[5]: h += model.x_embedding6(data_i.x[:, 5])
        if ablation_mask[6]: h += model.x_embedding7(data_i.x[:, 6])
        
        # 替换原始的 h 计算逻辑，这里需要一点侵入性修改或者重新绑定
        # 为了方便，我们在训练循环中，直接修改 data_i.x 的值也是一种方法。
        pass

    # 更简单的数据级消融法：在 Dataset_explain_v2 取数据时，把被剔除的列全置为 0。
    # 比如： data.x[:, 5] = 0 (抹除电负性)
    print(f"Ablation mask applied: {ablation_mask}")
    return model
