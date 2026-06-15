import os
import sys
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

# 将上级目录加入路径，以便导入 Model 和 Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GNN_for_property_prediction')))

from Model import IL_GAT
from GAT_Runner import IL_set_v2, Args, set_seed

def apply_perturbation(graph, cond, target, value, scalers=None):
    """
    对给定的特征施加扰动。
    注意：输入对象必须已经是 clone/deepcopy 过的独立副本！
    target 可以是:
      - 'T': 温度 +10K
      - 'P': 压力 +1MPa
      - 'ElectroNeg': 图中所有原子的电负性 +1 档
      - 'CovRadius': 图中所有原子的共价半径 +1 档
    """
    if target == 'T':
        # cond[0] 是标准化的 T，增加 10K 对应的标度差值为 10.0 / T的std
        delta = 10.0 / scalers[0].scale_[0]
        cond[:, 0] += delta
    elif target == 'P':
        # cond[1] 是标准化的 P，增加 1MPa
        delta = 1.0 / scalers[1].scale_[0]
        cond[:, 1] += delta
    elif target == 'ElectroNeg':
        # graph.x[:, 5] 为电负性分桶，增加 1 档，并强制截断最大值为 7（防止 Embedding 越界）
        graph.x[:, 5] = torch.clamp(graph.x[:, 5] + 1, max=7)
    elif target == 'CovRadius':
        # graph.x[:, 6] 为共价半径分桶，增加 1 档，强制截断最大值为 7
        graph.x[:, 6] = torch.clamp(graph.x[:, 6] + 1, max=7)
    return graph, cond

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")

    # 1. 恢复与训练时完全相同的测试集和特征转换 Scaler
    SPLIT_SEED = 42
    set_seed(SPLIT_SEED)
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'processed_tri_data')) + '/'
    Whole_set = IL_set_v2(path=data_path)
    
    train_size = int(len(Whole_set) * 0.7)
    test_size  = int(len(Whole_set) * 0.2)
    dev_size   = len(Whole_set) - train_size - test_size

    train_set, dev_set, test_set = random_split(
        Whole_set, [train_size, dev_size, test_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )
    # 重构 Scaler（训练集基准）
    Whole_set.fit_scalers(train_set.indices)
    
    # 我们只在测试集上进行反事实测试（保证泛化性）
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # 2. 加载训练好的 GAT 模型
    model_path = '../GNN_for_property_prediction/checkpoints_v2/best_gat_seed_42.pth'
    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}，请先运行 GAT_Runner.py。")
        return
    
    model = IL_GAT(Args).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 3. 定义我们想要测试的扰动
    perturbations = ['T', 'P', 'ElectroNeg', 'CovRadius']
    perturbation_labels = {
        'T': 'Temperature (+10K)',
        'P': 'Pressure (+1MPa)',
        'ElectroNeg': 'Electronegativity (+1 Bin)',
        'CovRadius': 'Covalent Radius (+1 Bin)'
    }
    
    # 存储每种扰动带来的 \Delta y 累计值
    delta_y_sum = {p: 0.0 for p in perturbations}
    total_samples = 0

    print("\n开始执行反事实干预测试 (Counterfactual Intervention)...")
    
    with torch.no_grad():
        for graph, cond, label in test_loader:
            graph = graph.to(device)
            cond = cond.to(device)
            bsz = cond.size(0)
            total_samples += bsz

            # a) 基线预测 (Original)
            y_orig = model(graph, cond).flatten()

            # b) 分别测试每种扰动
            for p in perturbations:
                # [避坑陷阱一]：必须做深度克隆，防止张量被就地修改污染！
                p_graph = graph.clone()
                p_cond = cond.clone()
                
                # [避坑陷阱二 & 三]：精确定位并加入截断保护
                p_graph, p_cond = apply_perturbation(p_graph, p_cond, p, value=1, scalers=Whole_set.scalers)
                
                # 反事实预测 (Counterfactual)
                y_new = model(p_graph, p_cond).flatten()
                
                # 累加差值
                delta_y = (y_new - y_orig).sum().item()
                delta_y_sum[p] += delta_y

    # 4. 计算平均 \Delta y
    avg_delta_y = {p: delta_y_sum[p] / total_samples for p in perturbations}
    
    print("\n--- 反事实干预分析结果 ---")
    for p in perturbations:
        sign = "上升" if avg_delta_y[p] > 0 else "下降"
        print(f"扰动 {perturbation_labels[p]:<30} -> 模型预测平均 {sign} {abs(avg_delta_y[p]):.5f}")

    # 5. 可视化
    plt.figure(figsize=(8, 6))
    labels = [perturbation_labels[p] for p in perturbations]
    values = [avg_delta_y[p] for p in perturbations]
    
    colors = ['#ff6b6b' if v < 0 else '#4ecdc4' for v in values]
    bars = plt.barh(labels, values, color=colors)
    
    plt.axvline(x=0, color='black', linewidth=1.2, linestyle='--')
    plt.xlabel('Average Prediction Change ($\Delta y$)')
    plt.title('Counterfactual Intervention Analysis (Directional Impact)')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        x_offset = 0.002 if width > 0 else -0.002
        ha = 'left' if width > 0 else 'right'
        plt.text(width + x_offset, bar.get_y() + bar.get_height()/2, 
                 f"{width:.4f}", ha=ha, va='center', fontsize=10, fontweight='bold')

    os.makedirs('fragment_explain_result', exist_ok=True)
    save_path = 'fragment_explain_result/counterfactual_analysis.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\n可视化条形图已保存至: {save_path}")
    print("结论推演：\n- 若温度上升导致预测下降，符合气体在液体中溶解度随温度降低的物理常识。\n- 若压力上升导致预测上升，符合亨利定律。")

if __name__ == "__main__":
    main()
