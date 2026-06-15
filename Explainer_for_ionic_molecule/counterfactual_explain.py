import os
import sys
import copy
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

# ── 将上级目录加入路径，以便导入 Model 和 Dataset ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'GNN_for_property_prediction'))

from Model import GIN, IL_Net_GCN, IL_GAT
from Dataset_explain_v2 import IL_set_v2

def set_seed(seed):
    """设置随机种子以确保测试集切分的复现性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def apply_perturbation(graph, cond, target, value, dataset):
    """
    对给定的物理特征施加扰动，并利用 Dataset 里的 Scaler 动态计算归一化标度变化。
    注意：输入对象必须已经是 clone 过的独立副本以防就地修改污染！
    """
    if target == 'T':
        # 温度 +10K。计算归一化后的偏移量：10.0 / T 的 std 标度
        delta = 10.0 / dataset.scaler_T.scale_[0]
        cond[:, 0] += delta
    elif target == 'P':
        # 压力 +1MPa
        delta = 1.0 / dataset.scaler_P.scale_[0]
        cond[:, 1] += delta
    elif target == 'ElectroNeg':
        # 所有原子电负性等级增加 1 档，并强制截断在 [0, 7] 范围内防止 Embedding 越界
        graph.x[:, 5] = torch.clamp(graph.x[:, 5] + 1, max=7)
    elif target == 'CovRadius':
        # 所有原子共价半径等级增加 1 档，并强制截断在 [0, 7] 范围内
        graph.x[:, 6] = torch.clamp(graph.x[:, 6] + 1, max=7)
    return graph, cond

def main():
    parser = argparse.ArgumentParser(description='制冷剂 GNN 体系反事实干预物理倾向性测试 V2')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型权重路径 (GIN/GAT/GCN)')
    parser.add_argument('--split_seed', type=int, default=42, help='数据集划分的随机种子，需与训练时保持一致')
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")
    set_seed(args_cli.split_seed)

    # ── 1. 加载数据集与重构归一化 Scaler ──────────────────────
    data_path = os.path.join(ROOT, 'processed_tri_data')
    data_npy = os.path.join(data_path, 'data.npy')
    label_npy = os.path.join(data_path, 'label.npy')

    if not os.path.exists(data_npy):
        raise FileNotFoundError(f"找不到数据集文件 {data_npy}，请先运行数据预处理。")

    Whole_set = IL_set_v2(data_npy_path=data_npy, label_npy_path=label_npy)
    
    # 按照与训练时完全一致的 7:2:1 比例进行划分
    train_size = int(len(Whole_set) * 0.7)
    test_size  = int(len(Whole_set) * 0.2)
    dev_size   = len(Whole_set) - train_size - test_size

    # 注意：这里的 random_split 需要手动指定 Generator 保证和训练时取出的测试集一模一样
    _, _, test_set = random_split(
        Whole_set, [train_size, dev_size, test_size],
        generator=torch.Generator().manual_seed(args_cli.split_seed)
    )
    
    # 我们只在测试集上进行反事实验证
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=IL_set_v2.collate_fn)

    # ── 2. 根据权重名称自适应推断模型架构并载入权重 ─────────────────
    model_path = args_cli.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到指定的模型文件: {model_path}")

    model_name_lower = os.path.basename(model_path).lower()
    if 'gcn' in model_name_lower:
        print("  => 检测到 GCN 模型权重，正在实例化 IL_Net_GCN")
        gcn_args = {'emb_dim': 300, 'dropout_rate': 0.4}
        model = IL_Net_GCN(gcn_args).to(device)
    elif 'gat' in model_name_lower:
        print("  => 检测到 GAT 模型权重，正在实例化 IL_GAT")
        gat_args = {'emb_dim': 300, 'dropout_rate': 0.2}
        model = IL_GAT(gat_args).to(device)
    else:
        print("  => 默认使用 GIN 模型架构")
        gin_args = {
            'num_gin_layer': 3,
            'emb_dim': 300,
            'feat_dim': 512,
            'drop_ratio': 0.2,
            'pool': 'mean',
        }
        model = GIN(gin_args).to(device)

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # ── 3. 运行物理特征扰动干预测试 ────────────────────────────
    perturbations = ['T', 'P', 'ElectroNeg', 'CovRadius']
    perturbation_labels = {
        'T': 'Temperature (+10K)',
        'P': 'Pressure (+1MPa)',
        'ElectroNeg': 'Electronegativity (+1 Bin)',
        'CovRadius': 'Covalent Radius (+1 Bin)'
    }
    
    delta_y_sum = {p: 0.0 for p in perturbations}
    total_samples = 0

    print(f"\n开始对模型 {os.path.basename(model_path)} 执行反事实干预测试...")
    
    with torch.no_grad():
        for graph, cond, label, _ in test_loader:
            graph = graph.to(device)
            cond = cond.to(device)
            bsz = cond.size(0)
            total_samples += bsz

            # 预测基准值
            y_orig = model(graph, cond).flatten()

            for p in perturbations:
                # 避坑指南：必须进行 clone 避免就地突变污染原图
                p_graph = graph.clone()
                p_cond = cond.clone()
                
                # 应用扰动
                p_graph, p_cond = apply_perturbation(p_graph, p_cond, p, value=1, dataset=Whole_set)
                
                # 反事实预测
                y_new = model(p_graph, p_cond).flatten()
                
                # 累加预测值变化量
                delta_y = (y_new - y_orig).sum().item()
                delta_y_sum[p] += delta_y

    # ── 4. 计算并输出平均 \Delta y ──────────────────────────
    avg_delta_y = {p: delta_y_sum[p] / total_samples for p in perturbations}
    
    print("\n--- 反事实物理倾向性干预结果 ---")
    for p in perturbations:
        sign = "上升" if avg_delta_y[p] > 0 else "下降"
        print(f"  扰动 {perturbation_labels[p]:<30} => 模型预测均值 {sign} {abs(avg_delta_y[p]):.6f}")

    # ── 5. 可视化分析结论 ──────────────────────────────
    plt.figure(figsize=(8, 5))
    labels = [perturbation_labels[p] for p in perturbations]
    values = [avg_delta_y[p] for p in perturbations]
    
    colors = ['#ff6b6b' if v < 0 else '#4ecdc4' for v in values]
    bars = plt.barh(labels, values, color=colors)
    
    plt.axvline(x=0, color='black', linewidth=1.2, linestyle='--')
    plt.xlabel('Average Prediction Change (\\Delta y)')
    plt.title(f'Counterfactual Intervention Analysis\n(Model: {os.path.basename(model_path)})')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    for bar in bars:
        width = bar.get_width()
        x_offset = 0.002 if width > 0 else -0.002
        ha = 'left' if width > 0 else 'right'
        plt.text(width + x_offset, bar.get_y() + bar.get_height()/2, 
                 f"{width:.5f}", ha=ha, va='center', fontsize=9, fontweight='bold')

    out_dir = os.path.join(ROOT, 'Explainer_for_ionic_molecule', 'fragment_explain_result')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'counterfactual_analysis.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\n📊 可视化条形图已成功保存至:\n   {save_path}")
    print("\n💡 [物理规律对照表]:")
    print("  - 温度升高 => 溶解度应下降 (\\Delta y < 0) | 符合物理热力学常识")
    print("  - 压力升高 => 溶解度应上升 (\\Delta y > 0) | 符合亨利定律")
    print("  - 电负性强 => 极性大，极性相互作用强 => 溶解度应上升 (\\Delta y > 0)")
    print("  - 空间半径大 => 位阻大，难以嵌入空隙 => 溶解度应下降 (\\Delta y < 0)")

if __name__ == "__main__":
    main()
