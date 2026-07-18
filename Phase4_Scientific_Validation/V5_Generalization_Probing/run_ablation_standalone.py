import sys
import os
import pathlib as pl
import argparse
import torch
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error

# 动态添加 GNN_for_property_prediction 到环境变量，以便导入 v5 模块
current_dir = str(pl.Path(__file__).resolve().parent)
root_dir = str(pl.Path(current_dir).parent.parent)
sys.path.append(os.path.join(root_dir, 'GNN_for_property_prediction'))

from Dataset_v5 import IL_set_v5
from Model_v5 import IL_GAT_v5

# 导入我们的掩码工具
from run_advanced_ablation import apply_feature_mask, ABLATION_MASKS, print_paper_discussion_template

def evaluate_ablation(model, loader, device, mode):
    model.eval()
    all_preds = []
    all_truth = []
    
    with torch.no_grad():
        for data_i, cond, y in loader:
            data_i = data_i.to(device)
            cond = cond.to(device)
            
            # 【核心逻辑】：在这里对数据进行化学掩码，绝不改变模型！
            data_i = apply_feature_mask(data_i, mode=mode)
            
            out = model(data_i, cond)
            all_preds.extend(out.cpu().numpy())
            all_truth.extend(y.cpu().numpy())
            
    r2 = r2_score(all_truth, all_preds)
    mae = mean_absolute_error(all_truth, all_preds)
    return r2, mae

if __name__ == "__main__":
    parser = argparse.add_argument_group("Ablation Config")
    parser.add_argument("--level", type=str, default="L2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 1. 构造模型参数 (必须与 GAT_Runner_v5 训练时一致)
    Args = {
        'data_path': os.path.join(root_dir, 'processed_tri_data/'),
        'batch_size': 64,
        'emb_dim': 300,
        'dropout_rate': 0.2,
        'pool': 'global',
        'use_ani_mw': False,
        'no_mol_embedding': False,
        'add_global': True
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== 正在进行 {args.level} 的高级化学掩码消融测试 ===")
    
    # 2. 加载数据集
    split_file = os.path.join(root_dir, f"split_{args.level}_indices.npz")
    loaded_idx = np.load(split_file)
    test_indices = loaded_idx['test'].tolist()
    
    Whole_set = IL_set_v5(path=Args['data_path'], args=Args)
    # 不需重新 Fit Scaler，直接用已经训练好的模型即可 (确保测试集被正确切分)
    test_set = torch.utils.data.Subset(Whole_set, test_indices)
    test_loader = DataLoader(test_set, batch_size=Args['batch_size'], shuffle=False)
    
    # 3. 加载模型权重
    model = IL_GAT_v5(Args).to(device)
    model_path = os.path.join(root_dir, f"checkpoints_v5/{args.level}/seed_{args.seed}_best.pth")
    if not os.path.exists(model_path):
        # 兼容一下保存的文件名可能叫 best_model.pth 等
        alt_path = os.path.join(root_dir, f"checkpoints_v5/{args.level}/best_model.pth")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"找不到模型权重文件: {model_path}。请确保先运行了 GAT_Runner_v5 训练过该级别的模型！")
            sys.exit(1)
            
    checkpoint = torch.load(model_path, map_location=device)
    # Runner 中可能是 self._model.state_dict() 保存的，字典键可能有一层嵌套
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"成功加载模型: {model_path}")
    
    # 4. 执行掩码测试
    print("\n=== 开始探测泛化边界的化学信息来源 ===")
    for mode in ["Full", "-Physical", "-Electronic", "-Topology", "Atom-only"]:
        r2, mae = evaluate_ablation(model, test_loader, device, mode)
        print(f"[{mode.ljust(11)}] R² = {r2:8.4f} | MAE = {mae:8.4f}")
        if mode in ["-Physical", "-Topology"]:
            print_paper_discussion_template(mode, r2)
