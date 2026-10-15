# run_dg3b_kaggle.py
import sys, os, time, json, random, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 自动定位项目根目录
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "GNN_for_property_prediction"))
from Dataset_v6 import IL_set_v6, MODE_COND_DIM
from Model_v6 import IL_GAT_v6

out_dir = ROOT / "output_results"
ckpt_dir = out_dir / "checkpoints"
preds_dir = out_dir / "predictions"
ckpt_dir.mkdir(parents=True, exist_ok=True)
preds_dir.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n==================================================================", flush=True)
print(f"  Decision Gate 3B (Kaggle GPU 加速版) 启动", flush=True)
print(f"  运行硬件: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
print(f"==================================================================\n", flush=True)

class EarlyStopping:
    def __init__(self, patience=15, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.val_loss_min = float('inf')
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_epoch = epoch
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return True

hfc12 = ['R32', 'R134a', 'R125', 'R134', 'R152a', 'R23', 'R41', 'R161', 'R143a', 'R245fa', 'R236fa', 'R227ea']
df_v6 = pd.read_csv(ROOT / "processed_tri_data_v6" / "meta_info.csv")
active_1403 = df_v6[df_v6['Refrigerant'].str.strip().str.upper().isin([h.upper() for h in hfc12])].index.values.tolist()
df_1403 = df_v6.iloc[active_1403].copy().reset_index(drop=True)
df_1403['ref_clean'] = df_1403['Refrigerant'].str.strip().str.upper()
df_1403['il_combo'] = df_1403['IL cation'].str.strip() + "_" + df_1403['IL anion'].str.strip()

df_2739 = pd.read_csv(ROOT / "processed_tri_data_hfc2739" / "meta_info.csv")
df_2739['ref_clean'] = df_2739['refrigerant'].str.strip().str.upper()
df_2739['il_combo'] = df_2739['cation'].str.strip() + "_" + df_2739['anion'].str.strip()

df_4444 = pd.read_csv(ROOT / "datasets" / "full_4444_v7" / "meta_info.csv")
blind9 = ['R114', 'R114A', 'R116', 'R124', 'R124A', 'R14', 'R218', 'R22', 'R22B1']
df_4444['ref_upper'] = df_4444['refrigerant'].str.strip().str.upper()
active_3845 = df_4444[~df_4444['ref_upper'].isin(blind9)].index.values.tolist()
df_3845 = df_4444.iloc[active_3845].copy().reset_index(drop=True)
df_3845['ref_clean'] = df_3845['refrigerant'].str.strip().str.upper()
df_3845['il_combo'] = df_3845['cation'].str.strip() + "_" + df_3845['anion'].str.strip()

model_args = {
    'batch_size': 32,
    'lr': 0.001,
    'epoch': 100,
    'weight_decay': 1e-6,
    'emb_dim': 300,
    'dropout_rate': 0.2,
    'patience': 15,
    'pool': 'global',
    'add_global': True,
    'descriptor_mode': 'Mreduced',
    'cond_dim': MODE_COND_DIM['Mreduced'],
    'use_interaction': False,
    'use_sigmoid': False,
    'use_cond_dropout': False,
    'use_layernorm': False,
    'use_adaptive_gate': False,
    'feature_clip': None,
}

print("  实例化 PyG 图数据集...", flush=True)
Whole_set_1403 = IL_set_v6(path=str(ROOT / "processed_tri_data_v6"), args=model_args, valid_indices=active_1403)
Whole_set_2739 = IL_set_v6(path=str(ROOT / "processed_tri_data_hfc2739"), args=model_args)
Whole_set_3845 = IL_set_v6(path=str(ROOT / "datasets" / "full_4444_v7"), args=model_args, valid_indices=active_3845)

def get_group_b_indices(df_pool, target_n, seed=42):
    grouped = df_pool.groupby('il_combo', group_keys=False)
    first_picks = grouped.apply(lambda g: g.sample(n=1, random_state=seed))
    remaining = df_pool.drop(index=first_picks.index)
    needed = target_n - len(first_picks)
    second_picks = remaining.sample(n=needed, random_state=seed)
    sampled = pd.concat([first_picks, second_picks]).sample(frac=1.0, random_state=seed)
    return sorted(sampled.index.tolist())

loro_splits = {
    'R32': {
        'split_file': ROOT / "splits" / "HFC_loro_all" / "loro_R32.npz",
        'test_ref': 'R32',
        'val_refs': ['R125', 'R161'],
        'expected_n_test': 287,
        'expected_n_val': 254,
        'expected_n_train_a': 862,
    },
    'R143a': {
        'split_file': ROOT / "splits" / "HFC_loro_all" / "loro_R143a.npz",
        'test_ref': 'R143A',
        'val_refs': ['R125', 'R227EA'],
        'expected_n_test': 83,
        'expected_n_val': 225,
        'expected_n_train_a': 1095,
    }
}

def run_single_fit(fold_name, group_id, group_name, train_dataset, train_indices,
                   scaler_dataset, Whole_set_1403, val_indices, test_indices,
                   test_df, metadata_info):
    fit_tag = f"{fold_name}_{group_id}"
    print(f"\n------------------------------------------------------------------", flush=True)
    print(f"  >>> [{fit_tag}] {group_name} <<<", flush=True)
    print(f"      Train: N={len(train_indices):>4} | IL组合={metadata_info['n_ils']:>2} | 冷媒数={metadata_info['n_refs']:>2}", flush=True)
    print(f"      Val  : N={len(val_indices):>4} | Test : N={len(test_indices):>4} ({metadata_info['test_ref']})", flush=True)
    print(f"------------------------------------------------------------------", flush=True)

    scaler_save_dir = ckpt_dir / fit_tag
    scaler_dataset.fit_scalers(train_indices, save_dir=str(scaler_save_dir))
    Whole_set_1403.means = scaler_dataset.means.copy()
    Whole_set_1403.scales = scaler_dataset.scales.copy()

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset   = torch.utils.data.Subset(Whole_set_1403, val_indices)
    test_subset  = torch.utils.data.Subset(Whole_set_1403, test_indices)

    set_seed(42)
    train_loader = DataLoader(train_subset, batch_size=model_args['batch_size'], shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_subset,   batch_size=model_args['batch_size'], shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_subset,  batch_size=model_args['batch_size'], shuffle=False, num_workers=2)

    model = IL_GAT_v6(model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_args['lr'], weight_decay=model_args['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=model_args['epoch'], eta_min=1e-5)
    criterion = nn.HuberLoss(delta=0.05)
    early_stopping = EarlyStopping(patience=model_args['patience'])

    ckpt_file = ckpt_dir / f"{fit_tag}_best.pth"
    t0 = time.time()

    for epoch in range(1, model_args['epoch'] + 1):
        model.train()
        train_loss = 0.0
        for graph, cond, label in train_loader:
            graph, cond, label = graph.to(device), cond.to(device), label.to(device)
            optimizer.zero_grad()
            y = model(graph, cond)
            loss = criterion(y.flatten(), label.flatten())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for graph, cond, label in val_loader:
                graph, cond, label = graph.to(device), cond.to(device), label.to(device)
                y = model(graph, cond)
                val_loss += criterion(y.flatten(), label.flatten()).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss / len(val_loader)

        is_best = early_stopping(avg_val, epoch)
        if is_best:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_loss': avg_val}, ckpt_file)

        if epoch % 10 == 0 or epoch == 1 or early_stopping.early_stop:
            print(f"  [{fit_tag}] Epoch {epoch:>3d}/{model_args['epoch']} | Train: {avg_train:.5f} | Val: {avg_val:.5f} {'*BEST*' if is_best else ''}", flush=True)

        if early_stopping.early_stop:
            print(f"  [{fit_tag}] ⏹ 早停触发于 Epoch {epoch} (最佳: {early_stopping.best_epoch})", flush=True)
            break

    train_time = time.time() - t0

    saved_ckpt = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(saved_ckpt['model_state_dict'])
    model.eval()

    raw_preds, true_labels = [], []
    with torch.no_grad():
        for graph, cond, label in test_loader:
            graph, cond, label = graph.to(device), cond.to(device), label.to(device)
            pred = model(graph, cond)
            raw_preds.extend(pred.flatten().cpu().numpy().tolist())
            true_labels.extend(label.cpu().numpy().tolist())

    raw_preds = np.array(raw_preds)
    true_labels = np.array(true_labels)
    clipped_preds = np.clip(raw_preds, 0.0, 1.0)

    clip_mae = mean_absolute_error(true_labels, clipped_preds)
    clip_rmse = np.sqrt(mean_squared_error(true_labels, clipped_preds))
    clip_r2 = r2_score(true_labels, clipped_preds)
    bias_clip = float(np.mean(clipped_preds - true_labels))

    print(f"  >>> 测试指标 [{fit_tag}]: Clipped MAE = {clip_mae:.5f} | R² = {clip_r2:.4f} | Bias = {bias_clip:+.5f} (耗时: {train_time:.1f}s)", flush=True)

    pred_df = pd.DataFrame({
        'fold': fold_name, 'group': group_id, 'group_name': group_name,
        'true_x1': true_labels, 'pred_x1_raw': raw_preds, 'pred_x1_clipped': clipped_preds,
        'error_clipped': clipped_preds - true_labels, 'abs_error_clipped': np.abs(clipped_preds - true_labels)
    })
    pred_df.to_csv(preds_dir / f"predictions_{fit_tag}.csv", index=False)

    return {
        'fold': fold_name, 'group_id': group_id, 'group_name': group_name,
        'n_train': len(train_indices), 'il_combos_train': metadata_info['n_ils'],
        'best_epoch': early_stopping.best_epoch, 'val_loss_best': float(early_stopping.val_loss_min),
        'clipped_mae': float(clip_mae), 'clipped_rmse': float(clip_rmse),
        'clipped_r2': float(clip_r2), 'mean_bias_clipped': bias_clip,
        'train_time_sec': round(train_time, 1)
    }

all_results = []
for fold_key, fold_info in loro_splits.items():
    sp = np.load(fold_info['split_file'])
    tr_idx_1403 = sp['train'].astype(int).tolist()
    va_idx_1403 = sp['val'].astype(int).tolist()
    te_idx_1403 = sp['test'].astype(int).tolist()
    test_sub_df = df_1403.iloc[te_idx_1403].copy().reset_index(drop=True)
    tgt_ref = fold_info['test_ref']
    val_refs = fold_info['val_refs']

    # Group A
    info_a = {'test_ref': tgt_ref, 'n_ils': df_1403.iloc[tr_idx_1403]['il_combo'].nunique(), 'n_refs': df_1403.iloc[tr_idx_1403]['ref_clean'].nunique()}
    res_a = run_single_fit(fold_key, "Group_A", "Core_1403_Baseline", Whole_set_1403, tr_idx_1403, Whole_set_1403, Whole_set_1403, va_idx_1403, te_idx_1403, test_sub_df, info_a)
    all_results.append(res_a)

    # Group B
    pool_b_df = df_2739[(df_2739['ref_clean'] != tgt_ref) & (~df_2739['ref_clean'].isin(val_refs))]
    b_indices = get_group_b_indices(pool_b_df, target_n=fold_info['expected_n_train_a'], seed=42)
    info_b = {'test_ref': tgt_ref, 'n_ils': df_2739.iloc[b_indices]['il_combo'].nunique(), 'n_refs': df_2739.iloc[b_indices]['ref_clean'].nunique()}
    res_b = run_single_fit(fold_key, "Group_B", "Expanded_Matched_IL_Diversity", Whole_set_2739, b_indices, Whole_set_2739, Whole_set_1403, va_idx_1403, te_idx_1403, test_sub_df, info_b)
    all_results.append(res_b)

    # Group C
    pool_c_df = df_2739[(df_2739['ref_clean'] != tgt_ref) & (~df_2739['ref_clean'].isin(val_refs))]
    c_indices = pool_c_df.index.tolist()
    info_c = {'test_ref': tgt_ref, 'n_ils': df_2739.iloc[c_indices]['il_combo'].nunique(), 'n_refs': df_2739.iloc[c_indices]['ref_clean'].nunique()}
    res_c = run_single_fit(fold_key, "Group_C", "Expanded_2739_Full_HFC", Whole_set_2739, c_indices, Whole_set_2739, Whole_set_1403, va_idx_1403, te_idx_1403, test_sub_df, info_c)
    all_results.append(res_c)

    # Group D
    pool_d_df = df_3845[(df_3845['ref_clean'] != tgt_ref) & (~df_3845['ref_clean'].isin(val_refs))]
    d_indices = pool_d_df.index.tolist()
    info_d = {'test_ref': tgt_ref, 'n_ils': df_3845.iloc[d_indices]['il_combo'].nunique(), 'n_refs': df_3845.iloc[d_indices]['ref_clean'].nunique()}
    res_d = run_single_fit(fold_key, "Group_D", "Extended_3845_Structural_Diversity", Whole_set_3845, d_indices, Whole_set_3845, Whole_set_1403, va_idx_1403, te_idx_1403, test_sub_df, info_d)
    all_results.append(res_d)

summary_df = pd.DataFrame(all_results)
summary_csv = out_dir / "gate3b_pilot_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print(f"\n==================================================================", flush=True)
print(f"  Decision Gate 3B 全部 8 组拟合圆满完成！", flush=True)
print(f"==================================================================", flush=True)
print(summary_df[['fold', 'group_id', 'n_train', 'il_combos_train', 'clipped_mae', 'clipped_r2', 'mean_bias_clipped', 'train_time_sec']].to_string(index=False), flush=True)

# 自动打包结果供用户一键下载
zip_out = ROOT / "gate3b_pilot_results.zip"
with zipfile.ZipFile(zip_out, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for f in out_dir.rglob('*'):
        if f.is_file():
            zipf.write(f, f.relative_to(ROOT))
print(f"\n  √ 结果已自动打包为: {zip_out} (包含完整预测与指标表，可直接下载)", flush=True)
