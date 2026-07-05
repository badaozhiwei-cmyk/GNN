"""
step1_anion_family_splitter.py
==============================
【目的】
  正式生成 Split A（随机划分）和 Split B（阴离子家族 OOD 划分）的
  训练/验证/测试索引，保存为 .npz 文件供 GAT 训练器直接加载。

【输入】
  index_with_anion.csv（Step 0 产出）

【输出】
  split_A_indices.npz   → 随机划分索引（Train 70% / Val 10% / Test 20%）
  split_B_indices.npz   → 阴离子家族 OOD 划分索引
  split_C_indices.npz   → 制冷剂类别划分索引（HFC↔HFO，供 Supplementary）
  split_summary.txt     → 三种划分的详细统计报告

【运行方法】
  python step1_anion_family_splitter.py

【关键设计说明】
  Split A：用固定 seed=42 的随机划分，保证可复现，且和现有文献划分方式一致。
  Split B：测试集 = F1_Sulfonimide + F2_FluoroAlkyl 家族（高吸收含氟阴离子），
           训练集再从剩余数据中随机切出 10% 作为验证集。
           如果测试集比例不在 15~25% 范围，脚本会自动调整并打印建议。
  Split C：两个方向：HFC→HFO 和 HFO→HFC，分别保存为独立键名。
"""

import os

# ── 自动定位项目根目录（脚本在 research_pipeline/ 子文件夹中）──────
import pathlib as _pl
ROOT = str(_pl.Path(__file__).resolve().parent.parent)
import os as _os; _os.chdir(ROOT)
# ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────
# 0. 读取索引映射表
# ──────────────────────────────────────────────────────────
CSV_PATH = 'index_with_anion.csv'
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        "找不到 index_with_anion.csv，请先运行 step0_verify_alignment.py"
    )

df = pd.read_csv(CSV_PATH)
N = len(df)
print(f"读取成功，共 {N} 条数据")

# ──────────────────────────────────────────────────────────
# 1. 阴离子家族映射（与 step0_5 保持完全一致）
# ──────────────────────────────────────────────────────────
ANION_FAMILY_MAP = {
    # F1: 含氟有机磺酸盐／磺酰亚胺 → Test
    'TF2N': 'F1', 'TFSI': 'F1', 'NTF2': 'F1',
    'OTF':  'F1', 'TFO':  'F1', 'TTES': 'F1',
    'HFPS': 'F1', 'PFBS': 'F1', 'TFES': 'F1',
    'TPES': 'F1', 'FS':   'F1',
    # F2: 氟代烷基磷酸盐 → Test
    'FEP':  'F2', 'BEI':  'F2', 'TMEM': 'F2', 'PFP': 'F2',
    # F3: 无机球形氟 → Train
    'BF4':  'F3', 'PF6':  'F3',
    # A1: 有机酸根 → Train
    'AC':   'A1', 'DCA':  'A1', 'SCN':  'A1',
    'PR':   'A1', 'PE':   'A1', 'ET2PO4': 'A1', 'TMPP': 'A1',
    # A2: 卤素/无机 → Train
    'CL':   'A2', 'BR':   'A2', 'I': 'A2', 'NO3': 'A2',
}

def assign_family(name):
    key = str(name).strip().upper().replace('[','').replace(']','').replace('-','')
    return ANION_FAMILY_MAP.get(key, 'Other')

df['family'] = df['anion'].apply(assign_family)

# ──────────────────────────────────────────────────────────
# 2. 制冷剂类型（用于 Split C）
# ──────────────────────────────────────────────────────────
def get_refri_type(sheet):
    s = str(sheet)
    if 'HFC' in s: return 'HFC'
    if 'HFO' in s: return 'HFO'
    return 'Other'

df['refri_type'] = df['sheet'].apply(get_refri_type)

all_idx = df['npy_idx'].values  # 全部索引（即 data.npy 里的行号）

report = []
report.append("=" * 60)
report.append("数据集划分方案汇总报告")
report.append("=" * 60)

# ──────────────────────────────────────────────────────────
# 3. Split A：随机划分（70 / 10 / 20）
# ──────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("【Split A：随机划分 70/10/20】")

SPLIT_SEED = 42
train_val_idx, test_A_idx = train_test_split(
    all_idx, test_size=0.20, random_state=SPLIT_SEED
)
train_A_idx, val_A_idx = train_test_split(
    train_val_idx, test_size=0.125, random_state=SPLIT_SEED
    # 0.125 × 0.80 = 0.10，最终比例：Train 70% / Val 10% / Test 20%
)

print(f"  Train: {len(train_A_idx)} ({len(train_A_idx)/N*100:.1f}%)")
print(f"  Val  : {len(val_A_idx)}   ({len(val_A_idx)/N*100:.1f}%)")
print(f"  Test : {len(test_A_idx)}  ({len(test_A_idx)/N*100:.1f}%)")

np.savez('split_A_indices.npz',
         train=train_A_idx,
         val=val_A_idx,
         test=test_A_idx)
print("  ✅ 已保存：split_A_indices.npz")

report.append("\n【Split A：随机划分】")
report.append(f"  Train: {len(train_A_idx)} ({len(train_A_idx)/N*100:.1f}%)")
report.append(f"  Val  : {len(val_A_idx)}   ({len(val_A_idx)/N*100:.1f}%)")
report.append(f"  Test : {len(test_A_idx)}  ({len(test_A_idx)/N*100:.1f}%)")
report.append(f"  说明 : seed={SPLIT_SEED}，完全随机，用于与文献公平比较")

# ──────────────────────────────────────────────────────────
# 4. Split B：阴离子家族 OOD 划分
# ──────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("【Split B：阴离子家族 OOD 划分】")

TEST_FAMILIES = {'F1', 'F2'}

test_B_mask  = df['family'].isin(TEST_FAMILIES)
train_B_pool = df.loc[~test_B_mask, 'npy_idx'].values
test_B_idx   = df.loc[test_B_mask,  'npy_idx'].values

test_B_pct = len(test_B_idx) / N * 100
print(f"  测试集（F1+F2）：{len(test_B_idx)} 条 ({test_B_pct:.1f}%)")

# 自动调整：如果比例不在 15~25%，给出明确提示
if test_B_pct < 15:
    print(f"\n  ⚠️  测试集比例 {test_B_pct:.1f}% 偏小（<15%）")
    print("      建议：将 DCA / SCN 家族也划入测试集")
    print("      → 已自动将 A1 中的 DCA 和 SCN 加入测试集")
    # 自动补充
    extra_mask = df['anion'].str.upper().str.replace('[','').str.replace(']','').isin(['DCA','SCN'])
    test_B_mask = test_B_mask | extra_mask
    test_B_idx  = df.loc[test_B_mask, 'npy_idx'].values
    train_B_pool = df.loc[~test_B_mask, 'npy_idx'].values
    test_B_pct  = len(test_B_idx) / N * 100
    print(f"      调整后测试集：{len(test_B_idx)} 条 ({test_B_pct:.1f}%)")

elif test_B_pct > 25:
    print(f"\n  ⚠️  测试集比例 {test_B_pct:.1f}% 偏大（>25%）")
    print("      建议：将 OTF/TTES 移回训练集，只保留 Tf2N 和 FEP 作测试")
    print("      → 已自动调整：测试集仅保留 TF2N 和 FEP")
    keep_test = {'TF2N', 'TFSI', 'NTF2', 'FEP'}
    test_B_mask2 = df['anion'].str.upper().str.replace('[','').str.replace(']','').str.replace('-','').isin(keep_test)
    test_B_idx   = df.loc[test_B_mask2, 'npy_idx'].values
    train_B_pool = df.loc[~test_B_mask2, 'npy_idx'].values
    test_B_pct   = len(test_B_idx) / N * 100
    print(f"      调整后测试集：{len(test_B_idx)} 条 ({test_B_pct:.1f}%)")
else:
    print(f"  ✅ 测试集比例 {test_B_pct:.1f}% 在合理范围（15~25%）内")

# 从 train_pool 里切出 10% 作为验证集
train_B_idx, val_B_idx = train_test_split(
    train_B_pool, test_size=0.10, random_state=SPLIT_SEED
)

print(f"  Train: {len(train_B_idx)} ({len(train_B_idx)/N*100:.1f}%)")
print(f"  Val  : {len(val_B_idx)}   ({len(val_B_idx)/N*100:.1f}%)")
print(f"  Test : {len(test_B_idx)}  ({len(test_B_idx)/N*100:.1f}%)")

# 打印测试集阴离子家族详情
test_B_families = df.loc[df['npy_idx'].isin(test_B_idx), 'anion'].value_counts()
print("\n  测试集阴离子详细分布：")
print(test_B_families.to_string())

np.savez('split_B_indices.npz',
         train=train_B_idx,
         val=val_B_idx,
         test=test_B_idx)
print("\n  ✅ 已保存：split_B_indices.npz")

report.append("\n【Split B：阴离子家族 OOD 划分】")
report.append(f"  测试集家族 : F1（含氟磺酸盐/磺酰亚胺） + F2（氟代烷基链）")
report.append(f"  Train: {len(train_B_idx)} ({len(train_B_idx)/N*100:.1f}%)")
report.append(f"  Val  : {len(val_B_idx)}   ({len(val_B_idx)/N*100:.1f}%)")
report.append(f"  Test : {len(test_B_idx)}  ({len(test_B_idx)/N*100:.1f}%)")
report.append(f"  说明 : 真正的 OOD 实验，测试集不含训练集阴离子家族")

# ──────────────────────────────────────────────────────────
# 5. Split C：制冷剂类别 OOD（两个方向）
# ──────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("【Split C：制冷剂 OOD 划分（HFC↔HFO）】")

hfc_idx   = df.loc[df['refri_type'] == 'HFC', 'npy_idx'].values
hfo_idx   = df.loc[df['refri_type'] == 'HFO', 'npy_idx'].values
other_idx = df.loc[df['refri_type'] == 'Other', 'npy_idx'].values

print(f"  HFC 样本：{len(hfc_idx)} ({len(hfc_idx)/N*100:.1f}%)")
print(f"  HFO 样本：{len(hfo_idx)} ({len(hfo_idx)/N*100:.1f}%)")
print(f"  Other  ：{len(other_idx)} ({len(other_idx)/N*100:.1f}%)")

# 方向1：HFC 训练 → HFO 测试
# 训练集 = HFC（+ Other 的部分），验证集从训练集切 10%，测试集 = HFO
train_C1_pool = np.concatenate([hfc_idx, other_idx])
train_C1_idx, val_C1_idx = train_test_split(
    train_C1_pool, test_size=0.10, random_state=SPLIT_SEED
)
test_C1_idx = hfo_idx

# 方向2：HFO 训练 → HFC 测试
train_C2_pool = np.concatenate([hfo_idx, other_idx])
train_C2_idx, val_C2_idx = train_test_split(
    train_C2_pool, test_size=0.10, random_state=SPLIT_SEED
)
test_C2_idx = hfc_idx

print(f"\n  方向1（HFC→HFO）: Train={len(train_C1_idx)}, Val={len(val_C1_idx)}, Test={len(test_C1_idx)}")
print(f"  方向2（HFO→HFC）: Train={len(train_C2_idx)}, Val={len(val_C2_idx)}, Test={len(test_C2_idx)}")

np.savez('split_C_indices.npz',
         # 方向1：HFC → HFO
         train_c1=train_C1_idx,
         val_c1=val_C1_idx,
         test_c1=test_C1_idx,
         # 方向2：HFO → HFC
         train_c2=train_C2_idx,
         val_c2=val_C2_idx,
         test_c2=test_C2_idx)
print("  ✅ 已保存：split_C_indices.npz")

report.append("\n【Split C：制冷剂 OOD 划分】")
report.append(f"  方向1 HFC→HFO: Train={len(train_C1_idx)}, Val={len(val_C1_idx)}, Test={len(test_C1_idx)}")
report.append(f"  方向2 HFO→HFC: Train={len(train_C2_idx)}, Val={len(val_C2_idx)}, Test={len(test_C2_idx)}")
report.append(f"  说明 : 用于 Supplementary 的制冷剂泛化实验")

# ──────────────────────────────────────────────────────────
# 6. 验证：三种划分的索引无重叠（Split A/B 内部）
# ──────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("【完整性校验】")

def check_no_overlap(name, train, val, test, total):
    all_used = np.concatenate([train, val, test])
    n_unique = len(np.unique(all_used))
    n_total  = len(train) + len(val) + len(test)
    overlap  = n_total - n_unique
    cover    = n_unique == total
    print(f"  {name}: 总={n_total}, 唯一={n_unique}, 重叠={overlap}, 全覆盖={cover}")
    assert overlap == 0,  f"❌ {name} 存在 Train/Val/Test 重叠！"
    assert cover,         f"❌ {name} 未覆盖全部 {total} 条数据（唯一={n_unique}）！"
    print(f"  ✅ {name} 无重叠，全覆盖")

check_no_overlap("Split A", train_A_idx, val_A_idx, test_A_idx, N)
check_no_overlap("Split B", train_B_idx, val_B_idx, test_B_idx, N)

# Split C 不要求全覆盖（Other 类别可能被丢弃），只检查方向内无重叠
def check_no_overlap_C(name, train, val, test):
    all_used = np.concatenate([train, val, test])
    n_unique = len(np.unique(all_used))
    n_total  = len(train) + len(val) + len(test)
    overlap  = n_total - n_unique
    print(f"  {name}: 总={n_total}, 唯一={n_unique}, 重叠={overlap}")
    assert overlap == 0, f"❌ {name} 存在重叠！"
    print(f"  ✅ {name} 无重叠")

check_no_overlap_C("Split C 方向1", train_C1_idx, val_C1_idx, test_C1_idx)
check_no_overlap_C("Split C 方向2", train_C2_idx, val_C2_idx, test_C2_idx)

# ──────────────────────────────────────────────────────────
# 7. 保存报告
# ──────────────────────────────────────────────────────────
report.append("\n" + "=" * 60)
report.append("所有划分校验通过 ✅")

with open('split_summary.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print("\n  📄 划分汇总报告已保存：split_summary.txt")
print("\n" + "=" * 60)
print("✅ Step 1 完成！产出文件：")
print("   split_A_indices.npz  （随机划分，与文献对比用）")
print("   split_B_indices.npz  （阴离子 OOD，论文核心）")
print("   split_C_indices.npz  （制冷剂 OOD，Supplementary）")
print("   split_summary.txt    （划分报告）")
print("\n下一步：运行 GAT_Runner_v4.py 加载上述索引文件训练模型")
print("=" * 60)
