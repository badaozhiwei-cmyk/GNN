import numpy as np
from collections import defaultdict

print("=" * 70)
print("快速诊断：检查 I 和 Br 在数据中的分布")
print("=" * 70)

# 加载数据
data = np.load('processed_tri_data/data.npy', allow_pickle=True)
label = np.load('processed_tri_data/label.npy', allow_pickle=True)

print(f"\n✓ 数据加载成功，共 {len(data)} 条样本")

# 检查元素分布
element_count = defaultdict(int)
element_samples = defaultdict(list)  # 记录哪些样本含特定元素
sample_idx_with_I_Br = []

for idx, sample in enumerate(data):
    c_graph, a_graph, r_graph = sample[:3]
    
    # 检查三个图的所有原子
    for graph_idx, graph in enumerate([c_graph, a_graph, r_graph]):
        if graph is None or len(graph[0]) == 0:
            continue
        for atom_features in graph[0]:
            atomic_num = atom_features[0]
            element_count[atomic_num] += 1
            
            # 记录含 I(53) 或 Br(35) 的样本
            if atomic_num in [35, 53]:
                element_samples[atomic_num].append(idx)
                if idx not in sample_idx_with_I_Br:
                    sample_idx_with_I_Br.append(idx)

print(f"\n【元素分布统计】")
print(f"  碘(I, 原子序数 53)：{element_count[53]} 个 → 在 {len(set(element_samples[53]))} 条样本中出现")
print(f"  溴(Br, 原子序数 35)：{element_count[35]} 个 → 在 {len(set(element_samples[35]))} 条样本中出现")
print(f"  氟(F, 原子序数 9)：{element_count[9]} 个")
print(f"  氯(Cl, 原子序数 17)：{element_count[17]} 个")
print(f"  碳(C, 原子序数 6)：{element_count[6]} 个")
print(f"  氮(N, 原子序数 7)：{element_count[7]} 个")

# 检查 I/Br 样本的溶解度
print(f"\n【含 I/Br 样本的溶解度分析】")
if len(sample_idx_with_I_Br) > 0:
    i_br_labels = label[sample_idx_with_I_Br]
    print(f"  含 I/Br 样本数：{len(sample_idx_with_I_Br)}")
    print(f"  这些样本的平均溶解度：{i_br_labels.mean():.6f}")
    print(f"  这些样本的最大溶解度：{i_br_labels.max():.6f}")
    print(f"  这些样本的最小溶解度：{i_br_labels.min():.6f}")
else:
    print(f"  数据中没有找到含 I/Br 的样本！")

# 检查全体数据的溶解度
print(f"\n【全体样本溶解度统计】")
print(f"  全体样本数：{len(label)}")
print(f"  全体平均溶解度：{label.mean():.6f}")
print(f"  全体最大溶解度：{label.max():.6f}")
print(f"  全体最小溶解度：{label.min():.6f}")
print(f"  全体标准差：{label.std():.6f}")

# 删除 I/Br 后的对比
if len(sample_idx_with_I_Br) > 0:
    mask = np.ones(len(label), dtype=bool)
    mask[sample_idx_with_I_Br] = False
    label_without_iBr = label[mask]
    
    print(f"\n【删除 I/Br 样本后的统计】")
    print(f"  删除后样本数：{len(label_without_iBr)}")
    print(f"  删除后平均溶解度：{label_without_iBr.mean():.6f}")
    print(f"  删除后标准差：{label_without_iBr.std():.6f}")
    print(f"  平均值变化：{(label_without_iBr.mean() - label.mean()):.6f}")
    
    if len(set(element_samples[53])) > 0:
        print(f"\n【含 I 的具体样本】")
        for idx in list(set(element_samples[53]))[:3]:  # 显示前 3 个
            print(f"  样本 {idx}：溶解度 = {label[idx]:.6f}")

print("\n" + "=" * 70)
print("诊断完成！")
print("=" * 70)
