import numpy as np

ELEMENT_SYMBOL = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

data = np.load('processed_tri_data/data.npy', allow_pickle=True)
label = np.load('processed_tri_data/label.npy', allow_pickle=True)

row = data[100]
cation_graph = row[0]
anion_graph = row[1]
refrigerant_graph = row[2]
T = row[3]
P = row[4]
target_solubility = label[100]

print("=== 样本 100 基础数据 ===")
print(f"温度 T (K): {T}")
print(f"压力 P (MPa): {P}")
print(f"实测溶解度 x1: {target_solubility}")

# 提取原子序数
c_atomic_nums = [node[0] for node in cation_graph[0]]
a_atomic_nums = [node[0] for node in anion_graph[0]]
r_atomic_nums = [node[0] for node in refrigerant_graph[0]]

c_symbols = [ELEMENT_SYMBOL.get(num, f"Z{num}") for num in c_atomic_nums]
a_symbols = [ELEMENT_SYMBOL.get(num, f"Z{num}") for num in a_atomic_nums]
r_symbols = [ELEMENT_SYMBOL.get(num, f"Z{num}") for num in r_atomic_nums]

print("\n=== 分子组成原子列表 ===")
print(f"阳离子 (原子数={len(c_symbols)}): {c_symbols}")
print(f"阴离子 (原子数={len(a_symbols)}): {a_symbols}")
print(f"制冷剂 (原子数={len(r_symbols)}): {r_symbols}")

print("\n=== 拼接后图节点全局索引映射 ===")
idx = 0
for i, sym in enumerate(c_symbols):
    print(f"原子索引 {idx:2d} -> {sym} (阳离子 Cation)")
    idx += 1
for i, sym in enumerate(a_symbols):
    print(f"原子索引 {idx:2d} -> {sym} (阴离子 Anion)")
    idx += 1
for i, sym in enumerate(r_symbols):
    print(f"原子索引 {idx:2d} -> {sym} (制冷剂 Refrigerant)")
    idx += 1

print(f"\n拼接后的全部真实原子数: {idx}")
print(f"加全局虚拟节点后的原子索引 {idx} -> 虚拟节点 (Global Node)")
