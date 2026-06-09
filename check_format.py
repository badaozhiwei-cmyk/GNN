import numpy as np

# 检查版本1
data1 = np.load('GNN_for_property_prediction/clean/data.npy', allow_pickle=True)
print("版本1 (GNN_for_property_prediction/clean/data.npy):")
print(f"  类型: {type(data1)}, shape: {data1.shape}")
print(f"  长度: {len(data1)}")
print(f"  第一个元素类型: {type(data1[0])}")

print("\n版本2 (processed_tri_data/data.npy):")
data2 = np.load('processed_tri_data/data.npy', allow_pickle=True)
print(f"  类型: {type(data2)}, shape: {data2.shape}")
print(f"  长度: {len(data2)}")
print(f"  第一个元素类型: {type(data2[0])}")
