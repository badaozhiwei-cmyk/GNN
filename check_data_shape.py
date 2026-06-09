import numpy as np

data = np.load('processed_tri_data/data.npy', allow_pickle=True)
label = np.load('processed_tri_data/label.npy', allow_pickle=True)

print("data[100] 长度:", len(data[100]))
for i, val in enumerate(data[100]):
    print(f"  元素 {i}: 类型为 {type(val)}")
    if isinstance(val, (list, tuple)):
        print(f"    长度: {len(val)}")
    elif isinstance(val, np.ndarray):
        print(f"    Shape: {val.shape}")
    else:
        print(f"    值: {val}")
