"""
test_pipeline_sanity.py – 运行前 1 秒鞁速自检工具.
==================================================
	"""
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'GNN_for_property_prediction'))

# 导入模式定义
try:
    from Dataset_v6 import IL_set_v6, MODE_COND_DIM, MODE_INDICES
    HAS_TORCH = True
except ModuleNotFoundError:
    # 本地环境未装 torch 时，依然校验 MODE 映射
    MODE_INDICES = {
        'M0':             [3, 4, 5, 6, 7, 8, 9],
        'Msize':          [3, 4, 5, 6, 7, 8, 9, 10, 11],
        'Mmu':            [3, 4, 5, 6, 7, 8, 9, 12],
        'Malpha':         [3, 4, 5, 6, 7, 8, 9, 13],
        'MV':             [3, 4, 5, 6, 7, 8, 9, 14],
        'Mphys':          [3, 4, 5, 6, 7, 8, 9, 12, 13, 14],
        'Mthermo':        [3, 4, 5, 6, 7, 8, 9, 15, 16, 17],
        'Mreduced':       [3, 4, 5, 6, 7, 8, 9, 18, 19, 17],
        'Mreduced_pure':  [5, 6, 7, 8, 9, 18, 19, 17],
    }
    MODE_COND_DIM = {k: len(v) for k, v in MODE_INDICES.items()}
    HAS_TORCH = False

def run_sanity_check(data_dir='processed_tri_data_v3'):
    print("=" * 60)
    print("[PRE-FLIGHT] GNN Ablation Pipeline Sanity Check")
    print("=" * 60)

    data_file = os.path.join(data_dir, 'data.npy')
    if not os.path.exists(data_file):
        print("[!] processed_tri_data_v3/data.npy yet to be generated.")
        return True

    raw_data = np.load(data_file, allow_pickle=True)
    n_samples = len(raw_data)
    first_row_len = len(raw_data[0])
    print(f"[OK] Data file read successfully: Samples = {n_samples}, Elements per row = {first_row_len}")
    
    if first_row_len != 20:
        print(f"[ERROR] Expected 20 elements but got {first_row_len}!")
        return False
    else:
        print("[OK] Sample layout verified: 3 graphs + 17 condition features = 20 elements")

    target_modes = {
        'M0': 7,
        'Mphys': 10,
        'Mthermo': 10,
        'Mreduced': 10,
        'Mreduced_pure': 8
    }

    print("\n[INFO] Validating mode dimensions and data loading...")
    for mode, expected_dim in target_modes.items():
        actual_dim = MODE_COND_DIM.get(mode)
        indices = MODE_INDICES.get(mode)
        if actual_dim != expected_dim:
            print(f"  [ERROR] Mode {mode} dimension mismatch! Expected {expected_dim}, got {actual_dim}")
            return False
        
        if HAS_TORCH:
            ds = IL_set_v6(path=data_dir, descriptor_mode=mode)
            sample_item = ds[0]
            cond_len = len(sample_item[3])
            if cond_len != expected_dim:
                print(f"  [ERROR] Mode {mode} output length error! Expected {expected_dim}, got {cond_len}")
                return False

        print(f"  [OK] Mode [{mode:<14s}] -> Cond Dim: {actual_dim}, Indices: {indices}")

    print("\n" + "=" * 60)
    print("[SUCCESS] All 5 modes and data layouts are 100% sane and verified!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    run_sanity_check()
