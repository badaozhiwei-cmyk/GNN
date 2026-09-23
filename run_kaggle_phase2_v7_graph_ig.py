"""
run_kaggle_phase2_v7_graph_ig.py — Master One-Click Kaggle GPU Controller for Phase 2 Graph-IG
==============================================================================================
【在 Kaggle Notebook (开启 T4 GPU) 中执行】:
    %cd /kaggle/working
    !git clone --depth 1 https://github.com/badaozhiwei-cmyk/GNN.git
    %cd /kaggle/working/GNN
    !git pull origin main

    !pip install -q torch_geometric rdkit

    # 一键启动 430 次正式批跑 (GPU 上仅需 ~1-2 分钟):
    !python run_kaggle_phase2_v7_graph_ig.py

【流水线说明】:
1. 自动检测 GPU 硬件与 CUDA 环境 (优先使用 T4 GPU 加速);
2. 自动检索 /kaggle/input 或本地挂载的 V7-A 与 V7-B 检查点及 scalers;
3. 调度执行 phase2_v7_graph_ig.py 完成 430 次 Graph-IG 归因 (50 步黎曼积分 + Refri Top-1 掩码);
4. 严格校验 Gate A ~ Gate E 质量门禁;
5. 自动将全部 6 项产物和 Provenance JSON 打包为 v7_phase2_attribution_results.zip，供一键下载。
"""

from __future__ import annotations
import os
import sys
import time
import zipfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

print("=" * 85)
print("  KAGGLE GPU: MASTER PHASE 2 GRAPH-IG ATTRIBUTION CONTROLLER")
print("=" * 85)


def auto_detect_checkpoints() -> Tuple[Optional[Path], Optional[Path]]:
    """
    Finds V7-A and V7-B checkpoint directories across Kaggle mount points and local dirs.
    Handles unzipping if zip files are found in /kaggle/input or /kaggle/working.
    """
    v7a_dir: Optional[Path] = None
    v7b_dir: Optional[Path] = None

    # 1. Search for existing extracted directories
    search_roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        ROOT / "v7_shadow_experiment" / "checkpoints",
        ROOT / "checkpoints",
        Path(r"D:\折腾\v7_pilot_gpu_results\v7_shadow_experiment\checkpoints"),
        Path(r"D:\折腾\v7_pilot_gpu_results2\v7_shadow_experiment\checkpoints"),
    ]

    for s_root in search_roots:
        if not s_root.exists():
            continue

        # Look for V7-A
        if not v7a_dir:
            for p in s_root.rglob("*V7-A*"):
                if p.is_dir() and (p / "best_seed_42.pth").exists():
                    v7a_dir = p
                    break

        # Look for V7-B
        if not v7b_dir:
            for p in s_root.rglob("*V7-B*"):
                if p.is_dir() and (p / "best_seed_42.pth").exists():
                    v7b_dir = p
                    break

    # 2. If not found, check if zip files exist to extract
    if not v7a_dir or not v7b_dir:
        for zip_p in list(Path("/kaggle/input").rglob("*.zip")) + list(Path("/kaggle/working").rglob("*.zip")):
            zname = zip_p.name.lower()
            if "v7_pilot" in zname or "gpu_results" in zname or "v7" in zname:
                print(f"[*] Found candidate checkpoint archive: {zip_p}")
                extract_target = Path("/kaggle/working/extracted_checkpoints")
                extract_target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_p, "r") as zf:
                    zf.extractall(extract_target)
                print(f"    Extracted to: {extract_target}")

                for p in extract_target.rglob("*V7-A*"):
                    if p.is_dir() and (p / "best_seed_42.pth").exists():
                        v7a_dir = p
                for p in extract_target.rglob("*V7-B*"):
                    if p.is_dir() and (p / "best_seed_42.pth").exists():
                        v7b_dir = p

    return v7a_dir, v7b_dir


def main():
    import torch
    is_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    print(f"\n[硬件检测] 计算设备: {device_name} (CUDA Available: {is_cuda})")
    if not is_cuda:
        print("  ⚠️ 警告: 未检测到 GPU 加速！在 Kaggle 中请在 Notebook Settings -> Accelerator 开启 GPU T4 x2。")
    else:
        print(f"  ✓ GPU 加速已就绪: {device_name} (预计 430 次评估总耗时 ~1-2 分钟)")

    # 1. 检查点定位
    print("\n>>> [STEP 1/3] 正在检索 V7-A 与 V7-B 权重与 Scaler...")
    v7a_dir, v7b_dir = auto_detect_checkpoints()

    if v7a_dir:
        print(f"  ✓ 锁定 V7-A 检查点目录: {v7a_dir}")
    else:
        print("  ❌ 未找到 V7-A 检查点目录！请检查 Kaggle Notebook 是否添加了 V7-A 训练结果输入。")

    if v7b_dir:
        print(f"  ✓ 锁定 V7-B 检查点目录: {v7b_dir}")
    else:
        print("  ❌ 未找到 V7-B 检查点目录！请检查 Kaggle Notebook 是否添加了 V7-B 训练结果输入。")

    if not v7a_dir or not v7b_dir:
        print("\n[使用提示]:")
        print("  若检查点在其他目录，请通过参数指定，例如:")
        print("    python run_kaggle_phase2_v7_graph_ig.py --v7a_dir /kaggle/input/.../V7-A --v7b_dir /kaggle/input/.../V7-B")
        sys.exit(1)

    # 2. 调度执行 430 次全量批跑
    out_dir = ROOT / "v7_shadow_experiment" / "results_attribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "v7_shadow_experiment/phase2_v7_graph_ig.py",
        "--run_production_batch",
        "--steps", "50",
        "--v7a_dir", str(v7a_dir),
        "--v7b_dir", str(v7b_dir),
        "--out_dir", str(out_dir),
    ]

    print("\n>>> [STEP 2/3] 正在启动 430 次全量 Graph-IG 批跑与 Gate A~E 门禁审计...")
    print("    CMD: " + " ".join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd)
    t_elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"\n❌ [ERROR] 批跑异常退出，返回码: {res.returncode}")
        sys.exit(res.returncode)

    print(f"\n✅ [SUCCESS] 430 次评估全部成功完成！总用时: {t_elapsed:.1f} 秒 ({t_elapsed/60:.2f} 分钟)")

    # 3. 自动打包产物
    print("\n" + "=" * 85)
    print(">>> [STEP 3/3] 正在打包 Phase 2 结果工件至 zip 文件供下载...")
    print("=" * 85)

    zip_p = Path("/kaggle/working/v7_phase2_attribution_results.zip") if Path("/kaggle/working").exists() \
            else ROOT / "v7_phase2_attribution_results.zip"

    files_to_pack = [
        out_dir / "graph_smiles_alignment_audit.csv",
        out_dir / "v7_graph_attribution_atoms.csv",
        out_dir / "v7_graph_attribution_edges.csv",
        out_dir / "v7_graph_attribution_groups.csv",
        out_dir / "v7_graph_attribution_faithfulness.csv",
        out_dir / "v7_graph_attribution_stability.csv",
        out_dir / "v7_graph_attribution_provenance.json",
    ]

    with zipfile.ZipFile(zip_p, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_pack:
            if f.exists():
                arcname = f"results_attribution/{f.name}"
                zf.write(f, arcname)
                print(f"  + {arcname} ({f.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"  ! 警告: 文件缺失: {f}")

    zip_mb = zip_p.stat().st_size / 1024 / 1024
    print(f"\n[✓] 产物已成功打包至: {zip_p} ({zip_mb:.2f} MB)")
    print("=" * 85)
    print("  请直接在 Kaggle Notebook 输出区 (Output) 点击下载 'v7_phase2_attribution_results.zip'！")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
