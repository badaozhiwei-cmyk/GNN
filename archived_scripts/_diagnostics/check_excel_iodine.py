"""
快速扫描 Excel 文件，查找含碘(I)和溴(Br)的 SMILES
"""

import openpyxl
import pandas as pd
from collections import Counter
import os

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
excel_path = os.path.join(project_root, 'ZLJ_DATA.xlsx')

try:
    # 读取所有 sheet
    xls = pd.ExcelFile(excel_path)
    print(f"✓ Excel 文件已加载")
    print(f"Sheet 列表: {xls.sheet_names}\n")
    
    all_smiles = []
    sheet_stats = {}
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # 查找 SMILES 列
        smiles_col = None
        for col in df.columns:
            if 'smiles' in col.lower():
                smiles_col = col
                break
        
        if smiles_col is None:
            print(f"⚠️  Sheet '{sheet_name}' 没有找到 SMILES 列")
            continue
        
        smiles_list = df[smiles_col].dropna().tolist()
        all_smiles.extend(smiles_list)
        
        # 统计这个 sheet 中的 I/Br
        i_count = sum(1 for s in smiles_list if 'I' in s)
        br_count = sum(1 for s in smiles_list if 'Br' in s)
        
        sheet_stats[sheet_name] = {
            'total': len(smiles_list),
            'with_I': i_count,
            'with_Br': br_count
        }
    
    # 打印统计
    print("="*60)
    print("📊 SMILES 中的 I/Br 分布")
    print("="*60)
    
    total_samples = 0
    total_i = 0
    total_br = 0
    
    for sheet_name, stats in sheet_stats.items():
        print(f"\n{sheet_name}:")
        print(f"  总数: {stats['total']}")
        print(f"  含 I 的 SMILES: {stats['with_I']} 个 ({stats['with_I']/stats['total']*100:.1f}%)")
        print(f"  含 Br 的 SMILES: {stats['with_Br']} 个 ({stats['with_Br']/stats['total']*100:.1f}%)")
        
        total_samples += stats['total']
        total_i += stats['with_I']
        total_br += stats['with_Br']
    
    print("\n" + "="*60)
    print(f"总计: {total_samples} 条 SMILES")
    print(f"  含 I: {total_i} 条")
    print(f"  含 Br: {total_br} 条")
    print("="*60)
    
    # 找出含 I 或 Br 的具体 SMILES
    if total_i > 0 or total_br > 0:
        print("\n🔍 含 I 的 SMILES (前5个):")
        i_smiles = [s for s in all_smiles if 'I' in s]
        for i, smi in enumerate(i_smiles[:5]):
            print(f"  {i+1}. {smi}")
        
        print("\n🔍 含 Br 的 SMILES (前5个):")
        br_smiles = [s for s in all_smiles if 'Br' in s]
        for i, smi in enumerate(br_smiles[:5]):
            print(f"  {i+1}. {smi}")

except FileNotFoundError:
    print(f"❌ 文件不存在: {excel_path}")
except Exception as e:
    print(f"❌ 错误: {e}")
