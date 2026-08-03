"""
检查另一个Excel文件中的I/Br SMILES
"""
import pandas as pd
import os

files_to_check = [
    '1-s2.0-S2666952824000554-mmc1.xlsx',
    'ZLJ_DATA.xlsx'
]

for file_name in files_to_check:
    excel_path = os.path.join(os.path.dirname(__file__), '..', file_name)
    
    print(f"\n{'='*70}")
    print(f"检查文件: {file_name}")
    print(f"{'='*70}")
    
    if not os.path.exists(excel_path):
        print(f"❌ 文件不存在: {excel_path}")
        continue
    
    try:
        xls = pd.ExcelFile(excel_path)
        print(f"✓ 文件加载成功")
        print(f"Sheet 列表 ({len(xls.sheet_names)} 个):")
        for i, sheet in enumerate(xls.sheet_names[:5]):  # 只显示前5个
            print(f"  {i+1}. {sheet}")
        if len(xls.sheet_names) > 5:
            print(f"  ... 还有 {len(xls.sheet_names) - 5} 个 Sheet")
        
        # 尝试找SMILES列
        all_smiles = []
        total_rows = 0
        
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                total_rows += len(df)
                
                # 查找可能的SMILES列
                for col in df.columns:
                    if any(x in col.lower() for x in ['smiles', 'smi', 'molecule']):
                        smiles_list = df[col].dropna().tolist()
                        all_smiles.extend(smiles_list)
                        print(f"  找到 SMILES 列: '{col}' in Sheet '{sheet_name}' ({len(smiles_list)} 条)")
            except:
                pass
        
        print(f"\n数据统计:")
        print(f"  总行数: {total_rows}")
        print(f"  收集到的 SMILES 总数: {len(all_smiles)}")
        
        if all_smiles:
            i_count = sum(1 for s in all_smiles if 'I' in str(s))
            br_count = sum(1 for s in all_smiles if 'Br' in str(s))
            
            print(f"\n  含 I 的 SMILES: {i_count} 条 ({i_count/len(all_smiles)*100:.1f}%)")
            print(f"  含 Br 的 SMILES: {br_count} 条 ({br_count/len(all_smiles)*100:.1f}%)")
            
            if i_count > 0:
                i_smiles = [s for s in all_smiles if 'I' in str(s)]
                print(f"\n  含 I 的 SMILES 示例 (前3个):")
                for smi in i_smiles[:3]:
                    print(f"    - {smi}")
            
            if br_count > 0:
                br_smiles = [s for s in all_smiles if 'Br' in str(s)]
                print(f"\n  含 Br 的 SMILES 示例 (前3个):")
                for smi in br_smiles[:3]:
                    print(f"    - {smi}")
    
    except Exception as e:
        print(f"❌ 错误: {e}")

print(f"\n{'='*70}")
print("✓ 检查完成")
print(f"{'='*70}")
