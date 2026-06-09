"""
检查数据中是否有 Br (溴) 相关的阴阳离子或制冷剂
"""
import pandas as pd

excel_path = '../ZLJ_DATA.xlsx'
sheets = ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']

print("="*80)
print("检查 Br (溴) 在数据中的来源")
print("="*80)

br_in_anions = 0
br_in_cations = 0
br_in_refrigerants = 0

for sheet_name in sheets:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=2)
    
    # 检查阴离子
    br_anion_count = df['IL anion'].str.contains('Br', na=False, case=False).sum()
    if br_anion_count > 0:
        br_in_anions += br_anion_count
        print(f"\n{sheet_name} - 阴离子中含 Br: {br_anion_count} 条")
        print("  阴离子列表:")
        for anion in df[df['IL anion'].str.contains('Br', na=False, case=False)]['IL anion'].unique():
            count = len(df[df['IL anion'] == anion])
            print(f"    {anion}: {count} 条")
    
    # 检查阳离子
    br_cation_count = df['IL cation'].str.contains('Br', na=False, case=False).sum()
    if br_cation_count > 0:
        br_in_cations += br_cation_count
        print(f"\n{sheet_name} - 阳离子中含 Br: {br_cation_count} 条")
    
    # 检查制冷剂
    br_ref_count = df['Refrigerant'].str.contains('Br', na=False, case=False).sum()
    if br_ref_count > 0:
        br_in_refrigerants += br_ref_count
        print(f"\n{sheet_name} - 制冷剂中含 Br: {br_ref_count} 条")
        print("  制冷剂列表:")
        for ref in df[df['Refrigerant'].str.contains('Br', na=False, case=False)]['Refrigerant'].unique():
            count = len(df[df['Refrigerant'] == ref])
            print(f"    {ref}: {count} 条")

print(f"\n{'='*80}")
print(f"总计:")
print(f"  阴离子中含 Br: {br_in_anions} 条")
print(f"  阳离子中含 Br: {br_in_cations} 条")
print(f"  制冷剂中含 Br: {br_in_refrigerants} 条")
print(f"  Br 总样本数: {br_in_anions + br_in_cations + br_in_refrigerants}")
print(f"{'='*80}")
