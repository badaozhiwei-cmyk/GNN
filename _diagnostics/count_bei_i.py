"""
统计 [BEI] 和 [I] 阴离子对应的样本数
"""
import pandas as pd

excel_path = '../ZLJ_DATA.xlsx'
sheets = ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']

print("="*80)
print("统计 [BEI] 和 [I] 阴离子的样本分布")
print("="*80)

bei_count = 0
i_count = 0

for sheet_name in sheets:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=2)
    
    sheet_bei = len(df[df['IL anion'].str.strip() == '[BEI]'])
    sheet_i = len(df[df['IL anion'].str.strip() == '[I]'])
    
    bei_count += sheet_bei
    i_count += sheet_i
    
    if sheet_bei > 0 or sheet_i > 0:
        print(f"\n{sheet_name}:")
        if sheet_bei > 0:
            print(f"  [BEI]: {sheet_bei} 条")
        if sheet_i > 0:
            print(f"  [I]: {sheet_i} 条")

print(f"\n{'='*80}")
print(f"总计:")
print(f"  [BEI] (bis(pentafluoroethylsulfonyl)imide): {bei_count} 条")
print(f"  [I] (iodide 碘离子): {i_count} 条")
print(f"{'='*80}")

# 显示[I]样本的详细信息
if i_count > 0:
    print(f"\n[I] 样本详情:")
    for sheet_name in sheets:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=2)
        i_samples = df[df['IL anion'].str.strip() == '[I]']
        if len(i_samples) > 0:
            print(f"\n  在 {sheet_name} 中的样本 ({len(i_samples)} 条):")
            for idx, row in i_samples.iterrows():
                print(f"    阳离子: {row['IL cation']}, 制冷剂: {row['Refrigerant']}, T={row['T (K)']:.1f}K, P={row['P (MPa)']:.4f}")
