"""
检查 4444 行数据中的阴阳离子是否包含 I 或 Br
"""
import pandas as pd
from collections import Counter

excel_path = '../ZLJ_DATA.xlsx'
sheets = ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']

print("="*80)
print("检查 4444 行数据中的阴阳离子")
print("="*80)

all_cations = Counter()
all_anions = Counter()

for sheet_name in sheets:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=2)
    
    for _, row in df.iterrows():
        cation = str(row['IL cation']).strip()
        anion = str(row['IL anion']).strip()
        all_cations[cation] += 1
        all_anions[anion] += 1

print(f"\n📊 阳离子分布 ({len(all_cations)} 种):")
for cation, count in all_cations.most_common(10):
    has_i = 'I' in cation
    has_br = 'Br' in cation
    mark = ' ⚠️ 含I/Br' if (has_i or has_br) else ''
    print(f"  {cation:<30} {count:>5} 次{mark}")

print(f"\n📊 阴离子分布 ({len(all_anions)} 种):")
for anion, count in all_anions.most_common(15):
    has_i = 'I' in anion
    has_br = 'Br' in anion
    mark = ' ⚠️ 含I/Br' if (has_i or has_br) else ''
    print(f"  {anion:<30} {count:>5} 次{mark}")

print(f"\n{'='*80}")

# 查找包含I或Br的
i_cations = [c for c, _ in all_cations.items() if 'I' in c]
br_cations = [c for c, _ in all_cations.items() if 'Br' in c]
i_anions = [a for a, _ in all_anions.items() if 'I' in a]
br_anions = [a for a, _ in all_anions.items() if 'Br' in a]

if i_cations:
    print(f"\n✓ 含 I 的阳离子: {i_cations}")
if br_cations:
    print(f"✓ 含 Br 的阳离子: {br_cations}")
if i_anions:
    print(f"✓ 含 I 的阴离子: {i_anions}")
if br_anions:
    print(f"✓ 含 Br 的阴离子: {br_anions}")

if not (i_cations or br_cations or i_anions or br_anions):
    print("\n❌ 4444 行数据中，阴阳离子列都不包含 I 或 Br")

print(f"{'='*80}")
