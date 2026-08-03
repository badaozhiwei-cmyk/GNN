# -*- coding: utf-8 -*-
"""量化分析：Scheme 3 中阳离子泄露的具体影响 + FEP 独立审视"""
import pandas as pd

dfs = []
for sheet in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']:
    try:
        dfs.append(pd.read_excel('ZLJ_DATA.xlsx', sheet_name=sheet, skiprows=2))
    except:
        pass
df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])
df['Anion'] = df['IL anion'].astype(str).str.strip()
df['Cation'] = df['IL cation'].astype(str).str.strip()

# ── Scheme 3 阳离子泄露量化 ──
f2_anions = ['[OTf]', '[TFES]', '[TPES]', '[TTES]', '[HFPS]', '[PFBS]', '[FS]']
test3 = df[df['Anion'].isin(f2_anions)]
train3 = df[~df['Anion'].isin(f2_anions)]
train3_cations = set(train3['Cation'].unique())

print("=" * 70)
print("  Scheme 3: F2 测试集中各阳离子的数据量")
print("=" * 70)
print("F2 测试集总量:", len(test3), "条")
print()
for cat in sorted(test3['Cation'].unique()):
    n = len(test3[test3['Cation'] == cat])
    seen = "SEEN" if cat in train3_cations else "UNSEEN"
    pct = n / len(test3) * 100
    print("  %-20s  N=%4d  (%5.1f%%)  [%s]" % (cat, n, pct, seen))

unseen3 = test3[~test3['Cation'].isin(train3_cations)]
print()
print("未见阳离子总数据: %d / %d = %.1f%%" % (len(unseen3), len(test3), len(unseen3)/len(test3)*100))

# ── Scheme 2 阳离子泄露量化 ──
o4h6_anions = ['[Ac]', '[PFP]', '[MeSO4]', '[Et2PO4]', '[Pr]', '[Pe]', '[Cl]', '[I]', '[SCN]']
test2 = df[df['Anion'].isin(o4h6_anions)]
train2 = df[~df['Anion'].isin(o4h6_anions)]
train2_cations = set(train2['Cation'].unique())

print()
print("=" * 70)
print("  Scheme 2: O4+H6 测试集中各阳离子的数据量")
print("=" * 70)
print("O4+H6 测试集总量:", len(test2), "条")
print()
for cat in sorted(test2['Cation'].unique()):
    n = len(test2[test2['Cation'] == cat])
    seen = "SEEN" if cat in train2_cations else "UNSEEN"
    pct = n / len(test2) * 100
    print("  %-20s  N=%4d  (%5.1f%%)  [%s]" % (cat, n, pct, seen))

unseen2 = test2[~test2['Cation'].isin(train2_cations)]
print()
print("未见阳离子总数据: %d / %d = %.1f%%" % (len(unseen2), len(test2), len(unseen2)/len(test2)*100))

# ── FEP 独立分析 ──
print()
print("=" * 70)
print("  FEP 独立分析")
print("=" * 70)
fep = df[df['Anion'] == '[FEP]']
bf4 = df[df['Anion'] == '[BF4]']
pf6 = df[df['Anion'] == '[PF6]']
print("FEP 总数据: %d 条, 平均溶解度: %.4f" % (len(fep), fep['x1'].mean()))
print("BF4 总数据: %d 条, 平均溶解度: %.4f" % (len(bf4), bf4['x1'].mean()))
print("PF6 总数据: %d 条, 平均溶解度: %.4f" % (len(pf6), pf6['x1'].mean()))
print("FEP 搭配阳离子:", sorted(fep['Cation'].unique()))
print("FEP 搭配制冷剂:", sorted(fep['Refrigerant'].astype(str).str.strip().unique()))

# ── 如果把 FEP 从 F3 移到 F2，Scheme 3 会怎样？ ──
print()
print("=" * 70)
print("  如果 FEP 并入 F2 测试集 (Scheme 3 修正版)")
print("=" * 70)
f2_plus_fep = f2_anions + ['[FEP]']
test3b = df[df['Anion'].isin(f2_plus_fep)]
train3b = df[~df['Anion'].isin(f2_plus_fep)]
print("修正后训练集: %d 条 (%.1f%%)" % (len(train3b), len(train3b)/len(df)*100))
print("修正后测试集: %d 条 (%.1f%%)" % (len(test3b), len(test3b)/len(df)*100))
print("修正后测试集平均溶解度: %.4f" % test3b['x1'].mean())
print("修正后训练集平均溶解度: %.4f" % train3b['x1'].mean())
