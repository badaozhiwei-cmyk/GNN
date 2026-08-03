import os
import pandas as pd
import numpy as np

def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(ROOT, 'ZLJ_DATA.xlsx')
    
    if not os.path.exists(excel_path):
        print(f"Error: Could not find {excel_path}")
        return
        
    print(f"Loading {excel_path}...")
    dfs = []
    for sheet in ['Table S3. VLE HFCs', 'Table S4. VLE HFOs', 'Table S5. VLE Other']:
        try:
            tmp_df = pd.read_excel(excel_path, sheet_name=sheet, skiprows=2)
            dfs.append(tmp_df)
        except Exception as e:
            pass
            
    if not dfs:
        return
        
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['IL cation', 'IL anion', 'Refrigerant', 'T (K)', 'P (MPa)', 'x1'])
    
    print("\n" + "="*75)
    print(" PRIORITY 1: DATASET BIAS AUDIT")
    print("="*75)
    
    print("\n[Frequency Audit]")
    print(f"Total valid samples: {len(df)}")
    
    anion_counts = df['IL anion'].value_counts()
    print("\nTop 10 Anions by Frequency:")
    for a_name, count in anion_counts.head(10).items():
        print(f"  {a_name:<20}: {count:<6} ({count/len(df)*100:.1f}%)")
        
    cation_counts = df['IL cation'].value_counts()
    print("\nTop 10 Cations by Frequency:")
    for c_name, count in cation_counts.head(10).items():
        print(f"  {c_name:<20}: {count:<6} ({count/len(df)*100:.1f}%)")
        
    ref_counts = df['Refrigerant'].value_counts()
    print("\nTop 10 Refrigerants by Frequency:")
    for r_name, count in ref_counts.head(10).items():
        print(f"  {r_name:<20}: {count:<6} ({count/len(df)*100:.1f}%)")
        
    print("\n[Label Distribution Audit (Conditioned on Anion)]")
    # Group by anion and compute Mean and Var of 'x1'
    anion_stats = df.groupby('IL anion')['x1'].agg(['count', 'mean', 'var', 'max']).dropna()
    anion_stats = anion_stats[anion_stats['count'] >= 20].sort_values(by='mean', ascending=False)
    
    print(f"{'Anion':<20} | {'Count':<8} | {'Mean (x1)':<12} | {'Var (x1)':<12} | {'Max (x1)':<12}")
    print("-" * 75)
    for index, row in anion_stats.iterrows():
        print(f"{index:<20} | {int(row['count']):<8} | {row['mean']:<12.4f} | {row['var']:<12.4f} | {row['max']:<12.4f}")

if __name__ == '__main__':
    main()
