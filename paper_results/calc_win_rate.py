import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

df = pd.read_csv('paper_results/recovered_overnight_results.csv')
m0 = df[df['Mode']=='M0'].set_index('Target')
mr = df[df['Mode']=='Mreduced'].set_index('Target')

rows = []
for t in m0.index:
    d_mae = m0.loc[t, 'MAE_mean'] - mr.loc[t, 'MAE_mean']
    d_r2 = mr.loc[t, 'R2_mean'] - m0.loc[t, 'R2_mean']
    rows.append({
        'Refrigerant': t.replace('loro_', ''),
        'MAE_M0': m0.loc[t, 'MAE_mean'],
        'MAE_Mreduced': mr.loc[t, 'MAE_mean'],
        'Delta_MAE (M0 - Mred)': d_mae,
        'R2_M0': m0.loc[t, 'R2_mean'],
        'R2_Mreduced': mr.loc[t, 'R2_mean'],
        'Win': 'YES (改善)' if d_mae > 0 else 'NO (退化)'
    })
res = pd.DataFrame(rows)
wins = sum(res['Win'].str.contains('YES'))
total = len(res)
print("=" * 85)
print(f"🏆 Mreduced 相对 M0 胜率: {wins}/{total} = {wins/total*100:.1f}%")
print("=" * 85)
print(res.to_string(index=False))
res.to_csv('paper_results/table_mreduced_win_analysis.csv', index=False)
