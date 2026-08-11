"""Recompute Vega statistics; no conclusions are hard-coded."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "audit_outputs"
OUT.mkdir(parents=True, exist_ok=True)
VEGA = {'R41':1.851,'R32':1.978,'R23':1.649,'R161':1.940,'R152a':2.262,'R134a':2.058,'R125':1.563,'R245fa':1.549,'R236fa':1.982,'R227ea':1.456,'R1234yf':2.011,'R1234ze(E)':1.440,'R1336mzz(Z)':3.190,'R1233zd(E)':1.143}

def bootstrap(x, y, statistic, n=1000):
    rng = np.random.default_rng(42); values = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x)); values.append(statistic(x[idx], y[idx]))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]

df = pd.read_csv(ROOT / 'xTB_Physics_Descriptors.csv')
df = df[(df.Category == 'Refrigerant') & df.Molecule.isin(VEGA)].copy()
df['Vega_Dipole'] = df.Molecule.map(VEGA); df = df.dropna(subset=['Dipole_Debye'])
x, y = df.Vega_Dipole.to_numpy(), df.Dipole_Debye.to_numpy()
pr, pp = stats.pearsonr(x, y); sr, sp = stats.spearmanr(x, y); slope, intercept = np.polyfit(x, y, 1)
result = {'n': len(df), 'pearson_r': pr, 'pearson_p': pp, 'pearson_ci95': bootstrap(x,y,lambda a,b:stats.pearsonr(a,b)[0]), 'spearman_rho': sr, 'spearman_p': sp, 'spearman_ci95': bootstrap(x,y,lambda a,b:stats.spearmanr(a,b).statistic), 'bias_xTB_minus_Vega': float(np.mean(y-x)), 'MAE': float(np.mean(np.abs(y-x))), 'RMSE': float(np.sqrt(np.mean((y-x)**2))), 'slope': slope, 'intercept': intercept}
(OUT / 'vega_statistics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
z = np.linspace(min(x.min(),y.min())-.2, max(x.max(),y.max())+.2, 100); plt.figure(figsize=(6,6)); plt.scatter(x,y); plt.plot(z,z,'k--'); plt.plot(z,slope*z+intercept,'r-'); plt.xlabel('Vega dipole (D)'); plt.ylabel('xTB dipole (D)'); plt.tight_layout(); plt.savefig(OUT/'Vega_Dipole_Validation_rigorous.png', dpi=300)
print(json.dumps(result, indent=2))
