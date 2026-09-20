"""
step17_paper_master_table.py — Phase II-D Paper Master Table Generation
========================================================================
【功能】
1. 从权威源 paper_results/FINAL_FREEZE.json 读取冻结数据
2. 确保 JSON 标准兼容性 (清洗任何非标 NaN 为 None/null)
3. 生成论文唯一核心总表:
   - paper_results/table_main_generalization_boundary.csv
   - paper_results/table_main_generalization_boundary.tex
"""

import sys, os
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

print("=" * 80)
print("  STEP 17: GENERATING PAPER MASTER GENERALIZATION TABLE")
print("=" * 80)

freeze_json_path = PROJECT_ROOT / "paper_results" / "FINAL_FREEZE.json"

# 1. 严格清洗并标准化 FINAL_FREEZE.json 中的 NaN
with open(freeze_json_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 替换非法 JSON 的 NaN
clean_text = raw_text.replace(": NaN", ": null").replace(": nan", ": null")
freeze_data = json.loads(clean_text)

# 写回清洗后的合法 JSON (两处均更新)
with open(freeze_json_path, "w", encoding="utf-8") as f:
    json.dump(freeze_data, f, indent=2, ensure_ascii=False)

assembly_freeze = PROJECT_ROOT / "paper_results" / "phase2d_paper_assembly" / "FINAL_FREEZE.json"
if assembly_freeze.parent.exists():
    with open(assembly_freeze, "w", encoding="utf-8") as f:
        json.dump(freeze_data, f, indent=2, ensure_ascii=False)

print("  [✓] Cleaned non-standard NaN to valid null in FINAL_FREEZE.json.")

# 2. 构造主表数据
# 权威指标定义: 5-seed mean (M1: macro across 12 held-out; others: pooled test points)
f_axes = {a["axis"]: a for a in freeze_data["cross_axis_5seed_mean"]}
f_uq = {a["axis"]: a for a in freeze_data.get("cross_axis_ensemble_uq", [])}

axis_metadata = [
    ("M1", "12 HFC refrigerants (LORO)", "Refrigerant-level heterogeneity"),
    ("B1", "Fam-2 fluorosulfonate anions", "Limited descriptor benefit"),
    ("B2", "Fam-3 inorganic fluorides (BF4/PF6)", "Family-specific recovery"),
    ("L2", "4 unseen IL ion pairs", "Compositional interpolation"),
    ("HFO/HCFO", "1106 unsaturated refrigerant points", "Cross-family transfer with boundary cases"),
]

master_rows = []
for axis, held_out, diag in axis_metadata:
    fa = f_axes[axis]
    fu = f_uq.get(axis, {})
    sig_val = fu.get("Mred_mean_sigma")
    sig_str = f"{sig_val:.4f}" if sig_val is not None else "—"
    delta_str = f"{fa['delta_MAE_pct']:.1f}%"
    master_rows.append({
        "Axis": axis,
        "Held-out object": held_out,
        "N": fa["N_test"],
        "M0 MAE": round(fa["M0_MAE"], 4),
        "Mred MAE": round(fa["Mred_MAE"], 4),
        "delta_MAE": delta_str,
        "M0 R2": round(fa["M0_R2"], 4),
        "Mred R2": round(fa["Mred_R2"], 4),
        "Mred sigma": sig_str,
        "Diagnostic": diag
    })

df_master = pd.DataFrame(master_rows)

csv_out = PROJECT_ROOT / "paper_results" / "table_main_generalization_boundary.csv"
df_master.to_csv(csv_out, index=False, encoding='utf-8-sig')
print(f"  [✓] Master CSV saved: {csv_out.resolve()}")

# 3. 生成 LaTeX 源码
latex_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\small",
    r"\caption{\textbf{Comprehensive cross-axis generalization diagnostics across five distribution shifts.} Performance is reported as the five-seed mean. For M1, metrics represent macro-averages across the 12 held-out refrigerants; for B1, B2, L2, and HFO/HCFO, metrics are pooled over all test points within each seed. $\Delta\text{MAE}$ denotes the relative error change from $M_0$ to $M_{\rm reduced}$. M1 does not expose a directly verified global $\bar{\sigma}$ in the frozen summary.}",
    r"\label{tab:main_generalization_boundary}",
    r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrrl@{}}",
    r"\toprule",
    r"OOD Axis & Held-out Target & $N_{\rm test}$ & $M_0$ MAE & $M_{\rm red}$ MAE & $\Delta\text{MAE}$ & $M_0$ $R^2$ & $M_{\rm red}$ $R^2$ & $M_{\rm red}$ $\bar{\sigma}$ & Diagnostic Summary \\",
    r"\midrule"
]

for r in master_rows:
    sigma_str = r['Mred sigma'] if r['Mred sigma'] != "—" else r"\text{---}"
    latex_lines.append(
        f"{r['Axis']} & {r['Held-out object']} & {r['N']} & {r['M0 MAE']:.4f} & {r['Mred MAE']:.4f} & {r['delta_MAE']} & {r['M0 R2']:.4f} & {r['Mred R2']:.4f} & {sigma_str} & {r['Diagnostic']} \\\\"
    )

latex_lines.extend([
    r"\bottomrule",
    r"\end{tabular*}",
    r"\end{table*}"
])

tex_out = PROJECT_ROOT / "paper_results" / "table_main_generalization_boundary.tex"
with open(tex_out, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_lines))
print(f"  [✓] Master LaTeX saved: {tex_out.resolve()}")

print("\n--- MASTER TABLE PREVIEW ---")
print(df_master.to_string(index=False))
