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
master_rows = [
    {
        "Axis": "M1",
        "Held-out object": "12 HFC refrigerants (LORO)",
        "N": 1403,
        "M0 MAE": 0.1049,
        "Mred MAE": 0.0666,
        "delta_MAE": "-36.5%",
        "M0 R2": -0.4158,
        "Mred R2": 0.3824,
        "Mred sigma": "—",
        "Diagnostic": "Refrigerant-level heterogeneity"
    },
    {
        "Axis": "B1",
        "Held-out object": "Fam-2 fluorosulfonate anions",
        "N": 513,
        "M0 MAE": 0.0304,
        "Mred MAE": 0.0302,
        "delta_MAE": "-0.7%",
        "M0 R2": 0.9265,
        "Mred R2": 0.9285,
        "Mred sigma": "0.0092",
        "Diagnostic": "Limited descriptor benefit"
    },
    {
        "Axis": "B2",
        "Held-out object": "Fam-3 inorganic fluorides (BF4/PF6)",
        "N": 598,
        "M0 MAE": 0.0534,
        "Mred MAE": 0.0483,
        "delta_MAE": "-9.6%",
        "M0 R2": 0.7931,
        "Mred R2": 0.8333,
        "Mred sigma": "0.0110",
        "Diagnostic": "Family-specific recovery"
    },
    {
        "Axis": "L2",
        "Held-out object": "4 unseen IL ion pairs",
        "N": 374,
        "M0 MAE": 0.0334,
        "Mred MAE": 0.0293,
        "delta_MAE": "-12.2%",
        "M0 R2": 0.8878,
        "Mred R2": 0.9126,
        "Mred sigma": "0.0142",
        "Diagnostic": "Compositional interpolation"
    },
    {
        "Axis": "HFO/HCFO",
        "Held-out object": "1106 unsaturated refrigerant points",
        "N": 1106,
        "M0 MAE": 0.0468,
        "Mred MAE": 0.0326,
        "delta_MAE": "-30.4%",
        "M0 R2": 0.4920,
        "Mred R2": 0.6926,
        "Mred sigma": "0.0144",
        "Diagnostic": "Cross-family transfer with boundary cases"
    }
]

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
