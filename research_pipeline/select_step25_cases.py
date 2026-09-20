"""
select_step25_cases.py — Phase III-F: Step 25 Case Selection Manifest Generator
==============================================================================
严格按用户定盘规范生成 results_attribution/case_selection_manifest.csv:
1. Case 1: R1234yf (HFO/HCFO zero-shot, model: HFC_all_M0, HFO external test set)
2. Case 2: R134 vs R134a (M1 LORO axis, model: HFC_all_M0, strictly matched T/P pairs in same IL)
3. Case 3: BF4 vs PF6 (B2 axis, model: B2_M0, strictly from B2 test split, matched T/P pairs in same cation/refri)
4. Case 4: R1336mzz(E/Z) (HFO/HCFO zero-shot, model: HFC_all_M0, 11 E + 12 Z points with matched pair links)
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results_attribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CSV = pd.read_csv(PROJECT_ROOT / "index_with_anion.csv")
INDEX_HFO = INDEX_CSV[INDEX_CSV["sheet"] == "Table S4. VLE HFOs"].copy()
META_HFC = pd.read_csv(PROJECT_ROOT / "processed_tri_data_hfc2739/meta_info.csv")
HFO_PREDS = pd.read_csv(PROJECT_ROOT / "paper_results/hfo_zeroshot_predictions_HFC_all.csv")
B2_SPLIT = np.load(PROJECT_ROOT / "splits_anion_ood/split_B2_Fam3_InorganicFluoride.npz")
B2_TEST_INDICES = set(B2_SPLIT["test"])

HFO_CANONICAL_SMILES = {
    "R1234YF": "C=C(F)C(F)(F)F",
    "R1234ZE(E)": "F/C=C/C(F)(F)F",
    "R1233ZD(E)": "FC(F)(F)/C=C/Cl",
    "R1336MZZ(E)": "FC(F)(F)/C=C/C(F)(F)F",
    "R1336MZZ(Z)": r"FC(F)(F)/C=C\C(F)(F)F",
}

manifest_rows = []

# =============================================================================
# Case 1: R1234yf (HFO Zero-Shot)
# =============================================================================
hfo_m0 = HFO_PREDS[HFO_PREDS["descriptor_mode"] == "M0"]
yf_pts = hfo_m0[hfo_m0["refrigerant"] == "R1234YF"].copy()

# 选取 4 个极具代表性的不同宿主环境与温压点
selected_yf_samples = [
    "[emim]__[Ac]__R1234yf__313.15__0.311",
    "[emim]__[BF4]__R1234yf__313.15__0.3919",
    "[bmim]__[Ac]__R1234yf__313.15__0.561",
    "[bmim]__[PF6]__R1234yf__313.15__0.23",
]

for s_id in selected_yf_samples:
    sub = yf_pts[yf_pts["sample_id"] == s_id]
    if not sub.empty:
        row = sub.iloc[0]
        # 在 Table S4 中检索匹配行获得精确 npy_idx 与原始 SMILES
        match_s4 = INDEX_HFO[
            (INDEX_HFO["refrigerant"].str.upper() == "R1234YF") &
            (np.abs(INDEX_HFO["T_K"] - float(row["T_K"])) < 0.1) &
            (np.abs(INDEX_HFO["P_MPa"] - float(row["P_MPa"])) < 0.002)
        ]
        s4_row = match_s4.iloc[0]
        data_idx = int(s4_row["npy_idx"])
        cat_smi = str(s4_row["cation_smiles"])
        ani_smi = str(s4_row["anion_smiles"])
        ref_smi = HFO_CANONICAL_SMILES.get("R1234YF", str(s4_row["refri_smiles"]))

        manifest_rows.append({
            "case_id": "Case1_R1234yf",
            "sample_id": row["sample_id"],
            "split_axis": "HFO/HCFO zero-shot",
            "split_role": "test",
            "model_dir": "results_hfc_all/HFC_all_M0",
            "mode": "M0",
            "seed": "42,43,44,45,46",
            "refrigerant": "R1234yf",
            "cation": row["cation"],
            "anion": row["anion"],
            "T_K": float(row["T_K"]),
            "P_MPa": float(row["P_MPa"]),
            "x1": float(row["true_x1"]),
            "pair_match_id": "None",
            "selection_reason": f"Representative HFO zero-shot point in [{row['cation']}][{row['anion']}]",
            "data_source_idx": data_idx,
            "is_hfo": True,
            "cation_smiles": cat_smi,
            "anion_smiles": ani_smi,
            "refri_smiles": ref_smi,
        })

# =============================================================================
# Case 2: R134 vs R134a (M1 LORO Isomer Paradox)
# =============================================================================
# 严格按同一 IL + 相同/极接近 T + 相同/极接近 P 匹配
r134_df = META_HFC[META_HFC["refrigerant"] == "R134"]
r134a_df = META_HFC[META_HFC["refrigerant"] == "R134a"]

target_systems = [("[emim]", "[BEI]"), ("[emim]", "[OTf]")]
pair_counter = 1

for cat, ani in target_systems:
    sub_134 = r134_df[(r134_df["cation"] == cat) & (r134_df["anion"] == ani)]
    sub_134a = r134a_df[(r134a_df["cation"] == cat) & (r134a_df["anion"] == ani)]

    # 寻找 2 个极度接近的压强点
    for idx_134, r_134 in sub_134.iterrows():
        cand = sub_134a[np.abs(sub_134a["T_K"] - r_134["T_K"]) < 1.0]
        if not cand.empty:
            p_diff = np.abs(cand["P_MPa"] - r_134["P_MPa"])
            best_idx = p_diff.idxmin()
            if p_diff.min() < 0.005 and pair_counter <= 4:
                r_134a = cand.loc[best_idx]
                pair_id = f"Pair_C2_{pair_counter:02d}_{cat}{ani}"
                
                # R134 row
                manifest_rows.append({
                    "case_id": "Case2_R134_vs_R134a",
                    "sample_id": r_134["sample_id"],
                    "split_axis": "M1 LORO",
                    "split_role": "test",
                    "model_dir": "results_hfc_all/HFC_all_M0",
                    "mode": "M0",
                    "seed": "42,43,44,45,46",
                    "refrigerant": "R134",
                    "cation": cat,
                    "anion": ani,
                    "T_K": float(r_134["T_K"]),
                    "P_MPa": float(r_134["P_MPa"]),
                    "x1": float(r_134["x1"]),
                    "pair_match_id": pair_id,
                    "selection_reason": f"Matched isomer pair {pair_id} (R134 arm)",
                    "data_source_idx": int(r_134["npy_idx"]),
                    "is_hfo": False,
                    "cation_smiles": str(r_134["cation_smiles"]),
                    "anion_smiles": str(r_134["anion_smiles"]),
                    "refri_smiles": str(r_134["refri_smiles"]),
                })
                # R134a row
                manifest_rows.append({
                    "case_id": "Case2_R134_vs_R134a",
                    "sample_id": r_134a["sample_id"],
                    "split_axis": "M1 LORO",
                    "split_role": "test",
                    "model_dir": "results_hfc_all/HFC_all_M0",
                    "mode": "M0",
                    "seed": "42,43,44,45,46",
                    "refrigerant": "R134a",
                    "cation": cat,
                    "anion": ani,
                    "T_K": float(r_134a["T_K"]),
                    "P_MPa": float(r_134a["P_MPa"]),
                    "x1": float(r_134a["x1"]),
                    "pair_match_id": pair_id,
                    "selection_reason": f"Matched isomer pair {pair_id} (R134a arm)",
                    "data_source_idx": int(r_134a["npy_idx"]),
                    "is_hfo": False,
                    "cation_smiles": str(r_134a["cation_smiles"]),
                    "anion_smiles": str(r_134a["anion_smiles"]),
                    "refri_smiles": str(r_134a["refri_smiles"]),
                })
                pair_counter += 1

# =============================================================================
# Case 3: BF4 vs PF6 (B2 Anion OOD Test Split)
# =============================================================================
meta_b2_test = META_HFC.iloc[sorted(list(B2_TEST_INDICES))].copy()
bf4_test = meta_b2_test[meta_b2_test["anion"] == "[BF4]"]
pf6_test = meta_b2_test[meta_b2_test["anion"] == "[PF6]"]

pair_b2_counter = 1
for cat_target, ref_target in [("[bmim]", "R32"), ("[emim]", "R134a")]:
    sub_b = bf4_test[(bf4_test["cation"] == cat_target) & (bf4_test["refrigerant"] == ref_target)]
    sub_p = pf6_test[(pf6_test["cation"] == cat_target) & (pf6_test["refrigerant"] == ref_target)]

    for idx_b, rb in sub_b.iterrows():
        cand = sub_p[np.abs(sub_p["T_K"] - rb["T_K"]) < 1.0]
        if not cand.empty:
            p_diff = np.abs(cand["P_MPa"] - rb["P_MPa"])
            best_idx = p_diff.idxmin()
            if p_diff.min() < 0.005 and pair_b2_counter <= 4:
                rp = cand.loc[best_idx]
                pair_id = f"Pair_C3_{pair_b2_counter:02d}_{cat_target}_{ref_target}"

                # BF4 row
                manifest_rows.append({
                    "case_id": "Case3_BF4_vs_PF6",
                    "sample_id": rb["sample_id"],
                    "split_axis": "B2",
                    "split_role": "test",
                    "model_dir": "results_split_B/B2_M0",
                    "mode": "M0",
                    "seed": "42,43,44,45,46",
                    "refrigerant": ref_target,
                    "cation": cat_target,
                    "anion": "[BF4]",
                    "T_K": float(rb["T_K"]),
                    "P_MPa": float(rb["P_MPa"]),
                    "x1": float(rb["x1"]),
                    "pair_match_id": pair_id,
                    "selection_reason": f"Strict B2 test matched pair (BF4 arm)",
                    "data_source_idx": int(rb["npy_idx"]),
                    "is_hfo": False,
                    "cation_smiles": str(rb["cation_smiles"]),
                    "anion_smiles": str(rb["anion_smiles"]),
                    "refri_smiles": str(rb["refri_smiles"]),
                })
                # PF6 row
                manifest_rows.append({
                    "case_id": "Case3_BF4_vs_PF6",
                    "sample_id": rp["sample_id"],
                    "split_axis": "B2",
                    "split_role": "test",
                    "model_dir": "results_split_B/B2_M0",
                    "mode": "M0",
                    "seed": "42,43,44,45,46",
                    "refrigerant": ref_target,
                    "cation": cat_target,
                    "anion": "[PF6]",
                    "T_K": float(rp["T_K"]),
                    "P_MPa": float(rp["P_MPa"]),
                    "x1": float(rp["x1"]),
                    "pair_match_id": pair_id,
                    "selection_reason": f"Strict B2 test matched pair (PF6 arm)",
                    "data_source_idx": int(rp["npy_idx"]),
                    "is_hfo": False,
                    "cation_smiles": str(rp["cation_smiles"]),
                    "anion_smiles": str(rp["anion_smiles"]),
                    "refri_smiles": str(rp["refri_smiles"]),
                })
                pair_b2_counter += 1

# =============================================================================
# Case 4: R1336mzz(E/Z) (HFO Zero-Shot Stereochemistry Boundary)
# =============================================================================
# 包含全部 11 个 E 和 12 个 Z 的官方 HFO 测试点
e_pts = hfo_m0[hfo_m0["refrigerant"] == "R1336MZZ(E)"].copy()
z_pts = hfo_m0[hfo_m0["refrigerant"] == "R1336MZZ(Z)"].copy()

# 为最接近的温压点建立配对标识
pair_ez_counter = 1
for _, re in e_pts.iterrows():
    cand = z_pts[np.abs(z_pts["T_K"] - re["T_K"]) < 2.0]
    best_pair_id = "Unpaired_E"
    if not cand.empty:
        p_diff = np.abs(cand["P_MPa"] - re["P_MPa"])
        if p_diff.min() < 0.05:
            best_pair_id = f"Pair_C4_{pair_ez_counter:02d}"
            pair_ez_counter += 1

    # 在 Table S4 中检索匹配行获得精确 npy_idx 与原始 SMILES
    match_s4 = INDEX_HFO[
        (INDEX_HFO["refrigerant"].str.upper() == "R1336MZZ(E)") &
        (np.abs(INDEX_HFO["T_K"] - float(re["T_K"])) < 0.1) &
        (np.abs(INDEX_HFO["P_MPa"] - float(re["P_MPa"])) < 0.002)
    ]
    s4_row = match_s4.iloc[0]
    data_idx = int(s4_row["npy_idx"])
    cat_smi = str(s4_row["cation_smiles"])
    ani_smi = str(s4_row["anion_smiles"])
    ref_smi = HFO_CANONICAL_SMILES["R1336MZZ(E)"]

    manifest_rows.append({
        "case_id": "Case4_R1336mzz_Stereo",
        "sample_id": re["sample_id"],
        "split_axis": "HFO/HCFO zero-shot",
        "split_role": "test",
        "model_dir": "results_hfc_all/HFC_all_M0",
        "mode": "M0",
        "seed": "42,43,44,45,46",
        "refrigerant": "R1336mzz(E)",
        "cation": re["cation"],
        "anion": re["anion"],
        "T_K": float(re["T_K"]),
        "P_MPa": float(re["P_MPa"]),
        "x1": float(re["true_x1"]),
        "pair_match_id": best_pair_id,
        "selection_reason": "Official HFO zero-shot E-isomer probe point",
        "data_source_idx": data_idx,
        "is_hfo": True,
        "cation_smiles": cat_smi,
        "anion_smiles": ani_smi,
        "refri_smiles": ref_smi,
    })

for _, rz in z_pts.iterrows():
    match_s4 = INDEX_HFO[
        (INDEX_HFO["refrigerant"].str.upper() == "R1336MZZ(Z)") &
        (np.abs(INDEX_HFO["T_K"] - float(rz["T_K"])) < 0.1) &
        (np.abs(INDEX_HFO["P_MPa"] - float(rz["P_MPa"])) < 0.002)
    ]
    s4_row = match_s4.iloc[0]
    data_idx = int(s4_row["npy_idx"])
    cat_smi = str(s4_row["cation_smiles"])
    ani_smi = str(s4_row["anion_smiles"])
    ref_smi = HFO_CANONICAL_SMILES["R1336MZZ(Z)"]

    manifest_rows.append({
        "case_id": "Case4_R1336mzz_Stereo",
        "sample_id": rz["sample_id"],
        "split_axis": "HFO/HCFO zero-shot",
        "split_role": "test",
        "model_dir": "results_hfc_all/HFC_all_M0",
        "mode": "M0",
        "seed": "42,43,44,45,46",
        "refrigerant": "R1336mzz(Z)",
        "cation": rz["cation"],
        "anion": rz["anion"],
        "T_K": float(rz["T_K"]),
        "P_MPa": float(rz["P_MPa"]),
        "x1": float(rz["true_x1"]),
        "pair_match_id": "Z_Isomer_Reference",
        "selection_reason": "Official HFO zero-shot Z-isomer probe point",
        "data_source_idx": data_idx,
        "is_hfo": True,
        "cation_smiles": cat_smi,
        "anion_smiles": ani_smi,
        "refri_smiles": ref_smi,
    })

df_manifest = pd.DataFrame(manifest_rows)
out_csv = OUT_DIR / "case_selection_manifest.csv"
df_manifest.to_csv(out_csv, index=False)

print("\n" + "=" * 78)
print(f"  [SUCCESS] Step 25 Case Selection Manifest 成功生成: {out_csv}")
print(f"  总样本量 N={len(df_manifest)} (Case 分布: {df_manifest['case_id'].value_counts().to_dict()})")
print("=" * 78 + "\n")
