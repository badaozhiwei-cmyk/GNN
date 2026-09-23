"""
audit_phase2_deep_dive.py — Rigorous Objective Deep Dive of Phase 2 Graph-IG Results
===================================================================================
Analyzes all 7 output artifacts in v7_shadow_experiment/results_attribution:
1. Provenance Integrity
2. Completeness & Gate B Breakdown (Small-denominator singularity vs real integration drift)
3. Edge Attribution Reality (Edge share vs Atom share)
4. Faithfulness Dissection (Why R_faith=1.0 occurs, why R_faith<1.0 occurs, V7-A vs V7-B)
5. Cross-Seed Stability (Spearman rank & Pearson r distribution: V7-A vs V7-B)
6. Functional Group Attribution Profiles
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "v7_shadow_experiment" / "results_attribution"

def main():
    print("=" * 85)
    print("  PHASE 2 ATTRIBUTION RIGOROUS AUDIT & STATISTICAL DISSECTION")
    print("=" * 85)

    # 1. Load Data
    df_manifest = pd.read_csv(ROOT / "results_attribution" / "case_selection_manifest.csv")
    df_audit = pd.read_csv(RES_DIR / "graph_smiles_alignment_audit.csv")
    df_atoms = pd.read_csv(RES_DIR / "v7_graph_attribution_atoms.csv")
    df_edges = pd.read_csv(RES_DIR / "v7_graph_attribution_edges.csv")
    df_groups = pd.read_csv(RES_DIR / "v7_graph_attribution_groups.csv")
    df_faith = pd.read_csv(RES_DIR / "v7_graph_attribution_faithfulness.csv")
    df_stab = pd.read_csv(RES_DIR / "v7_graph_attribution_stability.csv")
    with open(RES_DIR / "v7_graph_attribution_provenance.json", "r", encoding="utf-8") as f:
        prov = json.load(f)

    # 2. Basic Dimensions
    print("\n--- [1. ARTIFACT DIMENSIONS & PROVENANCE] ---")
    print(f"  • Alignment Audit Rows : {len(df_audit)} (Expected: 129)")
    print(f"  • Atom Attribution Rows: {len(df_atoms)} (Expected: 430 evals * avg 29.8 atoms = 12830)")
    print(f"  • Edge Attribution Rows: {len(df_edges)} (Expected: 430 evals * avg 55.7 edges = 23940)")
    print(f"  • Group Attribution Rows:{len(df_groups)} (Expected: 2870)")
    print(f"  • Faithfulness Rows    : {len(df_faith)} (Expected: 430)")
    print(f"  • Stability Rows       : {len(df_stab)} (Expected: 86 = 43 cases * 2 models)")
    print(f"  • Riemann Steps        : {prov['metadata']['riemann_steps']}")
    print(f"  • Device Recorded      : {prov['metadata']['device']}")

    # 3. Completeness & Gate B Dissection
    print("\n--- [2. GATE B COMPLETENESS & SINGULARITY AUDIT] ---")
    df_faith["delta_y_chem"] = (df_faith["y_orig"] - df_faith["y_base"]).abs()
    
    print("  Overall Completeness Metrics (430 evaluations):")
    for col in ["comp_abs_err", "comp_rel_err", "delta_y_chem"]:
        vals = df_faith[col].dropna()
        print(f"    • {col:<15}: Median={vals.median():.4e}, Mean={vals.mean():.4e}, "
              f"P25={np.percentile(vals, 25):.4e}, P75={np.percentile(vals, 75):.4e}, "
              f"P95={np.percentile(vals, 95):.4e}, Max={vals.max():.4e}")

    # Breakdown by Model Family
    for mf in ["V7-A", "V7-B"]:
        sub = df_faith[df_faith["model_family"] == mf]
        print(f"\n  Model {mf} Completeness Breakdown:")
        print(f"    - Comp Rel Err : Median={sub['comp_rel_err'].median()*100:.2f}%, Mean={sub['comp_rel_err'].mean()*100:.2f}%, "
              f"P95={np.percentile(sub['comp_rel_err'], 95)*100:.2f}%, Max={sub['comp_rel_err'].max()*100:.2f}%")
        print(f"    - Comp Abs Err : Median={sub['comp_abs_err'].median():.4e}, Mean={sub['comp_abs_err'].mean():.4e}, "
              f"Max={sub['comp_abs_err'].max():.4e}")
        print(f"    - Delta y Chem : Median={sub['delta_y_chem'].median():.4f}, Mean={sub['delta_y_chem'].mean():.4f}, "
              f"Min={sub['delta_y_chem'].min():.4e}")

    # Inspect Outliers (Comp Rel Err > 50%)
    outliers = df_faith[df_faith["comp_rel_err"] > 0.50]
    print(f"\n  Outliers with Comp Rel Err > 50%: Total {len(outliers)} / 430 ({len(outliers)/430*100:.1f}%)")
    for _, r in outliers.iterrows():
        print(f"    Case {r['case_id']:<6} | {r['model_family']} Seed {r['seed']} | Sample: {r['sample_id']:<45} | "
              f"RelErr: {r['comp_rel_err']*100:6.1f}% | AbsErr: {r['comp_abs_err']:.4e} | Delta_y: {r['delta_y_chem']:.4e}")

    # Filtered Rel Err for non-near-zero delta_y
    valid_sub = df_faith[df_faith["delta_y_chem"] >= 0.01]
    print(f"\n  Filtered Rel Err (|Delta y| >= 0.01, {len(valid_sub)} / 430 evals):")
    print(f"    Median={valid_sub['comp_rel_err'].median()*100:.2f}%, Mean={valid_sub['comp_rel_err'].mean()*100:.2f}%, "
          f"P95={np.percentile(valid_sub['comp_rel_err'], 95)*100:.2f}%, Max={valid_sub['comp_rel_err'].max()*100:.2f}%")

    # 4. Edge Attribution Reality Check
    print("\n--- [3. ATOM VS EDGE ATTRIBUTION MAGNITUDE AUDIT] ---")
    tot_atom_abs = df_atoms["a_i_abs"].sum()
    tot_edge_abs = df_edges["edge_abs"].sum()
    mean_atom_abs = df_atoms["a_i_abs"].mean()
    mean_edge_abs = df_edges["edge_abs"].mean()
    edge_share = tot_edge_abs / (tot_atom_abs + tot_edge_abs)

    print(f"  • Total Summed Atom Abs Attribution: {tot_atom_abs:.4f} (Mean per atom: {mean_atom_abs:.6f})")
    print(f"  • Total Summed Edge Abs Attribution: {tot_edge_abs:.4f} (Mean per edge: {mean_edge_abs:.6f})")
    print(f"  • Overall Edge Attribution Share    : {edge_share*100:.4f}% ({edge_share:.2e})")
    print(f"  • Atom-to-Edge Attribution Ratio    : {tot_atom_abs / max(tot_edge_abs, 1e-12):.1f} : 1")

    # By Component
    for comp in ["Cation", "Anion", "Refri"]:
        c_atoms = df_atoms[df_atoms["component"] == comp]["a_i_abs"].sum()
        c_edges = df_edges[df_edges["component"] == comp]["edge_abs"].sum()
        print(f"    - {comp:<7}: Atom Abs={c_atoms:.3f}, Edge Abs={c_edges:.5f} (Edge share: {c_edges/(c_atoms+c_edges)*100:.4f}%)")

    # 5. Faithfulness Dissection
    print("\n--- [4. FAITHFULNESS (R_faith) DETAILED DISSECTION] ---")
    rf = df_faith["r_faith_comp"].dropna()
    print(f"  Overall R_faith (430 evals):")
    print(f"    Median={rf.median():.3f}, Mean={rf.mean():.3f}, Std={rf.std():.3f}, "
          f"P25={np.percentile(rf, 25):.3f}, P75={np.percentile(rf, 75):.3f}, Max={rf.max():.3f}, Min={rf.min():.3f}")
    
    n_gt1 = (rf > 1.0).sum()
    n_eq1 = (rf.round(2) == 1.0).sum()
    n_lt1 = (rf < 0.99).sum()
    print(f"    • R_faith > 1.0 (Top-1 drop > random)  : {n_gt1} / 430 ({n_gt1/430*100:.1f}%)")
    print(f"    • R_faith ≈ 1.0 (Top-1 drop == random) : {n_eq1} / 430 ({n_eq1/430*100:.1f}%)")
    print(f"    • R_faith < 1.0 (Top-1 drop < random)  : {n_lt1} / 430 ({n_lt1/430*100:.1f}%)")

    # Why R_faith == 1.0?
    eq1_cases = df_faith[df_faith["r_faith_comp"].round(2) == 1.0]
    print(f"\n  Substructure of R_faith ≈ 1.0 cases:")
    print(eq1_cases["top_group_name"].value_counts().to_string())
    print("  Top refrigerants in R_faith ≈ 1.0:")
    r32_in_eq = eq1_cases["sample_id"].apply(lambda s: s.split("__")[2]).value_counts()
    print(r32_in_eq.to_string())

    # Comparison by Model Family
    for mf in ["V7-A", "V7-B"]:
        sub = df_faith[df_faith["model_family"] == mf]["r_faith_comp"]
        print(f"\n  Model {mf} Faithfulness:")
        print(f"    Median={sub.median():.3f}, Mean={sub.mean():.3f}, P90={np.percentile(sub, 90):.3f}, "
              f"> 1.0: {(sub > 1.0).mean()*100:.1f}%")

    # 6. Stability Dissection
    print("\n--- [5. CROSS-SEED STABILITY AUDIT (SPEARMAN & PEARSON)] ---")
    for mf in ["V7-A", "V7-B"]:
        sub = df_stab[df_stab["model_family"] == mf]
        sp = sub["mean_spearman_rank_r"].dropna()
        pr = sub["mean_pearson_r"].dropna()
        print(f"  Model {mf} Cross-Seed Stability (43 cases):")
        print(f"    • Spearman Rank r: Median={sp.median():.3f}, Mean={sp.mean():.3f}, Min={sp.min():.3f}, Max={sp.max():.3f}")
        print(f"    • Pearson r      : Median={pr.median():.3f}, Mean={pr.mean():.3f}, Min={pr.min():.3f}, Max={pr.max():.3f}")
        n_sp_high = (sp >= 0.70).sum()
        print(f"    • Cases with Spearman r >= 0.70: {n_sp_high} / {len(sp)} ({n_sp_high/len(sp)*100:.1f}%)")

    # 7. Substructure Attribution Profiles
    print("\n--- [6. DOMINANT FUNCTIONAL GROUPS BY COMPONENT] ---")
    for comp in ["Cation", "Anion", "Refri"]:
        sub_g = df_groups[df_groups["component"] == comp]
        grp_summary = sub_g.groupby("group_name").agg(
            mean_P_g_atom=("P_g_atom", "mean"),
            mean_P_g_graph=("P_g_graph", "mean"),
            mean_A_abs=("A_g_abs", "mean"),
            count=("A_g_abs", "count")
        ).sort_values("mean_P_g_atom", ascending=False)
        print(f"\n  Component: {comp}")
        print(grp_summary.to_string())

if __name__ == "__main__":
    main()
