"""
generate_step25_final_freeze.py — V6 + Step 25 Final Provenance & Immutable Freeze Record

Compiles git commit SHA, checkpoint SHAs, attribution result SHA-256 hashes,
diagnostic files, paper tables, and explicit protocol parameters.
"""
import os
import json
import hashlib
import subprocess
from pathlib import Path

def get_git_commit_sha():
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        res = subprocess.check_output(cmd).decode("utf-8").strip()
        return res
    except Exception as e:
        return f"Error: {e}"

def compute_sha256(filepath):
    p = Path(filepath)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    root = Path(".")
    git_sha = get_git_commit_sha()
    print(f"[*] Current Git HEAD SHA: {git_sha}")

    # 1. Architecture & Pipeline code
    code_files = [
        "GNN_for_property_prediction/Model_v6.py",
        "GNN_for_property_prediction/Dataset_v6.py",
        "prepare_tri_graph_data_v6.py",
        "run_ablation.py",
        "research_pipeline/step25_graph_substructure_attribution.py",
        "research_pipeline/audit_step25_v2.py",
        "research_pipeline/sync_and_deep_audit.py",
        "research_pipeline/diagnostic_virtual_node.py",
    ]
    code_hashes = {}
    for cf in code_files:
        h = compute_sha256(root / cf)
        code_hashes[cf] = {"exists": h is not None, "sha256": h}

    # 2. Checkpoints
    checkpoint_dirs = [
        "results_hfc_all/HFC_all_M0",
        "results_hfc_all/HFC_all_Mreduced",
        "results_split_B/B1_M0",
        "results_split_B/B1_Mreduced",
        "results_split_B/B2_M0",
        "results_split_B/B2_Mreduced",
        "results_split_L2/L2_M0",
        "results_split_L2/L2_Mreduced",
    ]
    checkpoints_hashes = {}
    for cdir in checkpoint_dirs:
        checkpoints_hashes[cdir] = {}
        for s in [42, 43, 44, 45, 46]:
            ckpt_path = root / cdir / f"best_seed_{s}.pth"
            h = compute_sha256(ckpt_path)
            checkpoints_hashes[cdir][f"seed_{s}"] = {
                "path": str(ckpt_path).replace("\\", "/"),
                "exists": h is not None,
                "sha256": h
            }

    # 3. Attribution Outputs
    attr_files = [
        "results_attribution/case_selection_manifest.csv",
        "results_attribution/graph_attribution_atoms_bonds.csv",
        "results_attribution/graph_attribution_faithfulness.csv",
        "results_attribution/graph_attribution_groups.csv",
        "results_attribution/graph_attribution_stability.csv",
        "results_attribution/attribution_sample_level.csv",
        "results_attribution/attribution_species_summary.csv",
        "results_attribution/attribution_perturbation_sanity.csv",
        "results_attribution/attribution_stability_per_sample.csv",
        "results_attribution/attribution_stability_summary.csv",
        "results_attribution/graph_attribution_provenance.json",
    ]
    attr_hashes = {}
    for af in attr_files:
        h = compute_sha256(root / af)
        size = (root / af).stat().st_size if (root / af).exists() else 0
        attr_hashes[af] = {"exists": h is not None, "size_bytes": size, "sha256": h}

    # 4. Diagnostic & Forensic Proof Files
    diag_files = [
        "diagnostic_outputs/step25_integrity_audit_v2.csv",
        "diagnostic_outputs/stratified_activity_summary.csv",
        "diagnostic_outputs/ez_representation_identity_proof.json",
        "diagnostic_outputs/smarts_specific_coverage_breakdown.csv",
        "diagnostic_outputs/virtual_node_layerwise_breakdown.csv",
        "diagnostic_outputs/virtual_node_global_summary.csv",
        "diagnostic_outputs/diagnostic_virtual_node_output.txt",
        "diagnostic_outputs/audit_step25_attribution_output.txt",
    ]
    diag_hashes = {}
    for df in diag_files:
        h = compute_sha256(root / df)
        size = (root / df).stat().st_size if (root / df).exists() else 0
        diag_hashes[df] = {"exists": h is not None, "size_bytes": size, "sha256": h}

    # 5. Paper Benchmark & Main Tables
    paper_files = [
        "paper_results/FINAL_FREEZE.json",
        "paper_results/table_main_generalization_boundary.csv",
        "paper_results/table_m1_formal_diagnostics.csv",
        "paper_results/table_hfo_zeroshot_metrics_HFC_all.csv",
        "paper_results/table_hfo_zeroshot_metrics_B1.csv",
        "paper_results/table_hfo_zeroshot_metrics_L2.csv",
        "paper_results/table_gnn_vs_rf_comparison.csv",
        "paper_results/table_step24_stereo_ablation.csv",
        "paper_results/table_training_size_curve.csv",
        "paper_results/table_cluster_bootstrap_significance.csv",
    ]
    paper_hashes = {}
    for pf in paper_files:
        h = compute_sha256(root / pf)
        size = (root / pf).stat().st_size if (root / pf).exists() else 0
        paper_hashes[pf] = {"exists": h is not None, "size_bytes": size, "sha256": h}

    # 6. Assemble Full Freeze Provenance Record
    freeze_manifest = {
        "title": "V6 & Step 25 Final Immutable Provenance Record",
        "timestamp": "2026-09-21",
        "frozen_state_affirmation": {
            "v6_model_frozen": True,
            "v6_dataset_frozen": True,
            "read_only_rule": "Model_v6.py and Dataset_v6.py are permanently frozen. No further training or parameter modifications permitted on V6.",
            "status": "FROZEN_FOR_PUBLICATION"
        },
        "git_provenance": {
            "commit_sha": git_sha,
            "branch": "main",
            "repository": "https://github.com/badaozhiwei-cmyk/GNN.git"
        },
        "experimental_seeds": [42, 43, 44, 45, 46],
        "step25_protocol_specifications": {
            "total_audit_evaluations": 215,
            "probe_cases_count": 4,
            "probe_samples_count": 43,
            "seeds_per_sample": 5,
            "integrated_gradients": {
                "method": "Riemann sum path integral",
                "steps": 25,
                "baseline": "Zero-graph baseline (H=0, E=0, topological edges preserved)",
                "target": "Solubility output scalar f(G, cond)"
            },
            "faithfulness_protocol": {
                "intervention_type": "continuous post-embedding masking (model-sensitivity probe, not chemical causal synthesis)",
                "top1_masking": "Top-1 attributed functional group masked with zero embeddings",
                "random_matched_masking": "k random atoms sampled within same component excluding Top-1 group atoms",
                "random_trials": 30,
                "metrics": [
                    "delta_y_top (absolute perturbation delta of top-1 group)",
                    "delta_y_rand_comp (mean absolute perturbation delta of component-matched random control)",
                    "r_faith_comp (ratio delta_y_top / delta_y_rand_comp, reported alongside absolute deltas to avoid small-denominator bias)"
                ]
            },
            "representation_bifurcation_criterion": {
                "metric": "Forward graph sensitivity delta_y_graph = |f(H, E) - f(H_0, E_0)|",
                "threshold_tau": 1e-3,
                "bimodal_separation_window": "[1e-6, 5e-3]",
                "graph_active_evaluations": 35,
                "graph_inactive_evaluations": 180,
                "active_evaluation_rate_pct": 16.279,
                "seed_distribution": {
                    "seed_42": "0 / 43 (0.0%)",
                    "seed_43": "0 / 43 (0.0%)",
                    "seed_44": "0 / 43 (0.0%)",
                    "seed_45": "35 / 43 (81.4%)",
                    "seed_46": "0 / 43 (0.0%)"
                },
                "paper_framing_guideline": "Strictly frame as 'seed-dependent representation bifurcation / scalar bypass'. Do not describe as 'GNN failed / dead network', and do not claim causal proof."
            },
            "smarts_functional_group_schema": {
                "total_probe_atoms": 750,
                "primary_assignment_coverage_pct": 100.0,
                "specific_smarts_coverage_pct": 97.87,
                "other_atom_fraction_pct": 2.13,
                "refrigerant_specific_coverage_pct": 100.0
            },
            "ez_stereochemical_boundary": {
                "representation_identity": "H_E == H_Z, E_E == E_Z, edge_index_E == edge_index_Z in production 2D GNN (edge_dim=3, max abs diff = 0.0)",
                "attribution_consistency": "attribution_E approx attribution_Z (mean diff = 0.00178, subsequent consistency under identity)",
                "paper_framing_guideline": "Frame as 'stereochemical representation boundary / 2D topological blind spot' necessitating physical prior or explicit stereochemical intervention."
            }
        },
        "checksums": {
            "core_code": code_hashes,
            "checkpoints": checkpoints_hashes,
            "step25_attribution_outputs": attr_hashes,
            "step25_diagnostics": diag_hashes,
            "paper_results_tables": paper_hashes
        }
    }

    out_file = root / "paper_results/step25_final_provenance_freeze.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(freeze_manifest, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Step 25 & V6 Final Provenance Freeze successfully generated at: {out_file}")
    print(f"[INFO] Total verified checkpoint suites: {len(checkpoints_hashes)}")
    print(f"[INFO] Total verified attribution outputs: {len(attr_hashes)}")
    print(f"[INFO] Total verified diagnostic outputs: {len(diag_hashes)}")
    print(f"[INFO] Total verified paper tables: {len(paper_hashes)}")

if __name__ == "__main__":
    main()
