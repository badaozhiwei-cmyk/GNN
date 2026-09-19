#!/usr/bin/env python3
"""
step19_final_paper_audit.py — Phase III-A Step 19
==================================================
Full quantitative + semantic paper consistency audit.

NOT a simple string grep. Uses semantic numeric matching:
  "Is this displayed number a valid rounding of the frozen canonical value?"

Audit Layers:
  L1  FREEZE ↔ CSV main table
  L2  FREEZE ↔ manuscript LaTeX / Markdown tables
  L3  FREEZE ↔ prose numeric claims
  L4  Aggregation basis consistency
  L5  High-risk terminology scan
  L6  Stale / deprecated number & protocol violation detection

Usage:
  uv run python research_pipeline/step19_final_paper_audit.py

Outputs:
  paper_results/audit_paper_consistency.json
  paper_results/audit_paper_consistency.csv
"""

import json
import csv
import re
import sys
import io
import pathlib
from collections import OrderedDict
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════
ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper_results"

FILES = {
    "freeze":        PAPER / "FINAL_FREEZE.json",
    "csv_table":     PAPER / "table_main_generalization_boundary.csv",
    "results_md":    PAPER / "manuscript_results.md",
    "discussion_md": PAPER / "manuscript_discussion.md",
    "si_md":         PAPER / "manuscript_supplementary.md",
}

# ═══════════════════════════════════════════════════════════════
# High-risk terms (regex pattern, risk category)
# ═══════════════════════════════════════════════════════════════
RISK_TERMS = [
    (r"\bproves?\b",                "CAUSAL_CLAIM"),
    (r"\bproving\b",                "CAUSAL_CLAIM"),
    (r"\bproven\b",                 "CAUSAL_CLAIM"),
    (r"\bproof\b",                  "CAUSAL_CLAIM"),
    (r"\bconclusively\b",           "OVERCLAIM"),
    (r"\bcalibratio?n\b",           "UQ_TERMINOLOGY"),
    (r"\bcalibrated\b",             "UQ_TERMINOLOGY"),
    (r"\bcatastrophic\b",           "LANGUAGE_INTENSITY"),
    (r"\bstochastically smaller\b", "STATISTICAL_CLAIM"),
    (r"\bcausal(?:ly|ity|ation)?\b","CAUSAL_CLAIM"),
]

# Deprecated sample counts that must NOT appear as N values
STALE_N_VALUES = {260, 584, 162}

# ═══════════════════════════════════════════════════════════════
# Utility: load files
# ═══════════════════════════════════════════════════════════════

def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def load_text(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read()

# ═══════════════════════════════════════════════════════════════
# Utility: parse numbers from LaTeX / Markdown
# ═══════════════════════════════════════════════════════════════

def parse_float(s):
    """Parse a string into float, stripping LaTeX / Markdown formatting."""
    if s is None:
        return None
    s = str(s).strip()
    # Strip markdown bold
    s = s.replace("**", "")
    # Strip LaTeX wrappers
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
    # Normalise minus signs
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    # Remove % sign
    s = s.replace("%", "")
    # Remove commas and em-dashes
    s = s.strip(" \u2014\t\n\r,")
    if s in ("", "\u2014", "-", "—"):
        return None
    try:
        return float(s)
    except ValueError:
        return None

# ═══════════════════════════════════════════════════════════════
# Core: semantic numeric matching
# ═══════════════════════════════════════════════════════════════

def semantic_match(canonical, displayed):
    """Return (is_match: bool, detail: str).

    Determines the display precision of *displayed*, rounds *canonical*
    to that precision, and checks for exact match (within float ε).
    This catches stale-artifact numbers that are "close but wrong".
    """
    if canonical is None or displayed is None:
        return None, "Cannot compare (None)"

    canonical = float(canonical)
    displayed = float(displayed)

    # --- integer comparison ---
    if displayed == int(displayed) and abs(canonical - round(canonical)) < 1e-9:
        ok = int(round(canonical)) == int(displayed)
        return ok, f"Integer: {int(round(canonical))} vs {int(displayed)}"

    # --- determine display precision (# decimal places) ---
    for n_dec in range(1, 9):
        if abs(round(displayed, n_dec) - displayed) < 1e-12:
            break
    else:
        n_dec = 8

    rounded_can = round(canonical, n_dec)
    diff = abs(rounded_can - displayed)
    threshold = 0.6 * 10 ** (-n_dec)          # generous ε
    ok = diff < threshold
    return ok, (
        f"Precision {n_dec}d: round({canonical:.8g}, {n_dec}) = "
        f"{rounded_can:.{n_dec}f} vs displayed {displayed:.{n_dec}f}  "
        f"(diff={diff:.2e}, thr={threshold:.2e})"
    )

# ═══════════════════════════════════════════════════════════════
# Audit item container
# ═══════════════════════════════════════════════════════════════

class AuditItem:
    __slots__ = ("check_id", "layer", "status", "canonical",
                 "found", "location", "detail", "category")

    def __init__(self, check_id, layer, status, *,
                 canonical=None, found=None, location="",
                 detail="", category="quantitative"):
        self.check_id = check_id
        self.layer    = layer
        self.status   = status          # PASS / FAIL / WARNING / INFO
        self.canonical = canonical
        self.found     = found
        self.location  = location
        self.detail    = detail
        self.category  = category

    def to_dict(self):
        return OrderedDict([
            ("check_id",  self.check_id),
            ("layer",     self.layer),
            ("category",  self.category),
            ("status",    self.status),
            ("canonical", self.canonical),
            ("found",     self.found),
            ("location",  self.location),
            ("detail",    self.detail),
        ])

# ═══════════════════════════════════════════════════════════════
# Layer 1 — CSV main table ↔ FREEZE
# ═══════════════════════════════════════════════════════════════

def audit_csv_table(freeze, results):
    path = FILES["csv_table"]
    if not path.exists():
        results.append(AuditItem("L1_CSV_MISSING", "L1", "FAIL",
                                 detail="CSV table file not found"))
        return

    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    f_axes = {a["axis"]: a for a in freeze["cross_axis_5seed_mean"]}
    f_uq   = {a["axis"]: a for a in freeze["cross_axis_ensemble_uq"]}

    for row in rows:
        # Strip any BOM or whitespace from keys
        clean_row = {k.strip().lstrip("\ufeff"): v for k, v in row.items()}
        axis = clean_row.get("Axis", "").strip()
        if axis not in f_axes:
            continue
        fa = f_axes[axis]
        fu = f_uq.get(axis, {})

        checks = [
            ("N",         "N_test",    fa),
            ("M0 MAE",    "M0_MAE",    fa),
            ("Mred MAE",  "Mred_MAE",  fa),
            ("M0 R2",     "M0_R2",     fa),
            ("Mred R2",   "Mred_R2",   fa),
            ("Mred sigma", "Mred_mean_sigma", fu),
        ]
        for col, fkey, src in checks:
            displayed = parse_float(clean_row.get(col, ""))
            canonical = src.get(fkey)
            if displayed is None or canonical is None:
                continue
            ok, detail = semantic_match(canonical, displayed)
            results.append(AuditItem(
                f"L1_{axis}_{fkey}", "L1",
                "PASS" if ok else "FAIL",
                canonical=canonical, found=displayed,
                location=f"CSV row «{axis}», col «{col}»",
                detail=detail,
            ))

        # delta_MAE%
        delta_d = parse_float(clean_row.get("delta_MAE", ""))
        delta_c = fa.get("delta_MAE_pct")
        if delta_d is not None and delta_c is not None:
            ok, detail = semantic_match(delta_c, delta_d)
            results.append(AuditItem(
                f"L1_{axis}_delta_MAE_pct", "L1",
                "PASS" if ok else "FAIL",
                canonical=delta_c, found=delta_d,
                location=f"CSV row «{axis}», col «delta_MAE»",
                detail=detail,
            ))

# ═══════════════════════════════════════════════════════════════
# Helpers: extract table rows from LaTeX / Markdown
# ═══════════════════════════════════════════════════════════════

def _latex_table_rows(text, label_fragment):
    """Return list-of-lists of cell strings from a LaTeX tabular
    whose \\label{...} or \\caption{...} contains *label_fragment*."""
    m = re.search(re.escape(label_fragment), text)
    if not m:
        return []
    # In LaTeX, \label can be before or after \begin{tabular}
    # Check after label first:
    start = text.find("\\begin{tabular", m.start())
    if start == -1 or start - m.start() > 500:
        # Check before label
        start = text.rfind("\\begin{tabular", 0, m.start())
    if start < 0:
        return []

    end = text.find("\\end{tabular}", start)
    if end < 0:
        return []

    block = text[start:end]
    rows, in_data = [], False
    for chunk in block.split("\\\\"):
        s = chunk.strip()
        if "\\midrule" in s:
            in_data = True
            s = s.replace("\\midrule", "").strip()
            if not s:
                continue
        if "\\bottomrule" in s:
            break
        if in_data and s:
            rows.append([c.strip() for c in s.split("&")])
    return rows


def _md_table_rows(text, header_fragment):
    """Return list-of-lists of cell strings from a Markdown table
    whose preceding header / text contains *header_fragment*."""
    m = re.search(re.escape(header_fragment), text, re.IGNORECASE)
    if not m:
        return []
    out, started = [], False
    for line in text[m.start():].split("\n"):
        if "|" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(set(c) <= set("-: ") for c in cells):
                continue                       # separator
            out.append(cells)
            started = True
        elif started:
            break
    return out

# ═══════════════════════════════════════════════════════════════
# Layer 2 — Manuscript tables ↔ FREEZE
# ═══════════════════════════════════════════════════════════════

def _match_axis(text):
    """Guess axis name from a table cell."""
    t = text.lower().replace("$", "").replace("\\", "")
    for key in ("hfo/hcfo", "hfo"):
        if key in t:
            return "HFO/HCFO"
    for key in ("m1", "b1", "b2", "l2"):
        if key in t:
            return key.upper()
    return None


def audit_manuscript_tables(freeze, results):
    results_md = load_text(FILES["results_md"])
    si_md      = load_text(FILES["si_md"])
    if results_md is None:
        results.append(AuditItem("L2_NO_RESULTS", "L2", "FAIL",
                                 detail="manuscript_results.md missing"))
        return

    f_axes = {a["axis"]: a for a in freeze["cross_axis_5seed_mean"]}
    f_uq   = {a["axis"]: a for a in freeze["cross_axis_ensemble_uq"]}
    f_pairs = {p["pair"]: p for p in freeze.get("l2_pair_level", [])}
    f_spec  = {s["species"]: s for s in freeze.get("unsaturated_species_breakdown", [])}
    f_sens  = freeze.get("hfo_only_sensitivity_N1016", {})

    # ── Table 1: cross-axis master ──────────────────────────────
    for row in _latex_table_rows(results_md, "cross_axis_master"):
        if len(row) < 8:
            continue
        axis = _match_axis(row[0])
        if axis is None:
            continue
        fa = f_axes.get(axis, {})
        col_map = [
            (2, "N_test"), (3, "M0_MAE"), (4, "Mred_MAE"),
            (6, "M0_R2"),  (7, "Mred_R2"),
        ]
        for ci, fk in col_map:
            if ci >= len(row):
                continue
            d = parse_float(row[ci])
            c = fa.get(fk)
            if d is not None and c is not None:
                ok, det = semantic_match(c, d)
                results.append(AuditItem(
                    f"L2_T1_{axis}_{fk}", "L2",
                    "PASS" if ok else "FAIL",
                    canonical=c, found=d,
                    location=f"Results Table 1, {axis}",
                    detail=det,
                ))

    # ── Table 2: L2 pair-level ──────────────────────────────────
    pair_pat = {
        r"emim.*bf":  "EMIM__BF4",
        r"emim.*otf": "EMIM__OTF",
        r"bmim.*otf": "BMIM__OTF",
        r"hmim.*bf":  "HMIM__BF4",
    }
    for row in _latex_table_rows(results_md, "l2_pairs"):
        if len(row) < 6:
            continue
        cell0 = row[0].lower()
        pk = None
        for pat, key in pair_pat.items():
            if re.search(pat, cell0):
                pk = key
                break
        if pk is None:
            # pooled row
            if "pooled" in cell0 or "pool" in cell0:
                fu_l2 = f_uq.get("L2", {})
                for ci, fk in [(2, "M0_ensemble_MAE"), (3, "Mred_ensemble_MAE")]:
                    if ci >= len(row):
                        continue
                    d = parse_float(row[ci])
                    c = fu_l2.get(fk)
                    if d is not None and c is not None:
                        ok, det = semantic_match(c, d)
                        results.append(AuditItem(
                            f"L2_T2_Pooled_{fk}", "L2",
                            "PASS" if ok else "FAIL",
                            canonical=c, found=d,
                            location="Results Table 2, Pooled",
                            detail=det,
                        ))
            continue
        fp = f_pairs.get(pk, {})
        for ci, fk in [(2, "M0_MAE"), (3, "Mred_MAE"),
                        (5, "Mred_R2"), (6, "Mred_mean_sigma")]:
            if ci >= len(row):
                continue
            d = parse_float(row[ci])
            c = fp.get(fk)
            if d is not None and c is not None:
                ok, det = semantic_match(c, d)
                results.append(AuditItem(
                    f"L2_T2_{pk}_{fk}", "L2",
                    "PASS" if ok else "FAIL",
                    canonical=c, found=d,
                    location=f"Results Table 2, {pk}",
                    detail=det,
                ))

    # ── Table 3: HFO species ────────────────────────────────────
    sp_pat = [
        (r"r1234yf",              "R1234yf"),
        (r"r1234ze",              "R1234ze(E)"),
        (r"r1233zd",              "R1233zd(E)"),
        (r"r1336mzz.*trans|r1336mzz\(e\)|r1336mzz.*e.*butene", "R1336mzz(E)"),
        (r"r1336mzz.*cis|r1336mzz\(z\)|r1336mzz.*z.*butene",   "R1336mzz(Z)"),
    ]
    for row in _latex_table_rows(results_md, "unsat_species"):
        if len(row) < 8:
            continue
        cell0 = row[0].lower()
        sk = None
        for pat, key in sp_pat:
            if re.search(pat, cell0):
                sk = key
                break
        if sk is None:
            # total / pooled row
            if "total" in cell0 or "pooled" in cell0:
                fu_hfo = f_uq.get("HFO/HCFO", {})
                for ci, fk in [(3, "M0_ensemble_MAE"), (4, "Mred_ensemble_MAE")]:
                    if ci >= len(row):
                        continue
                    d = parse_float(row[ci])
                    c = fu_hfo.get(fk)
                    if d is not None and c is not None:
                        ok, det = semantic_match(c, d)
                        results.append(AuditItem(
                            f"L2_T3_Total_{fk}", "L2",
                            "PASS" if ok else "FAIL",
                            canonical=c, found=d,
                            location="Results Table 3, Total",
                            detail=det,
                        ))
            continue
        fs = f_spec.get(sk, {})
        for ci, fk in [(2, "N"), (3, "M0_MAE"), (4, "Mred_MAE"),
                        (6, "M0_R2"), (7, "Mred_R2")]:
            if ci >= len(row):
                continue
            d = parse_float(row[ci])
            c = fs.get(fk)
            if d is not None and c is not None:
                ok, det = semantic_match(c, d)
                results.append(AuditItem(
                    f"L2_T3_{sk}_{fk}", "L2",
                    "PASS" if ok else "FAIL",
                    canonical=c, found=d,
                    location=f"Results Table 3, {sk}",
                    detail=det,
                ))

    # ── Table 4: UQ ─────────────────────────────────────────────
    for row in _latex_table_rows(results_md, "uncertainty_risk"):
        if len(row) < 7:
            continue
        axis = _match_axis(row[0])
        if axis is None:
            continue
        fu = f_uq.get(axis, {})
        for ci, fk in [(1, "N_test"), (2, "Mred_ensemble_MAE"),
                        (4, "Mred_rho_sigma_error")]:
            if ci >= len(row):
                continue
            d = parse_float(row[ci])
            c = fu.get(fk)
            if d is not None and c is not None:
                ok, det = semantic_match(c, d)
                results.append(AuditItem(
                    f"L2_T4_{axis}_{fk}", "L2",
                    "PASS" if ok else "FAIL",
                    canonical=c, found=d,
                    location=f"Results Table 4, {axis}",
                    detail=det,
                ))
        # sigma column (index 3)
        if len(row) > 3:
            d_sig = parse_float(row[3])
            c_sig = fu.get("Mred_mean_sigma")
            if d_sig is not None and c_sig is not None:
                ok, det = semantic_match(c_sig, d_sig)
                results.append(AuditItem(
                    f"L2_T4_{axis}_sigma", "L2",
                    "PASS" if ok else "FAIL",
                    canonical=c_sig, found=d_sig,
                    location=f"Results Table 4, {axis}, sigma col",
                    detail=det,
                ))

    # ── SI Tables ───────────────────────────────────────────────
    if si_md is None:
        return

    # S1.2 — authoritative N values
    for row in _md_table_rows(si_md, "Authoritative Test"):
        if len(row) < 3:
            continue
        axis = _match_axis(row[0])
        if axis is None:
            continue
        n_d = parse_float(row[2])
        n_c = f_axes.get(axis, {}).get("N_test")
        if n_d is not None and n_c is not None:
            ok = int(n_d) == int(n_c)
            results.append(AuditItem(
                f"L2_SI_S12_{axis}_N", "L2",
                "PASS" if ok else "FAIL",
                canonical=int(n_c), found=int(n_d),
                location=f"SI S1.2, {axis}",
                detail=f"N: {int(n_c)} vs {int(n_d)}",
            ))

    # S3.2 — HFO-only sensitivity
    for row in _md_table_rows(si_md, "Dataset Scope"):
        if len(row) < 7:
            continue
        cell0 = row[0].lower()
        if "pure" not in cell0 and "excl" not in cell0 and "1016" not in cell0:
            continue
        for ci, fk in [(1, "N"), (2, "M0_MAE"), (3, "Mred_MAE"),
                        (5, "M0_R2"), (6, "Mred_R2")]:
            if ci >= len(row):
                continue
            d = parse_float(row[ci])
            c = f_sens.get(fk)
            if d is not None and c is not None:
                ok, det = semantic_match(c, d)
                results.append(AuditItem(
                    f"L2_SI_S32_HFOonly_{fk}", "L2",
                    "PASS" if ok else "FAIL",
                    canonical=c, found=d,
                    location=f"SI S3.2 sensitivity, HFO-only",
                    detail=det,
                ))

    # S4.1 — LORO species table (check N per species sums to 1403)
    loro_rows = _md_table_rows(si_md, "Macro Average")
    if not loro_rows:
        loro_rows = _md_table_rows(si_md, "R134a")
    n_sum = 0
    for row in loro_rows:
        if len(row) < 3:
            continue
        n_val = parse_float(row[2])
        if n_val is not None and 10 < n_val < 500:
            n_sum += int(n_val)
    if n_sum > 0:
        ok = n_sum == 1403
        results.append(AuditItem(
            "L2_SI_S41_LORO_N_sum", "L2",
            "PASS" if ok else "FAIL",
            canonical=1403, found=n_sum,
            location="SI S4.1 LORO species table",
            detail=f"Sum of per-species N = {n_sum}; expected 1403",
        ))

    # S4.2 — risk-coverage (check 50% reduction in col 7)
    for row in _md_table_rows(si_md, "50% Coverage"):
        if len(row) < 8:
            continue
        axis = _match_axis(row[0])
        if axis is None:
            continue
        # row[6] is MAE at 50% coverage; row[7] is Total Risk Reduction %
        risk_red_d = parse_float(row[7])
        fu = f_uq.get(axis, {})
        risk50_c = fu.get("risk50_Mred")
        if risk_red_d is not None and risk50_c is not None:
            # canonical is fraction like 0.272, displayed is percentage like -27.2%
            disp_pct = abs(risk_red_d)
            canon_pct = risk50_c * 100.0 if risk50_c < 1.0 else risk50_c
            ok, det = semantic_match(canon_pct, disp_pct)
            results.append(AuditItem(
                f"L2_SI_S42_{axis}_risk50_reduction", "L2",
                "PASS" if ok else "FAIL",
                canonical=canon_pct, found=disp_pct,
                location=f"SI S4.2, {axis}, Total Risk Reduction",
                detail=det,
            ))


# ═══════════════════════════════════════════════════════════════
# Layer 3 — Prose numeric claims ↔ FREEZE
# ═══════════════════════════════════════════════════════════════

def _find_near(text, context_re, value_re=r"-?\d+\.?\d*(?:e[+-]?\d+)?",
               window=250):
    """Return all (float, match_context) near *context_re*."""
    out = []
    for cm in re.finditer(context_re, text, re.I | re.S):
        lo = max(0, cm.start() - window)
        hi = min(len(text), cm.end() + window)
        region = text[lo:hi]
        for nm in re.finditer(value_re, region):
            v = parse_float(nm.group())
            if v is not None:
                out.append(v)
    return out


def audit_prose_claims(freeze, results):
    docs = {k: load_text(FILES[k]) for k in ("results_md", "discussion_md", "si_md")}
    f_paired = freeze.get("l2_paired_statistics", {})
    f_sens   = freeze.get("hfo_only_sensitivity_N1016", {})
    f_spec   = {s["species"]: s for s in freeze.get("unsaturated_species_breakdown", [])}
    f_uq     = {u["axis"]: u for u in freeze["cross_axis_ensemble_uq"]}

    # Each: (id, file_key, search_context, canonical, description)
    checks = [
        # L2 paired stats in Results
        ("L3_L2_mean_delta",  "results_md",
         r"Mean Paired Effect.*?0\.\d+", f_paired.get("mean_delta_mae"),
         "L2 mean paired effect Δe"),
        ("L3_L2_CI_lo",  "results_md",
         r"confidence interval.*?\[?\s*0\.\d+", f_paired.get("bootstrap_95_ci", [None])[0],
         "L2 bootstrap CI lower bound"),
        ("L3_L2_winrate", "results_md",
         r"59\.\d.*?percent|59\.\d.*?instance|win.*?59",
         (f_paired.get("win_rate", 0) * 100 if f_paired.get("win_rate") else None),
         "L2 win rate (%)"),
        # R1336mzz(E) false-confidence sigma in Results
        ("L3_R1336mzzE_sigma", "results_md",
         r"R1336mzz\(E\).*?sigma.*?0\.\d+|R1336mzz\(E\).*?disagreement.*?0\.\d+",
         f_spec.get("R1336mzz(E)", {}).get("Mred_sigma"),
         "R1336mzz(E) ensemble σ = 0.0213"),
        # HFO sensitivity ΔMAE% in SI
        ("L3_SI_HFOonly_pct", "si_md",
         r"18\.\d.*?%|error reduction",
         f_sens.get("delta_MAE_pct"),
         "HFO-only ΔMAE percentage"),
        # M1 Ensemble MAE in Results UQ section
        ("L3_M1_ens_MAE", "results_md",
         r"M1.*?ensemble MAE.*?0\.\d+|ensemble MAE.*?0\.0613",
         f_uq.get("M1", {}).get("Mred_ensemble_MAE"),
         "M1 Mred ensemble MAE"),
        # Discussion rho range
        ("L3_rho_range_lo", "results_md",
         r"0\.31.*?0\.61|rho.*?0\.31",
         0.3084,     # lowest rho across axes
         "Spearman ρ range lower end (~0.31)"),
    ]

    for cid, fkey, ctx_re, canonical, desc in checks:
        text = docs.get(fkey)
        if text is None or canonical is None:
            continue
        vals = _find_near(text, ctx_re)
        if not vals:
            results.append(AuditItem(cid, "L3", "INFO",
                                     canonical=canonical,
                                     location=fkey,
                                     detail=f"{desc}: context pattern not found"))
            continue
        matched = False
        for v in vals:
            ok, det = semantic_match(canonical, v)
            if ok:
                results.append(AuditItem(cid, "L3", "PASS",
                                         canonical=canonical, found=v,
                                         location=fkey, detail=f"{desc}: {det}"))
                matched = True
                break
        if not matched:
            results.append(AuditItem(cid, "L3", "WARNING",
                                     canonical=canonical,
                                     found=str(vals[:5]),
                                     location=fkey,
                                     detail=f"{desc}: pattern found but no semantic match among {vals[:5]}"))


# ═══════════════════════════════════════════════════════════════
# Layer 4 — Aggregation basis consistency
# ═══════════════════════════════════════════════════════════════

def audit_aggregation(results):
    for dk in ("results_md", "si_md"):
        text = load_text(FILES[dk])
        if text is None:
            continue

        # M1 should mention "macro" near MAE / ensemble
        found_macro = bool(re.search(
            r"M1.*?macro|macro.*?12.*?refrigerant|macro.averaged",
            text, re.I | re.S))
        results.append(AuditItem(
            f"L4_{dk}_M1_macro", "L4",
            "PASS" if found_macro else "WARNING",
            location=dk,
            detail="M1 aggregation labelled as macro-averaged" if found_macro
                   else "M1 'macro' label not found — risk of reader confusion",
            category="aggregation",
        ))

        # Check that "seed-mean" and "ensemble" are not conflated
        if re.search(r"5.seed mean.*?ensemble|ensemble.*?5.seed mean", text, re.I):
            results.append(AuditItem(
                f"L4_{dk}_seed_vs_ensemble", "L4", "WARNING",
                location=dk,
                detail="'5-seed mean' and 'ensemble' appear nearby — verify no conflation",
                category="aggregation",
            ))


# ═══════════════════════════════════════════════════════════════
# Layer 5 — High-risk terminology scan
# ═══════════════════════════════════════════════════════════════

def audit_terminology(results):
    for dk in ("results_md", "discussion_md", "si_md"):
        text = load_text(FILES[dk])
        if text is None:
            continue
        lines = text.split("\n")
        for term_re, risk_cat in RISK_TERMS:
            for ln, line in enumerate(lines, 1):
                for m in re.finditer(term_re, line, re.I):
                    lo = max(0, m.start() - 50)
                    hi = min(len(line), m.end() + 50)
                    ctx = line[lo:hi].strip()
                    results.append(AuditItem(
                        f"L5_{dk}_L{ln}_{risk_cat}", "L5", "WARNING",
                        found=m.group(),
                        location=f"{dk}:{ln}",
                        detail=f"[{risk_cat}] «…{ctx}…»",
                        category="terminology",
                    ))


# ═══════════════════════════════════════════════════════════════
# Layer 6 — Stale numbers & FREEZE protocol violations
# ═══════════════════════════════════════════════════════════════

def audit_stale_numbers(results):
    for dk in ("results_md", "discussion_md", "si_md"):
        text = load_text(FILES[dk])
        if text is None:
            continue
        lines = text.split("\n")

        # 6a — deprecated N values in sample-count contexts
        count_ctx = re.compile(
            r"N\s*=|N\s*\\?=|\$N\$|sample|point|test|instance|observation",
            re.I)
        for ln, line in enumerate(lines, 1):
            for sv in STALE_N_VALUES:
                if str(sv) not in line:
                    continue
                # If this line is documenting outlawed/superseded legacy numbers, don't flag as accidental usage:
                low = line.lower()
                if any(kw in low for kw in ("outlawed", "legacy", "superseded", "preliminary cluster")):
                    continue
                # Check surrounding context looks like a sample count
                lo = line.find(str(sv))
                ctx = line[max(0, lo - 40):lo + len(str(sv)) + 40]
                if count_ctx.search(ctx):
                    results.append(AuditItem(
                        f"L6_{dk}_L{ln}_stale_N{sv}", "L6", "FAIL",
                        found=sv,
                        location=f"{dk}:{ln}",
                        detail=f"Deprecated count {sv} in count context: «…{ctx.strip()}…»",
                        category="stale",
                    ))

        # 6b — "calibration" near Figure 4 / uncertainty
        #       (violates FREEZE protocol_rules.uncertainty_wording)
        for m in re.finditer(r"calibrat\w+", text, re.I):
            lo = max(0, m.start() - 80)
            hi = min(len(text), m.end() + 80)
            ctx = text[lo:hi].replace("\n", " ")
            if re.search(r"figure\s*4|fig\.?\s*4|uncertainty|epistemic|selective",
                         ctx, re.I):
                ln = text[:m.start()].count("\n") + 1
                results.append(AuditItem(
                    f"L6_{dk}_L{ln}_calibration_protocol", "L6", "WARNING",
                    found=m.group(),
                    location=f"{dk}:{ln}",
                    detail=(f"FREEZE protocol violation: 'calibration' near "
                            f"uncertainty context. Use 'reliability' instead. "
                            f"«…{ctx.strip()}…»"),
                    category="stale",
                ))

        # 6c — stale SI "proving" (known issue #2)
        for m in re.finditer(r"\bproving\b", text, re.I):
            lo = max(0, m.start() - 60)
            hi = min(len(text), m.end() + 60)
            ctx = text[lo:hi].replace("\n", " ").strip()
            ln = text[:m.start()].count("\n") + 1
            results.append(AuditItem(
                f"L6_{dk}_L{ln}_proving", "L6", "WARNING",
                found="proving",
                location=f"{dk}:{ln}",
                detail=f"Strong causal language: «…{ctx}…»",
                category="stale",
            ))


# ═══════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════

def save_results(items):
    out = PAPER
    ts = datetime.now().isoformat()

    # JSON
    summary = {s: sum(1 for r in items if r.status == s)
               for s in ("PASS", "FAIL", "WARNING", "INFO")}
    by_layer = {}
    for layer in ("L1", "L2", "L3", "L4", "L5", "L6"):
        li = [r for r in items if r.layer == layer]
        by_layer[layer] = {
            "total": len(li),
            **{s: sum(1 for r in li if r.status == s)
               for s in ("PASS", "FAIL", "WARNING")},
        }
    blob = {
        "audit_timestamp": ts,
        "total_checks": len(items),
        "summary": summary,
        "by_layer": by_layer,
        "items": [r.to_dict() for r in items],
    }
    jpath = out / "audit_paper_consistency.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False, default=str)

    # CSV
    cpath = out / "audit_paper_consistency.csv"
    fields = ["check_id", "layer", "category", "status",
              "canonical", "found", "location", "detail"]
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in items:
            w.writerow(r.to_dict())

    print(f"\n📁 Saved:")
    print(f"   {jpath}")
    print(f"   {cpath}")


def print_summary(items):
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  STEP 19 — PAPER CONSISTENCY AUDIT REPORT")
    print(SEP)

    n = {s: sum(1 for r in items if r.status == s)
         for s in ("PASS", "FAIL", "WARNING", "INFO")}
    print(f"\n  Total checks : {len(items)}")
    print(f"  ✅ PASS      : {n.get('PASS',0)}")
    print(f"  ❌ FAIL      : {n.get('FAIL',0)}")
    print(f"  ⚠️  WARNING   : {n.get('WARNING',0)}")
    print(f"  ℹ️  INFO      : {n.get('INFO',0)}")

    layer_names = {
        "L1": "CSV Table ↔ FREEZE",
        "L2": "Manuscript Tables ↔ FREEZE",
        "L3": "Prose Claims ↔ FREEZE",
        "L4": "Aggregation Basis",
        "L5": "Terminology Risk",
        "L6": "Stale / Protocol",
    }
    print(f"\n{'─'*70}")
    for layer, name in layer_names.items():
        li = [r for r in items if r.layer == layer]
        if not li:
            continue
        p = sum(1 for r in li if r.status == "PASS")
        f_ = sum(1 for r in li if r.status == "FAIL")
        w = sum(1 for r in li if r.status == "WARNING")
        print(f"  {layer} {name:30s}  {p}✅  {f_}❌  {w}⚠️   ({len(li)} total)")

    # ── FAIL details ──
    fails = [r for r in items if r.status == "FAIL"]
    if fails:
        print(f"\n{'─'*70}")
        print(f"  ❌ FAILURES ({len(fails)}) — must fix before submission:\n")
        for r in fails:
            print(f"  [{r.check_id}]  {r.location}")
            print(f"    Canonical : {r.canonical}")
            print(f"    Found     : {r.found}")
            print(f"    Detail    : {r.detail}")
            print()

    # ── WARNING highlights ──
    warns = [r for r in items if r.status == "WARNING"]
    if warns:
        print(f"{'─'*70}")
        print(f"  ⚠️  WARNINGS ({len(warns)}) — showing first 25:\n")
        for r in warns[:25]:
            print(f"  [{r.check_id}]  {r.location}")
            print(f"    {r.detail}")
            print()
        if len(warns) > 25:
            print(f"  … and {len(warns)-25} more. See CSV for full list.\n")

    print(SEP)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("🔍 Step 19 — Paper Consistency Audit (semantic version)")
    print(f"   Root : {ROOT}")
    print(f"   Paper: {PAPER}")

    for key, path in FILES.items():
        tag = "✅" if path.exists() else "⚠️  MISSING"
        print(f"   {tag} {key}: {path.name}")

    freeze = load_json(FILES["freeze"])
    print(f"\n   FINAL_FREEZE loaded  ({len(freeze)} top-level keys)")

    items = []

    print("\n   ▸ Layer 1: CSV Table …")
    audit_csv_table(freeze, items)

    print("   ▸ Layer 2: Manuscript Tables …")
    audit_manuscript_tables(freeze, items)

    print("   ▸ Layer 3: Prose Claims …")
    audit_prose_claims(freeze, items)

    print("   ▸ Layer 4: Aggregation Basis …")
    audit_aggregation(items)

    print("   ▸ Layer 5: Terminology Risk …")
    audit_terminology(items)

    print("   ▸ Layer 6: Stale / Protocol …")
    audit_stale_numbers(items)

    save_results(items)
    print_summary(items)

    n_fail = sum(1 for r in items if r.status == "FAIL")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
