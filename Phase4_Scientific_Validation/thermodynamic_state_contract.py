"""
thermodynamic_state_contract.py — Canonical Thermodynamic State Key Contract (v1.0)
===================================================================================
Protocol: Scientific Integrity Framework (Nature MI / JACS Red-Team Standards)

Core Objective & Epistemological Definition:
  `state_key_v1` strictly defines a:
    "canonical thermodynamic state record identity at declared measurement precision"
  (rather than an abstract metaphysical "exact physical state identity").

Contract Invariants:
  1. Chemical species are defined strictly by canonical InChIKeys from frozen manifests.
  2. Temperature T is canonicalized to 0.01 K resolution (f"{T:.2f}").
  3. Pressure P is canonicalized to 0.0001 MPa resolution (f"{P:.4f}").
  4. Measurement context (VLE, saturated liquid solution, liquid mole fraction x1)
     is strictly verified as homogeneous across the entire experimental universe.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from chemical_identity_contract import get_refrigerant_identity

IL_MANIFEST_P = MODULE_DIR / "ionic_liquid_identity_manifest.csv"
if not IL_MANIFEST_P.exists():
    raise FileNotFoundError(f"Missing IL manifest: {IL_MANIFEST_P}")

_df_il = pd.read_csv(IL_MANIFEST_P)
_IL_BY_NAME: Dict[str, Dict[str, Any]] = {}
for _, row in _df_il.iterrows():
    entry = row.to_dict()
    cname = str(entry['canonical_name']).strip().upper()
    cname_clean = cname.replace('[', '').replace(']', '')
    _IL_BY_NAME[cname] = entry
    _IL_BY_NAME[cname_clean] = entry

def get_il_ion_identity(identifier: str) -> Dict[str, Any]:
    clean = str(identifier).strip().upper()
    clean_no_bracket = clean.replace('[', '').replace(']', '')
    if clean in _IL_BY_NAME:
        return _IL_BY_NAME[clean]
    if clean_no_bracket in _IL_BY_NAME:
        return _IL_BY_NAME[clean_no_bracket]
    raise KeyError(f"🚨 [IL Registry Breach] Unknown ionic liquid ion: '{identifier}'")

def canonicalize_temperature(T_K: float) -> str:
    """Canonicalizes temperature in Kelvin to 2 decimal places (0.01 K resolution)"""
    val = round(float(T_K), 2)
    return f"{val:.2f}"

def canonicalize_pressure(P_MPa: float) -> str:
    """Canonicalizes pressure in MPa to 4 decimal places (0.1 kPa resolution)"""
    val = round(float(P_MPa), 4)
    return f"{val:.4f}"

def build_state_key_v1(
    cation: str,
    anion: str,
    refrigerant: str,
    T_K: float,
    P_MPa: float
) -> str:
    """
    Constructs the canonical thermodynamic state key: state_key_v1.
    Format:
      CAT:<cat_inchikey>|ANI:<ani_inchikey>|REF:<ref_inchikey>|T:<T_K_canonical>|P:<P_MPa_canonical>
    """
    cat_entry = get_il_ion_identity(cation)
    ani_entry = get_il_ion_identity(anion)
    ref_entry = get_refrigerant_identity(refrigerant)

    cat_ikey = cat_entry['inchi_key']
    ani_ikey = ani_entry['inchi_key']
    ref_ikey = ref_entry['inchi_key']

    t_str = canonicalize_temperature(T_K)
    p_str = canonicalize_pressure(P_MPa)

    return f"CAT:{cat_ikey}|ANI:{ani_ikey}|REF:{ref_ikey}|T:{t_str}|P:{p_str}"

def parse_state_key_v1(state_key: str) -> Dict[str, str]:
    parts = state_key.split('|')
    if len(parts) != 5:
        raise ValueError(f"Invalid state_key_v1 format: {state_key}")
    return {
        "cat_inchikey": parts[0].split(':', 1)[1],
        "ani_inchikey": parts[1].split(':', 1)[1],
        "ref_inchikey": parts[2].split(':', 1)[1],
        "T_K": parts[3].split(':', 1)[1],
        "P_MPa": parts[4].split(':', 1)[1]
    }

def assert_dataset_measurement_context_homogeneity(df: pd.DataFrame) -> Dict[str, str]:
    """
    Asserts that all rows in the dataset share identical measurement context:
    - Source: 'Table S3. VLE HFCs'
    - Measurement Type: Vapor-Liquid Equilibrium (VLE)
    - Phase: Saturated Liquid Solution
    - Composition Basis: Liquid-phase refrigerant mole fraction x1
    """
    required_cols = {'sheet', 'cation', 'anion', 'refrigerant', 'T_K', 'P_MPa', 'x1'}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"🚨 [Measurement Context Breach] Missing required columns: {missing}")

    # Check nulls
    null_counts = df[list(required_cols)].isnull().sum()
    if null_counts.any():
        raise RuntimeError(f"🚨 [Measurement Context Breach] Null values detected: {null_counts.to_dict()}")

    # Check sheet homogeneity
    sheets = set(df['sheet'].dropna().unique())
    if sheets != {'Table S3. VLE HFCs'}:
        raise RuntimeError(f"🚨 [Measurement Context Breach] Inhomogeneous sheets detected: {sheets}")

    # Check physical ranges
    if not ((df['T_K'] >= 250.0) & (df['T_K'] <= 400.0)).all():
        raise RuntimeError("🚨 [Measurement Context Breach] Temperature out of physical VLE bounds [250, 400] K")
    if not ((df['P_MPa'] > 0.0) & (df['P_MPa'] <= 10.0)).all():
        raise RuntimeError("🚨 [Measurement Context Breach] Pressure out of physical VLE bounds (0, 10] MPa")
    if not ((df['x1'] >= 0.0) & (df['x1'] <= 1.0)).all():
        raise RuntimeError("🚨 [Measurement Context Breach] x1 out of physical mole fraction bounds [0, 1]")

    return {
        "source_sheet": "Table S3. VLE HFCs",
        "measurement_type": "Vapor_Liquid_Equilibrium_VLE",
        "phase_regime": "saturated_liquid_solution",
        "composition_basis": "liquid_mole_fraction_x1",
        "n_samples": str(len(df)),
        "is_homogeneous": "True"
    }
