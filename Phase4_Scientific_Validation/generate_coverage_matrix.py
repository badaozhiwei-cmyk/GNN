import pandas as pd
import numpy as np
import os
import requests
import io
import urllib3
urllib3.disable_warnings()

# 1. Vega 2022 Table S2/S3 (Highly calibrated EOS descriptors)
VEGA_REFRIGERANTS = {
    'R41', 'R32', 'R23', 'R161', 'R152a', 'R134a', 'R125', 
    'R245fa', 'R236fa', 'R227ea', 'R1234yf', 'R1234ze(E)', 
    'R1336mzz(Z)', 'R1233zd(E)'
}

def check_ilp_database(cation_smiles_list, anion_smiles_list):
    """
    Attempts to cross-reference our IL SMILES with the ILP database on GitHub.
    Returns sets of SMILES that were successfully found.
    """
    url = "https://raw.githubusercontent.com/wangyingxie/ILP/main/dataset_iolitech_final.csv"
    found_cations = set()
    found_anions = set()
    try:
        print("Attempting to fetch ILP database from GitHub...")
        response = requests.get(url, verify=False, timeout=10)
        df_ilp = pd.read_csv(io.StringIO(response.text))
        ilp_text = df_ilp.to_string()
        
        for smi in cation_smiles_list:
            if type(smi) == str and smi in ilp_text:
                found_cations.add(smi)
        for smi in anion_smiles_list:
            if type(smi) == str and smi in ilp_text:
                found_anions.add(smi)
        print(f"ILP Audit Complete: Matched {len(found_cations)} cations, {len(found_anions)} anions.")
    except Exception as e:
        print(f"ILP Database unavailable or parse error: {e}")
        print("Falling back to Unified xTB Protocol for all Ionic Liquids.")
    
    return found_cations, found_anions

def generate_coverage_matrix():
    print("Starting Descriptor Provenance Audit...\n")
    
    # Load our 66 molecules
    df = pd.read_csv('../index_with_anion.csv')
    
    refrigerants = df[['refrigerant', 'refri_smiles']].drop_duplicates().values
    cations = df[['cation', 'cation_smiles']].drop_duplicates().values
    anions = df[['anion', 'anion_smiles']].drop_duplicates().values
    
    results = []
    
    # Audit Refrigerants
    for name, smi in refrigerants:
        in_vega = name in VEGA_REFRIGERANTS
        source = "Vega 2022 SI (Literature)" if in_vega else "Pending Unified Computation"
        mu_status = "✅" if in_vega else "❌"
        v_status = "✅" if in_vega else "❌"
        alpha_status = "❌" # Polarizability generally not in macroscopic Vega tables, needs API/Compute
        
        results.append({
            'Category': 'Refrigerant',
            'Molecule': name,
            'Dipole_μ': mu_status,
            'Polarizability_α': alpha_status,
            'Volume_V': v_status,
            'Primary_Source': source,
            'Fallback_Protocol': "Unified xTB Computation" if not in_vega else "-"
        })
        
    # Audit Ionic Liquids against open databases
    cat_smiles = [c[1] for c in cations]
    ani_smiles = [a[1] for a in anions]
    
    # Try fetching from GitHub
    found_cat, found_ani = check_ilp_database(cat_smiles, ani_smiles)
    
    # Cations
    for name, smi in cations:
        in_db = smi in found_cat
        status = "✅" if in_db else "❌"
        results.append({
            'Category': 'Cation',
            'Molecule': name,
            'Dipole_μ': 'N/A', # Not in 7D minimal set
            'Polarizability_α': status,
            'Volume_V': status,
            'Primary_Source': "ILP Database" if in_db else "Pending Unified Computation",
            'Fallback_Protocol': "Unified xTB Computation"
        })
        
    # Anions
    for name, smi in anions:
        in_db = smi in found_ani
        status = "✅" if in_db else "❌"
        results.append({
            'Category': 'Anion',
            'Molecule': name,
            'Dipole_μ': 'N/A',
            'Polarizability_α': status,
            'Volume_V': status,
            'Primary_Source': "ILP Database" if in_db else "Pending Unified Computation",
            'Fallback_Protocol': "Unified xTB Computation"
        })
        
    audit_df = pd.DataFrame(results)
    output_path = 'Descriptor_Coverage_Matrix.csv'
    audit_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nAudit completed. Matrix saved to {output_path}")
    print("\nSummary:")
    print(audit_df['Primary_Source'].value_counts())

if __name__ == "__main__":
    generate_coverage_matrix()
