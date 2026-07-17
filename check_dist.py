# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd

df = pd.read_csv('Original_Data/whole.csv')

ANION_FAMILY_MAP = {
    'TF2N': 'F1', 'TFSI': 'F1', 'NTF2': 'F1',
    'OTF':  'F1', 'TFO':  'F1', 'TTES': 'F1',
    'HFPS': 'F1', 'PFBS': 'F1', 'TFES': 'F1',
    'TPES': 'F1', 'FS':   'F1',
    'FEP':  'F2', 'BEI':  'F2', 'TMEM': 'F2', 'PFP': 'F2',
    'BF4':  'F3', 'PF6':  'F3',
    'AC':   'A1', 'DCA':  'A1', 'SCN':  'A1',
    'PR':   'A1', 'PE':   'A1', 'ET2PO4': 'A1', 'TMPP': 'A1',
    'CL':   'A2', 'BR':   'A2', 'I': 'A2', 'NO3': 'A2',
}

def assign_family(name):
    key = str(name).strip().upper().replace('[','').replace(']','').replace('-','')
    return ANION_FAMILY_MAP.get(key, 'Other')

df['family'] = df['anion'].apply(assign_family)

# Check Other
other = df[df['family'] == 'Other']
print("=== 'Other' anions ===")
for anion, cnt in other['anion'].value_counts().items():
    subset = other[other['anion'] == anion]['x_CO2']
    print(f"  {anion:20s}: {cnt:4d}  x1 mean={subset.mean():.4f}")

# Check F1 subtypes
f1 = df[df['family'] == 'F1']
print("\n=== F1 subtypes ===")
for anion, cnt in f1['anion'].value_counts().items():
    subset = f1[f1['anion'] == anion]['x_CO2']
    print(f"  {anion:20s}: {cnt:4d}  x1 mean={subset.mean():.4f}")
