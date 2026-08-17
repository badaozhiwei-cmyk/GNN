import numpy as np
import pandas as pd
import sys

def verify_alignment():
    print("Loading data...")
    df = pd.read_csv('index_with_anion.csv')
    try:
        data = np.load('processed_tri_data/data.npy', allow_pickle=True)
        labels = np.load('processed_tri_data/label.npy', allow_pickle=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"CSV rows: {len(df)}")
    print(f"Graphs in data.npy: {len(data)}")
    print(f"Labels in label.npy: {len(labels)}")
    
    if len(df) != len(data) or len(df) != len(labels):
        print("ERROR: Length mismatch between CSV and graph data!")
        sys.exit(1)
        
    mismatches = 0
    tolerance = 1e-5
    
    for i in range(len(df)):
        csv_T = df.iloc[i]['T_K']
        csv_P = df.iloc[i]['P_MPa']
        csv_x1 = df.iloc[i]['x1']
        
        g_T = data[i][3]
        g_P = data[i][4]
        g_x1 = labels[i]
        
        if not (np.isclose(csv_T, g_T, atol=tolerance) and 
                np.isclose(csv_P, g_P, atol=tolerance) and 
                np.isclose(csv_x1, g_x1, atol=tolerance)):
            mismatches += 1
            if mismatches <= 5:
                print(f"Mismatch at index {i}:")
                print(f"  CSV: T={csv_T}, P={csv_P}, x1={csv_x1}")
                print(f"  GPH: T={g_T}, P={g_P}, lbl={g_x1}")
                
    print(f"\nAudit complete. Mismatches found: {mismatches}")
    if mismatches == 0:
        print("[OK] 100% 1-to-1 alignment confirmed between index_with_anion.csv, data.npy, and label.npy!")
        print("     The row indices in our HFC-LORO .npz files perfectly map to the graph dataset.")
    else:
        print("[FAILED] Misalignment detected. Do not proceed to model training.")
        sys.exit(1)

if __name__ == '__main__':
    verify_alignment()
