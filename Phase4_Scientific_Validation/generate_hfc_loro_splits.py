import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load full dataset
data_path = 'index_with_anion.csv'
df = pd.read_csv(data_path)

print(f"Total rows in dataset: {len(df)}")

# Define the 12 pure HFCs
hfc_list = ['R32', 'R134a', 'R125', 'R134', 'R152a', 'R23', 'R41', 'R161', 'R143a', 'R245fa', 'R236fa', 'R227ea']

# Get rows belonging to any of these HFCs
hfc_df = df[df['refrigerant'].isin(hfc_list)]
print(f"Total HFC rows: {len(hfc_df)}")

for hfc in hfc_list:
    count = len(hfc_df[hfc_df['refrigerant'] == hfc])
    print(f"  {hfc}: {count} rows")

# We want to use ONLY the HFC rows for this experiment.
# However, to be compatible with existing pipeline indexing (which expects indices from 0 to 4445),
# we must output the absolute row indices of the original dataframe!

output_dir = 'Phase4_Scientific_Validation/HFC_LORO_Splits'
os.makedirs(output_dir, exist_ok=True)

# Generate 3-Val Nested Splits
for test_hfc in hfc_list:
    # Pool of possible validation HFCs
    possible_vals = [h for h in hfc_list if h != test_hfc]
    
    # Randomly select 3 validation HFCs
    selected_vals = np.random.choice(possible_vals, size=3, replace=False)
    
    for val_idx, val_hfc in enumerate(selected_vals, start=1):
        # Identify Train HFCs
        train_hfcs = [h for h in hfc_list if h != test_hfc and h != val_hfc]
        
        # Get actual row indices from the original dataframe
        test_indices = df[df['refrigerant'] == test_hfc].index.values
        val_indices = df[df['refrigerant'] == val_hfc].index.values
        train_indices = df[df['refrigerant'].isin(train_hfcs)].index.values
        
        # Rigorous Disjoint Assertions
        assert set(train_indices).isdisjoint(set(test_indices)), f"Leakage detected between Train and Test for {test_hfc}!"
        assert set(val_indices).isdisjoint(set(train_indices)), f"Leakage detected between Val and Train for {test_hfc}!"
        assert set(test_indices).isdisjoint(set(val_indices)), f"Leakage detected between Test and Val for {test_hfc}!"
        
        # Save to npz
        filename = os.path.join(output_dir, f"split_hfc_loro_{test_hfc}_val{val_idx}.npz")
        np.savez(filename, 
                 train_idx=train_indices,
                 val_idx=val_indices,
                 test_idx=test_indices,
                 test_hfc=test_hfc,
                 val_hfc=val_hfc,
                 train_hfcs=train_hfcs)
        
    print(f"Generated 3 splits for Test={test_hfc} (Vals: {', '.join(selected_vals)})")

print(f"\nAll {len(hfc_list) * 3} nested split combinations successfully generated in {output_dir}")
