import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import os

# Set paths
base_dir = r"C:\Users\霸道志伟\Desktop\GNN\Refrigerant-Solubility-GNN"
csv_path = os.path.join(base_dir, "index_with_anion.csv")
output_dir = os.path.join(base_dir, "Phase4_Scientific_Validation")
os.makedirs(output_dir, exist_ok=True)

# Read dataset
df = pd.read_csv(csv_path)

# Extract unique refrigerants and their SMILES
ref_df = df[['refrigerant', 'refri_smiles']].drop_duplicates().reset_index(drop=True)
print(f"Found {len(ref_df)} unique refrigerants.")

# Generate Morgan Fingerprints
fps = []
valid_refs = []
for i, row in ref_df.iterrows():
    ref_name = row['refrigerant']
    smiles = row['refri_smiles']
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fps.append(fp)
        valid_refs.append(ref_name)
    else:
        print(f"Failed to parse SMILES for {ref_name}: {smiles}")

# Calculate Tanimoto Similarity Matrix
n = len(valid_refs)
sim_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])

# Sort refrigerants to group similar ones together (simple hierarchical clustering order)
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# Convert similarity to distance
distance_matrix = 1.0 - sim_matrix
# Ensure diagonal is exactly 0
np.fill_diagonal(distance_matrix, 0)
# Make condensed distance matrix
condensed_dist = squareform(distance_matrix)
# Hierarchical clustering
Z = linkage(condensed_dist, method='average')
order = leaves_list(Z)

# Reorder matrix and labels
sim_matrix_ordered = sim_matrix[order, :][:, order]
valid_refs_ordered = [valid_refs[i] for i in order]

# Plot Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(sim_matrix_ordered, annot=False, cmap="YlGnBu", 
            xticklabels=valid_refs_ordered, yticklabels=valid_refs_ordered)
plt.title("Morgan Fingerprint Tanimoto Similarity Matrix (Radius=2)", fontsize=16)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Save plot
output_path = os.path.join(output_dir, "refrigerant_similarity_matrix.png")
plt.savefig(output_path, dpi=300)
print(f"Similarity matrix saved to {output_path}")

# Print highly similar pairs (excluding self)
print("\nHighly similar pairs (>0.85):")
for i in range(n):
    for j in range(i+1, n):
        if sim_matrix[i, j] > 0.85:
            print(f"{valid_refs[i]} and {valid_refs[j]}: {sim_matrix[i, j]:.4f}")
