import os
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT_DIR / "Phase4_Scientific_Validation"
AUDIT_OUT = Path(__file__).resolve().parent / "audit_outputs"
AUDIT_OUT.mkdir(parents=True, exist_ok=True)

DESC_CSV = WORK_DIR / "xTB_Physics_Descriptors.csv"

def get_2d_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
        
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    num_atoms = mol.GetNumAtoms()
    hac = mol.GetNumHeavyAtoms()
    h_donors = rdMolDescriptors.CalcNumHBD(mol)
    h_acceptors = rdMolDescriptors.CalcNumHBA(mol)
    
    # Elemental ratios
    f_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
    c_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    h_count = sum(1 for atom in Chem.AddHs(mol).GetAtoms() if atom.GetSymbol() == 'H')
    
    fc_ratio = f_count / max(c_count, 1)
    hc_ratio = h_count / max(c_count, 1)
    
    # Morgan Fingerprint
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.array(list(fp))
    
    scalar_features = np.array([mw, tpsa, num_atoms, hac, h_donors, h_acceptors, fc_ratio, hc_ratio])
    return scalar_features, fp_array

def run_loocv_evaluation(X_scalar, X_fp, y, target_name):
    loo = LeaveOneOut()
    
    y_pred_ridge = []
    y_pred_rf = []
    y_pred_baseline = []
    y_true = []
    
    for train_index, test_index in loo.split(X_scalar):
        # 1. Split data STRICTLY FIRST
        X_scalar_train, X_scalar_test = X_scalar[train_index], X_scalar[test_index]
        X_fp_train, X_fp_test = X_fp[train_index], X_fp[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 2. PCA on Fingerprints (FIT ON TRAIN ONLY)
        pca = PCA(n_components=min(10, X_fp_train.shape[0] - 1, X_fp_train.shape[1]), random_state=42)
        X_fp_pca_train = pca.fit_transform(X_fp_train)
        X_fp_pca_test = pca.transform(X_fp_test)
        
        # Combine features
        X_train_full = np.hstack((X_scalar_train, X_fp_pca_train))
        X_test_full = np.hstack((X_scalar_test, X_fp_pca_test))
        
        # 3. Scale combined features (FIT ON TRAIN ONLY)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test_full)
        
        # 4. Train Models
        # Ridge with internal CV for hyperparameter tuning
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)
        ridge.fit(X_train_scaled, y_train)
        y_pred_ridge.append(ridge.predict(X_test_scaled)[0])
        
        # Random Forest
        # Fixed, predeclared parameters: no test-fold information is used.
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X_train_scaled, y_train)
        y_pred_rf.append(rf.predict(X_test_scaled)[0])
        
        # Baseline (mean of train)
        y_pred_baseline.append(np.mean(y_train))
        y_true.append(y_test[0])
        
    y_true = np.array(y_true)
    y_pred_ridge = np.array(y_pred_ridge)
    y_pred_rf = np.array(y_pred_rf)
    y_pred_baseline = np.array(y_pred_baseline)
    
    metrics = {
        'Target': target_name,
        'Baseline_MAE': mean_absolute_error(y_true, y_pred_baseline),
        'Ridge_MAE': mean_absolute_error(y_true, y_pred_ridge),
        'Ridge_R2': r2_score(y_true, y_pred_ridge),
        'RF_MAE': mean_absolute_error(y_true, y_pred_rf),
        'RF_R2': r2_score(y_true, y_pred_rf)
    }
    return metrics, y_true, y_pred_ridge, y_pred_rf

def main():
    if not DESC_CSV.is_file():
        print("Descriptor CSV not found.")
        return
        
    df = pd.read_csv(DESC_CSV)
    
    # Filter only refrigerants
    ref_df = df[df['Category'] == 'Refrigerant'].copy()
    ref_df = ref_df.dropna(subset=['Polarizability_au', 'Volume_A3', 'Dipole_Debye'])
    
    print(f"Loaded {len(ref_df)} refrigerants for 2D LOOCV analysis.")
    
    X_scalars = []
    X_fps = []
    
    for smi in ref_df['SMILES']:
        scalar, fp = get_2d_features(smi)
        X_scalars.append(scalar)
        X_fps.append(fp)
        
    X_scalars = np.array(X_scalars)
    X_fps = np.array(X_fps)
    
    targets = {
        'Polarizability (α)': ref_df['Polarizability_au'].values,
        'Volume (V)': ref_df['Volume_A3'].values,
        'Dipole (μ)': ref_df['Dipole_Debye'].values
    }
    
    results = []
    plot_data = {}
    
    for t_name, y in targets.items():
        print(f"\nRunning LOOCV for {t_name}...")
        metrics, y_true, y_ridge, y_rf = run_loocv_evaluation(X_scalars, X_fps, y, t_name)
        results.append(metrics)
        plot_data[t_name] = {'True': y_true, 'Ridge': y_ridge, 'RF': y_rf}
        print(f"  Ridge R2: {metrics['Ridge_R2']:.3f} | MAE: {metrics['Ridge_MAE']:.3f}")
        print(f"  RF    R2: {metrics['RF_R2']:.3f}    | MAE: {metrics['RF_MAE']:.3f}")
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(AUDIT_OUT / "2d_predictability_results.csv", index=False)
    
    # Plotting Information Gain
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (t_name, data) in enumerate(plot_data.items()):
        ax = axes[i]
        
        # Use Ridge results for the plot
        y_true = data['True']
        y_pred = data['Ridge']
        
        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
        
        # Plot y=x line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{t_name}\nRidge LOOCV $R^2$ = {r2:.3f}")
        ax.set_xlabel("True (xTB)")
        ax.set_ylabel("Predicted (2D Features)")
        ax.grid(True, linestyle='--', alpha=0.5)
        
    plt.suptitle("Information Gain: Can 2D Predict 3D Physics? (Strict LOOCV on N=26 refrigerants)", y=1.05)
    plt.tight_layout()
    plt.savefig(AUDIT_OUT / "2d_vs_3d_information_gain.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved analysis to {AUDIT_OUT / '2d_vs_3d_information_gain.png'}")
    
    # Also create the bar chart for R2
    plt.figure(figsize=(8, 5))
    x = np.arange(3)
    width = 0.35
    
    r2_ridge = results_df['Ridge_R2'].values
    r2_rf = results_df['RF_R2'].values
    
    plt.bar(x - width/2, r2_ridge, width, label='Ridge')
    plt.bar(x + width/2, r2_rf, width, label='Random Forest')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.ylabel('LOOCV $R^2$')
    plt.title('Predictability of xTB Descriptors from 2D Topology (在本数据集上)')
    plt.xticks(x, results_df['Target'].values)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(AUDIT_OUT / "2d_predictability_r2_bars.png", dpi=300)

if __name__ == "__main__":
    main()
