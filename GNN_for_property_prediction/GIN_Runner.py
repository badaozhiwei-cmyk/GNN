import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import random
import os
import pandas as pd

from Dataset import IL_set
from Model import GIN

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

Args = {
    # general arguments
    'smiles_dict_path':'data/smiles.csv',
    'data_path':'processed_tri_data/',
    'load_history_model':False,
    'batch_size':64,
    'lr':0.001,
    'epoch':150,
    'weight_decay':1e-6,

    # GIN model hyperparameter
    'num_gin_layer':5,
    'emb_dim':300,
    'feat_dim':512,
    'drop_ratio':0.2,
    'pool':'mean',
    'warmup':40,
    'patience': 30  # Early stopping patience
}

class Runner(object):
    """
    include all the function needed for training
    """
    def __init__(self,args, seed=42):
        self.args = args
        self.seed = seed
        self._device = self._get_device()

        if args['load_history_model'] == True:
            print("loading history model..")
            self._model = GIN(args)
            state_dict_mod = torch.load('pretrained_model/GIN_300/best_model_para.pth', map_location=self._device)
            self._model.load_state_dict(state_dict_mod)
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
            self._scheduler = CosineAnnealingLR(self._optimizer, T_max=args['epoch']-9)
            print("finish loading")
        else:
            self._model = GIN(args)
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
            self._scheduler = CosineAnnealingLR(self._optimizer, T_max=args['epoch'] - 9)

        self._criterion = nn.L1Loss()


    def _get_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on:", device)
        return device

    def _save_para(self,title):
        save_dir = 'checkpoints'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        filename = f"{save_dir}/{title}_seed_{self.seed}.pth"
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict(),
            'seed': self.seed,
            'args': self.args
        }, filename)


    def train(self,train_loader,dev_loader,args):
        model = self._model.to(self._device)
        optimizer = self._optimizer
        scheduler = self._scheduler
        early_stopping = EarlyStopping(patience=args['patience'], verbose=False)

        best_v_loss = float('inf')
        
        for epoch in range(1,args['epoch'] + 1):
            model.train()
            train_loss = 0
            batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, position=0, leave=False, desc=f'Epoch {epoch} Train')
            
            for batch_idx,(graph,cond,label) in enumerate(train_loader):
                graph = graph.to(self._device)
                cond = cond.to(self._device)
                label = label.to(self._device)

                optimizer.zero_grad()
                y = model(graph,cond)
                loss = self._criterion(y.flatten(),label.flatten())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batch_bar.set_postfix(loss="{:.04f}".format(train_loss/(batch_idx + 1)), lr="{:.06f}".format(optimizer.param_groups[0]['lr']))
                batch_bar.update()
            batch_bar.close()

            # validate process
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for graph,cond,label in dev_loader:
                    graph = graph.to(self._device)
                    cond = cond.to(self._device)
                    label = label.to(self._device)
                    y = model(graph,cond)
                    val_loss += self._criterion(y.flatten(),label.flatten()).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(dev_loader)

            if avg_val_loss < best_v_loss:
                best_v_loss = avg_val_loss
                self._save_para('best')

            if epoch >= args['warmup']:
                scheduler.step()

            print(f"Epoch {epoch}/{args['epoch']}: Train loss {avg_train_loss:.04f}, Val loss {avg_val_loss:.04f}, LR {optimizer.param_groups[0]['lr']:.06f}")
            
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        return best_v_loss

    def test(self,test_loader):
        model = self._model.to(self._device)
        # Load best model for testing
        checkpoint_path = f"checkpoints/best_seed_{self.seed}.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        pred_y = []
        true_y = []
        model.eval()
        with torch.no_grad():
            for graph,cond,label in tqdm(test_loader, desc='Testing', leave=False):
                graph = graph.to(self._device)
                cond = cond.to(self._device)
                pred = model(graph,cond)
                pred_y.extend(pred.flatten().cpu().numpy().tolist())
                true_y.extend(label.flatten().numpy().tolist())

        mae = mean_absolute_error(true_y, pred_y)
        r2 = r2_score(true_y,pred_y)
        print(f"Test MAE: {mae:.4f}, R2: {r2:.4f}")
        return pred_y,true_y

def plot_results(true_y, pred_y, title, filename):
    plt.figure(figsize=(8, 8))
    plt.scatter(true_y, pred_y, alpha=0.5, color='royalblue', label='Predictions')
    
    # Perfect prediction line
    lims = [min(min(true_y), min(pred_y)), max(max(true_y), max(pred_y))]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Ideal')
    
    plt.xlabel('Experimental Value', fontsize=12)
    plt.ylabel('Predicted Value', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if not os.path.exists('figure'):
        os.makedirs('figure')
    plt.savefig(f"figure/{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    seeds = [42, 123, 2024, 7, 88]
    ensemble_results = []
    
    # Global containers for ensemble visualization
    all_test_true = []
    all_test_pred = []

    for seed in seeds:
        print(f"\n{'='*20} Training Seed: {seed} {'='*20}")
        set_seed(seed)
        
        Whole_set = IL_set(path = Args['data_path'])
        train_size = int(len(Whole_set) * 0.7)
        test_size  = int(len(Whole_set) * 0.2)
        dev_size   = len(Whole_set) - train_size - test_size
        
        train_set, dev_set, test_set = random_split(
            Whole_set, [train_size, dev_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(train_set, batch_size=Args['batch_size'], shuffle=True)
        dev_loader   = DataLoader(dev_set,   batch_size=Args['batch_size'], shuffle=False)
        test_loader  = DataLoader(test_set,  batch_size=Args['batch_size'], shuffle=False)

        run_G = Runner(Args, seed=seed)
        if Args['epoch'] != 0:
            run_G.train(train_loader, dev_loader, Args)

        test_pred, test_true = run_G.test(test_loader)
        
        mae = mean_absolute_error(test_true, test_pred)
        r2 = r2_score(test_true, test_pred)
        ensemble_results.append({'seed': seed, 'mae': mae, 'r2': r2})
        
        all_test_true.extend(test_true)
        all_test_pred.extend(test_pred)
        
        # Plot individual seed results
        plot_results(test_true, test_pred, f"Seed {seed} Prediction", f"pred_seed_{seed}")

    # Summary
    print(f"\n{'='*20} Ensemble Summary {'='*20}")
    df_results = pd.DataFrame(ensemble_results)
    print(df_results)
    
    avg_mae = df_results['mae'].mean()
    std_mae = df_results['mae'].std()
    avg_r2 = df_results['r2'].mean()
    
    print(f"\nAverage Test MAE: {avg_mae:.4f} (+/- {std_mae:.4f})")
    print(f"Average Test R2:  {avg_r2:.4f}")
    
    # Save results to CSV
    df_results.to_csv('ensemble_results_summary.csv', index=False)
    print("Summary saved to ensemble_results_summary.csv")

    # Plot total ensemble results
    plot_results(all_test_true, all_test_pred, "Ensemble Prediction Performance", "ensemble_prediction_final")




