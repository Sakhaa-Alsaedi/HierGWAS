"""
Enhanced KGWAS Class with Multi-Scale Hierarchical Attention and Advanced Training
"""

import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from .enhanced_model import EnhancedHeteroGNN
from .utils import print_sys, compute_metrics, save_dict, load_dict, save_model
from .eval_utils import storey_ribshirani_integrate, get_clumps_gold_label


class EnhancedKGWAS:
    """
    Enhanced KGWAS with Multi-Scale Hierarchical Attention and improved training strategies
    
    Key improvements:
    1. Multi-Scale Hierarchical Attention (MSHA) model
    2. Advanced training strategies (learning rate scheduling, early stopping)
    3. Enhanced evaluation metrics and visualization
    4. Biological prior integration
    5. Attention weight analysis
    """
    
    def __init__(self,
                 data,
                 weight_bias_track=False,
                 device='cuda',
                 proj_name='Enhanced_KGWAS',
                 exp_name='MSHA_KGWAS',
                 seed=42):
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        
        # Device setup
        use_cuda = torch.cuda.is_available()
        self.device = device if use_cuda else "cpu"
        print_sys(f"Using device: {self.device}")
        
        # Data and experiment setup
        self.data = data
        self.data_path = data.data_path
        self.exp_name = exp_name
        
        # Weights & Biases tracking
        if weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)
            self.wandb = wandb
        else:
            self.wandb = False
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'learning_rates': []
        }
    
    def initialize_model(self, 
                        gnn_num_layers=3,
                        gnn_hidden_dim=128,
                        gnn_backbone='MSHA',
                        gnn_aggr='sum',
                        gat_num_head=8,
                        num_scales=3,
                        dropout=0.1,
                        biological_prior=True,
                        cross_modal_attention=True,
                        no_relu=False):
        """
        Initialize the enhanced KGWAS model with MSHA
        """
        
        self.config = {
            'gnn_num_layers': gnn_num_layers,
            'gnn_hidden_dim': gnn_hidden_dim,
            'gnn_backbone': gnn_backbone,
            'gnn_aggr': gnn_aggr,
            'gat_num_head': gat_num_head,
            'num_scales': num_scales,
            'dropout': dropout,
            'biological_prior': biological_prior,
            'cross_modal_attention': cross_modal_attention,
            'no_relu': no_relu
        }
        
        self.gnn_num_layers = gnn_num_layers
        
        # Initialize enhanced model
        self.model = EnhancedHeteroGNN(
            pyg_data=self.data.data,
            hidden_channels=gnn_hidden_dim,
            out_channels=1,
            num_layers=gnn_num_layers,
            gnn_backbone=gnn_backbone,
            gnn_aggr=gnn_aggr,
            snp_init_dim_size=self.data.snp_init_dim_size,
            gene_init_dim_size=self.data.gene_init_dim_size,
            go_init_dim_size=self.data.go_init_dim_size,
            gat_num_head=gat_num_head,
            num_scales=num_scales,
            dropout=dropout,
            biological_prior=biological_prior,
            cross_modal_attention=cross_modal_attention,
            no_relu=no_relu
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print_sys(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    def train(self, 
              batch_size=512,
              num_workers=0,
              lr=1e-4,
              weight_decay=5e-4,
              epoch=20,
              patience=5,
              min_delta=1e-4,
              lr_scheduler='cosine',
              warmup_epochs=3,
              save_best_model=True,
              save_name=None,
              data_to_cuda=False,
              eval_every=1):
        """
        Enhanced training with advanced strategies
        """
        
        total_epoch = epoch
        if save_name is None:
            save_name = self.exp_name
        self.save_name = save_name
        
        print_sys('Creating enhanced data loaders...')
        
        # Data loader configuration
        kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'drop_last': True}
        eval_kwargs = {'batch_size': 512, 'num_workers': num_workers, 'drop_last': False}
        
        if data_to_cuda:
            self.data.data = self.data.data.to(self.device)
        
        # Create data loaders
        self.train_loader = NeighborLoader(
            self.data.data, 
            num_neighbors=[-1] * self.gnn_num_layers,
            input_nodes=('SNP', self.data.train_idx),
            **kwargs
        )
        
        self.val_loader = NeighborLoader(
            self.data.data,
            num_neighbors=[-1] * self.gnn_num_layers,
            input_nodes=('SNP', self.data.val_idx),
            **eval_kwargs
        )
        
        self.test_loader = NeighborLoader(
            self.data.data,
            num_neighbors=[-1] * self.gnn_num_layers,
            input_nodes=('SNP', self.data.test_idx),
            **eval_kwargs
        )
        
        # Optimizer setup
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=lr/100
            )
        elif lr_scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3, verbose=True
            )
        else:
            scheduler = None
        
        # Early stopping setup
        best_val_auc = 0
        patience_counter = 0
        best_model_state = None
        
        print_sys(f'Starting enhanced training for {total_epoch} epochs...')
        
        for epoch_idx in range(total_epoch):
            # Training phase
            train_loss, train_auc = self._train_epoch(optimizer, epoch_idx, warmup_epochs, lr)
            
            # Validation phase
            if epoch_idx % eval_every == 0:
                val_loss, val_auc, val_metrics = self._validate_epoch()
                
                # Update training history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['train_auc'].append(train_auc)
                self.training_history['val_auc'].append(val_auc)
                self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
                
                # Learning rate scheduling
                if scheduler is not None:
                    if lr_scheduler == 'plateau':
                        scheduler.step(val_auc)
                    else:
                        scheduler.step()
                
                # Early stopping check
                if val_auc > best_val_auc + min_delta:
                    best_val_auc = val_auc
                    patience_counter = 0
                    if save_best_model:
                        best_model_state = deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                
                # Logging
                print_sys(f'Epoch {epoch_idx+1}/{total_epoch}: '
                         f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, '
                         f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, '
                         f'LR: {optimizer.param_groups[0][\"lr\"]:.2e}')
                
                if self.wandb:
                    self.wandb.log({
                        'epoch': epoch_idx,
                        'train_loss': train_loss,
                        'train_auc': train_auc,
                        'val_loss': val_loss,
                        'val_auc': val_auc,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })
                
                # Early stopping
                if patience_counter >= patience:
                    print_sys(f'Early stopping triggered after {epoch_idx+1} epochs')
                    break
        
        # Load best model
        if save_best_model and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.best_model = self.model
            print_sys(f'Loaded best model with validation AUC: {best_val_auc:.4f}')
        
        # Final evaluation
        print_sys('Performing final evaluation...')
        test_loss, test_auc, test_metrics = self._test_epoch()
        
        print_sys(f'Final Test Results: Loss: {test_loss:.4f}, AUC: {test_auc:.4f}')
        print_sys(f'Test Metrics: {test_metrics}')
        
        return {
            'best_val_auc': best_val_auc,
            'test_auc': test_auc,
            'test_metrics': test_metrics,
            'training_history': self.training_history
        }
    
    def _train_epoch(self, optimizer, epoch_idx, warmup_epochs, base_lr):
        """Training for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Warmup learning rate
        if epoch_idx < warmup_epochs:
            lr_scale = (epoch_idx + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * lr_scale
        
        for batch in tqdm(self.train_loader, desc=f'Training Epoch {epoch_idx+1}'):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass
            out = self.model(
                batch.x_dict,
                batch.edge_index_dict,
                batch_size=batch['SNP'].batch_size
            )
            
            # Compute loss
            labels = batch['SNP'].y[:batch['SNP'].batch_size].float()
            loss = F.binary_cross_entropy_with_logits(out.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            with torch.no_grad():
                preds = torch.sigmoid(out.squeeze()).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        auc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, auc
    
    def _validate_epoch(self):
        """Validation for one epoch"""
        return self._evaluate_loader(self.val_loader, 'Validation')
    
    def _test_epoch(self):
        """Test evaluation"""
        return self._evaluate_loader(self.test_loader, 'Test')
    
    def _evaluate_loader(self, loader, phase_name):
        """Generic evaluation function"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'{phase_name} Evaluation'):
                batch = batch.to(self.device)
                
                # Forward pass
                out = self.model(
                    batch.x_dict,
                    batch.edge_index_dict,
                    batch_size=batch['SNP'].batch_size
                )
                
                # Compute loss
                labels = batch['SNP'].y[:batch['SNP'].batch_size].float()
                loss = F.binary_cross_entropy_with_logits(out.squeeze(), labels)
                
                # Accumulate metrics
                total_loss += loss.item()
                preds = torch.sigmoid(out.squeeze()).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        auc = roc_auc_score(all_labels, all_preds)
        
        # Additional metrics
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        pr_auc = auc(recall, precision)
        
        metrics = {
            'auc': auc,
            'pr_auc': pr_auc,
            'precision': precision,
            'recall': recall
        }
        
        return avg_loss, auc, metrics
    
    def analyze_attention_weights(self, num_samples=100):
        """
        Analyze attention weights from the MSHA mechanism
        """
        if self.config['gnn_backbone'] != 'MSHA':
            print_sys("Attention analysis only available for MSHA backbone")
            return None
        
        self.model.eval()
        attention_data = []
        
        with torch.no_grad():
            sample_count = 0
            for batch in self.test_loader:
                if sample_count >= num_samples:
                    break
                
                batch = batch.to(self.device)
                
                # Get predictions with attention weights
                out, attention_weights = self.model(
                    batch.x_dict,
                    batch.edge_index_dict,
                    batch_size=batch['SNP'].batch_size,
                    return_attention_weights=True
                )
                
                attention_data.append({
                    'predictions': out.cpu(),
                    'attention_weights': attention_weights,
                    'labels': batch['SNP'].y[:batch['SNP'].batch_size].cpu()
                })
                
                sample_count += batch['SNP'].batch_size
        
        return attention_data
    
    def save_model(self, path):
        """Save the enhanced model"""
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pth'))
        
        # Save configuration
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
        
        # Save training history
        with open(os.path.join(path, 'training_history.pkl'), 'wb') as f:
            pickle.dump(self.training_history, f)
        
        print_sys(f"Enhanced model saved to {path}")
    
    def load_model(self, path):
        """Load the enhanced model"""
        # Load configuration
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        # Initialize model with loaded config
        self.initialize_model(**config)
        
        # Load model state
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=self.device))
        
        # Load training history if available
        history_path = os.path.join(path, 'training_history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                self.training_history = pickle.load(f)
        
        print_sys(f"Enhanced model loaded from {path}")


# Backward compatibility
KGWAS = EnhancedKGWAS

