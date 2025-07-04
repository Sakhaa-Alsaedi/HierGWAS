"""
Enhanced KGWAS Model with Multi-Scale Hierarchical Attention (MSHA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear
from .msha_conv import MultiScaleHierarchicalAttention, EnhancedSimpleMLP
from .conv import GATConv
from torch_geometric.nn import SAGEConv, GCNConv, SGConv


class EnhancedHeteroGNN(torch.nn.Module):
    """
    Enhanced Heterogeneous Graph Neural Network with Multi-Scale Hierarchical Attention
    
    Key improvements:
    1. Multi-Scale Hierarchical Attention (MSHA) mechanism
    2. Enhanced feature fusion with cross-modal attention
    3. Residual connections and layer normalization
    4. Adaptive regularization
    5. Biological prior integration
    """
    
    def __init__(self, 
                 pyg_data, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 gnn_backbone='MSHA',
                 gnn_aggr='sum',
                 snp_init_dim_size=None,
                 gene_init_dim_size=None,
                 go_init_dim_size=None,
                 gat_num_head=8,
                 num_scales=3,
                 dropout=0.1,
                 biological_prior=True,
                 cross_modal_attention=True,
                 no_relu=False):
        
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.gnn_backbone = gnn_backbone
        self.biological_prior = biological_prior
        self.cross_modal_attention = cross_modal_attention
        self.no_relu = no_relu
        
        edge_types = pyg_data.edge_types
        self.edge_types = edge_types
        
        # Enhanced feature transformation MLPs
        self.snp_feat_mlp = EnhancedSimpleMLP(
            snp_init_dim_size, hidden_channels, hidden_channels, dropout=dropout
        )
        self.go_feat_mlp = EnhancedSimpleMLP(
            go_init_dim_size, hidden_channels, hidden_channels, dropout=dropout
        )
        self.gene_feat_mlp = EnhancedSimpleMLP(
            gene_init_dim_size, hidden_channels, hidden_channels, dropout=dropout
        )
        
        # Cross-modal attention for feature fusion
        if cross_modal_attention:
            self.cross_modal_attn = CrossModalAttention(hidden_channels, gat_num_head)
        
        # Graph convolution layers
        self.convs = torch.nn.ModuleList()
        
        for layer_idx in range(num_layers):
            conv_layer = {}
            
            for edge_type in edge_types:
                if gnn_backbone == 'MSHA':
                    conv_layer[edge_type] = MultiScaleHierarchicalAttention(
                        in_channels=(-1, -1),
                        out_channels=hidden_channels,
                        heads=gat_num_head,
                        num_scales=num_scales,
                        dropout=dropout,
                        biological_prior=biological_prior,
                        add_self_loops=False
                    )
                elif gnn_backbone == 'SAGE':
                    conv_layer[edge_type] = SAGEConv((-1, -1), hidden_channels)
                elif gnn_backbone == 'GAT':
                    conv_layer[edge_type] = GATConv(
                        (-1, -1), hidden_channels, 
                        heads=gat_num_head, 
                        add_self_loops=False
                    )
                elif gnn_backbone == 'GCN':
                    conv_layer[edge_type] = GCNConv(-1, hidden_channels, add_self_loops=False)
                elif gnn_backbone == 'SGC':
                    conv_layer[edge_type] = SGConv(-1, hidden_channels, add_self_loops=False)
                else:
                    raise ValueError(f"Unsupported GNN backbone: {gnn_backbone}")
            
            conv = HeteroConv(conv_layer, aggr=gnn_aggr)
            self.convs.append(conv)
        
        # Layer normalization for each node type
        self.layer_norms = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                'SNP': nn.LayerNorm(hidden_channels),
                'Gene': nn.LayerNorm(hidden_channels),
                'CellularComponent': nn.LayerNorm(hidden_channels),
                'BiologicalProcess': nn.LayerNorm(hidden_channels),
                'MolecularFunction': nn.LayerNorm(hidden_channels),
            }) for _ in range(num_layers)
        ])
        
        # Adaptive dropout
        self.adaptive_dropout = AdaptiveDropout(hidden_channels, dropout)
        
        # Final prediction layers with residual connection
        self.prediction_mlp = nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels // 2, out_channels)
        )
        
        # Residual projection for final layer
        if hidden_channels != out_channels:
            self.final_residual_proj = Linear(hidden_channels, out_channels)
        else:
            self.final_residual_proj = None
        
        self.ReLU = nn.ReLU()
    
    def forward(self, x_dict, edge_index_dict, batch_size, genotype=None, 
                return_h=False, return_attention_weights=False):
        
        # Enhanced feature transformation
        x_dict['SNP'] = self.snp_feat_mlp(x_dict['SNP'])
        x_dict['Gene'] = self.gene_feat_mlp(x_dict['Gene'])
        x_dict['CellularComponent'] = self.go_feat_mlp(x_dict['CellularComponent'])
        x_dict['BiologicalProcess'] = self.go_feat_mlp(x_dict['BiologicalProcess'])
        x_dict['MolecularFunction'] = self.go_feat_mlp(x_dict['MolecularFunction'])
        
        # Cross-modal attention for feature fusion
        if self.cross_modal_attention:
            x_dict = self.cross_modal_attn(x_dict)
        
        # Store attention weights if requested
        all_attention_weights = []
        
        # Graph convolution layers with residual connections
        for layer_idx, (conv, layer_norm) in enumerate(zip(self.convs, self.layer_norms)):
            # Store input for residual connection
            x_dict_residual = {key: x.clone() for key, x in x_dict.items()}
            
            # Apply convolution
            if return_attention_weights and self.gnn_backbone == 'MSHA':
                x_dict_new = {}
                layer_attention_weights = {}
                for node_type in x_dict.keys():
                    # Find edges involving this node type
                    relevant_edges = {k: v for k, v in edge_index_dict.items() 
                                    if k[0] == node_type or k[2] == node_type}
                    if relevant_edges:
                        out, attn_weights = conv(x_dict, relevant_edges, 
                                               return_attention_weights=True)
                        x_dict_new.update(out)
                        layer_attention_weights.update(attn_weights)
                    else:
                        x_dict_new[node_type] = x_dict[node_type]
                x_dict = x_dict_new
                all_attention_weights.append(layer_attention_weights)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            
            # Residual connection
            for key in x_dict.keys():
                if key in x_dict_residual and x_dict[key].shape == x_dict_residual[key].shape:
                    x_dict[key] = x_dict[key] + x_dict_residual[key]
            
            # Layer normalization
            for key in x_dict.keys():
                if key in layer_norm:
                    x_dict[key] = layer_norm[key](x_dict[key])
            
            # Activation
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            
            # Adaptive dropout
            for key in x_dict.keys():
                x_dict[key] = self.adaptive_dropout(x_dict[key])
        
        # Final prediction
        snp_embeddings = x_dict['SNP'][:batch_size]
        
        # Apply prediction MLP
        predictions = self.prediction_mlp(snp_embeddings)
        
        # Final residual connection
        if self.final_residual_proj is not None:
            residual = self.final_residual_proj(snp_embeddings)
            predictions = predictions + residual
        
        # Apply final activation
        if not self.no_relu:
            predictions = self.ReLU(predictions)
        
        # Return based on requested outputs
        if return_h and return_attention_weights:
            return predictions, snp_embeddings, all_attention_weights
        elif return_h:
            return predictions, snp_embeddings
        elif return_attention_weights:
            return predictions, all_attention_weights
        else:
            return predictions


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for better feature fusion across node types
    """
    
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x_dict):
        """
        Apply cross-modal attention across different node types
        """
        # Combine all node features for cross-attention
        all_features = []
        node_type_indices = {}
        start_idx = 0
        
        for node_type, features in x_dict.items():
            all_features.append(features)
            node_type_indices[node_type] = (start_idx, start_idx + features.size(0))
            start_idx += features.size(0)
        
        if not all_features:
            return x_dict
        
        combined_features = torch.cat(all_features, dim=0)  # [total_nodes, hidden_dim]
        
        # Apply multi-head attention
        batch_size, seq_len = combined_features.size(0), 1
        
        # Reshape for multi-head attention
        q = self.q_proj(combined_features).view(batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(combined_features).view(batch_size, self.num_heads, self.head_dim)
        v = self.v_proj(combined_features).view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attended = torch.matmul(attn_weights, v)
        attended = attended.view(batch_size, self.hidden_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + combined_features)
        
        # Split back to original node types
        enhanced_x_dict = {}
        for node_type, (start, end) in node_type_indices.items():
            enhanced_x_dict[node_type] = output[start:end]
        
        return enhanced_x_dict


class AdaptiveDropout(nn.Module):
    """
    Adaptive dropout that adjusts dropout rate based on node importance
    """
    
    def __init__(self, hidden_dim, base_dropout=0.1):
        super().__init__()
        self.base_dropout = base_dropout
        self.importance_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if not self.training:
            return x
        
        # Compute node importance
        importance = self.importance_net(x)  # [N, 1]
        
        # Adaptive dropout rate (higher importance = lower dropout)
        adaptive_dropout_rate = self.base_dropout * (1 - importance)
        
        # Apply dropout with adaptive rates
        dropout_mask = torch.bernoulli(1 - adaptive_dropout_rate).to(x.device)
        return x * dropout_mask / (1 - adaptive_dropout_rate + 1e-8)


# Backward compatibility - alias for the original model
HeteroGNN = EnhancedHeteroGNN

