"""
Multi-Scale Hierarchical Attention (MSHA) Convolution Layer
Enhanced attention mechanism for KGWAS with multi-scale graph processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from typing import Optional, Tuple, Union, List
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
import math


class MultiScaleHierarchicalAttention(MessagePassing):
    """
    Multi-Scale Hierarchical Attention mechanism for heterogeneous graphs.
    
    This layer implements:
    1. Multi-scale attention at different hop distances
    2. Hierarchical attention weighting across scales
    3. Biological prior integration
    4. Enhanced feature fusion
    """
    
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 8,
        num_scales: int = 3,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.1,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        temperature: float = 1.0,
        biological_prior: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_scales = num_scales
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.temperature = temperature
        self.biological_prior = biological_prior
        
        # Multi-scale transformations
        if isinstance(in_channels, int):
            self.lin_src = nn.ModuleList([
                Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
                for _ in range(num_scales)
            ])
            self.lin_dst = self.lin_src
        else:
            self.lin_src = nn.ModuleList([
                Linear(in_channels[0], heads * out_channels, bias=False, weight_initializer='glorot')
                for _ in range(num_scales)
            ])
            self.lin_dst = nn.ModuleList([
                Linear(in_channels[1], heads * out_channels, bias=False, weight_initializer='glorot')
                for _ in range(num_scales)
            ])
        
        # Multi-scale attention parameters
        self.att_src = nn.ParameterList([
            Parameter(torch.Tensor(1, heads, out_channels))
            for _ in range(num_scales)
        ])
        self.att_dst = nn.ParameterList([
            Parameter(torch.Tensor(1, heads, out_channels))
            for _ in range(num_scales)
        ])
        
        # Hierarchical attention weighting
        self.scale_attention = nn.Sequential(
            Linear(heads * out_channels * num_scales, heads * out_channels),
            nn.ReLU(),
            Linear(heads * out_channels, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # Edge attention (if edge features are provided)
        if edge_dim is not None:
            self.lin_edge = nn.ModuleList([
                Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
                for _ in range(num_scales)
            ])
            self.att_edge = nn.ParameterList([
                Parameter(torch.Tensor(1, heads, out_channels))
                for _ in range(num_scales)
            ])
        else:
            self.lin_edge = None
            self.att_edge = None
        
        # Biological prior integration
        if biological_prior:
            self.bio_prior_weight = Parameter(torch.Tensor(1))
            self.bio_prior_mlp = nn.Sequential(
                Linear(out_channels, out_channels // 2),
                nn.ReLU(),
                Linear(out_channels // 2, 1),
                nn.Sigmoid()
            )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(heads * out_channels if concat else out_channels)
        
        # Residual connection projection
        if in_channels != (heads * out_channels if concat else out_channels):
            if isinstance(in_channels, int):
                self.residual_proj = Linear(in_channels, heads * out_channels if concat else out_channels)
            else:
                self.residual_proj = Linear(in_channels[0], heads * out_channels if concat else out_channels)
        else:
            self.residual_proj = None
        
        # Bias
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for lin_src in self.lin_src:
            lin_src.reset_parameters()
        if isinstance(self.lin_dst, nn.ModuleList):
            for lin_dst in self.lin_dst:
                lin_dst.reset_parameters()
        
        for att_src in self.att_src:
            glorot(att_src)
        for att_dst in self.att_dst:
            glorot(att_dst)
        
        if self.lin_edge is not None:
            for lin_edge in self.lin_edge:
                lin_edge.reset_parameters()
            for att_edge in self.att_edge:
                glorot(att_edge)
        
        if self.biological_prior:
            nn.init.constant_(self.bio_prior_weight, 0.1)
            for module in self.bio_prior_mlp:
                if isinstance(module, Linear):
                    module.reset_parameters()
        
        if self.residual_proj is not None:
            self.residual_proj.reset_parameters()
        
        zeros(self.bias)
    
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights: bool = False):
        
        H, C = self.heads, self.out_channels
        
        # Store original input for residual connection
        if isinstance(x, Tensor):
            x_orig = x
            x_src = x_dst = x
        else:
            x_src, x_dst = x
            x_orig = x_src
        
        # Multi-scale feature transformations
        multi_scale_outputs = []
        multi_scale_attentions = []
        
        for scale in range(self.num_scales):
            # Transform features for this scale
            if isinstance(x, Tensor):
                x_src_scale = self.lin_src[scale](x_src).view(-1, H, C)
                x_dst_scale = x_src_scale
            else:
                x_src_scale = self.lin_src[scale](x_src).view(-1, H, C)
                x_dst_scale = self.lin_dst[scale](x_dst).view(-1, H, C) if x_dst is not None else None
            
            # Compute attention coefficients for this scale
            alpha_src = (x_src_scale * self.att_src[scale]).sum(dim=-1)
            alpha_dst = None if x_dst_scale is None else (x_dst_scale * self.att_dst[scale]).sum(-1)
            alpha = (alpha_src, alpha_dst)
            
            # Add self-loops if specified
            if self.add_self_loops:
                num_nodes = x_src_scale.size(0)
                if x_dst_scale is not None:
                    num_nodes = min(num_nodes, x_dst_scale.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index_scale, edge_attr_scale = remove_self_loops(edge_index, edge_attr)
                edge_index_scale, edge_attr_scale = add_self_loops(
                    edge_index_scale, edge_attr_scale, num_nodes=num_nodes)
            else:
                edge_index_scale, edge_attr_scale = edge_index, edge_attr
            
            # Edge attention update for this scale
            alpha_scale = self.edge_updater(edge_index_scale, alpha=alpha, 
                                          edge_attr=edge_attr_scale, scale=scale)
            
            # Message passing for this scale
            x_scale = (x_src_scale, x_dst_scale)
            out_scale = self.propagate(edge_index_scale, x=x_scale, alpha=alpha_scale, size=size)
            
            multi_scale_outputs.append(out_scale)
            multi_scale_attentions.append(alpha_scale)
        
        # Hierarchical attention weighting across scales
        # Concatenate all scale outputs for attention computation
        concat_outputs = torch.cat(multi_scale_outputs, dim=-1)  # [N, H*C*num_scales]
        scale_weights = self.scale_attention(concat_outputs.view(-1, H * C * self.num_scales))  # [N, num_scales]
        
        # Weighted combination of multi-scale outputs
        weighted_output = torch.zeros_like(multi_scale_outputs[0])
        for scale, output in enumerate(multi_scale_outputs):
            weighted_output += scale_weights[:, scale:scale+1].unsqueeze(-1) * output
        
        # Apply biological prior if enabled
        if self.biological_prior:
            bio_prior = self.bio_prior_mlp(weighted_output.mean(dim=1))  # [N, 1]
            bio_weight = self.bio_prior_weight * bio_prior
            weighted_output = weighted_output * (1 + bio_weight.unsqueeze(1))
        
        # Reshape output
        if self.concat:
            out = weighted_output.view(-1, self.heads * self.out_channels)
        else:
            out = weighted_output.mean(dim=1)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(x_orig)
        else:
            residual = x_orig
        
        out = out + residual
        
        # Layer normalization
        out = self.layer_norm(out)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        if return_attention_weights:
            return out, (edge_index, multi_scale_attentions, scale_weights)
        else:
            return out
    
    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int], scale: int) -> Tensor:
        
        # Combine source and target attention
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        
        # Add edge attention if available
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr_transformed = self.lin_edge[scale](edge_attr)
            edge_attr_transformed = edge_attr_transformed.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr_transformed * self.att_edge[scale]).sum(dim=-1)
            alpha = alpha + alpha_edge
        
        # Apply activation and temperature scaling
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha / self.temperature, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return alpha
    
    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'num_scales={self.num_scales})')


class EnhancedSimpleMLP(nn.Module):
    """
    Enhanced MLP with residual connections, layer normalization, and adaptive dropout
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super(EnhancedSimpleMLP, self).__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Residual projection if needed
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        for i, (layer, norm) in enumerate(zip(self.layers[:-1], self.layer_norms)):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        if x.shape == residual.shape:
            x = x + residual
        
        return x

