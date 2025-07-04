"""
Hierarchical Multi-Scale Attention (HMSA) Module

This module implements the core hierarchical multi-scale attention mechanism
that captures genomic interactions at different biological scales.

The HMSA mechanism operates at three biological scales:
1. Local Scale (1-10 kb): SNP-SNP interactions within genes
2. Regional Scale (10-100 kb): Gene-gene interactions within pathways  
3. Global Scale (>100 kb): Pathway-pathway interactions across chromosomes

Key Features:
- Scale-specific attention computation with different receptive fields
- Hierarchical weighting to learn optimal scale combinations
- Biological prior integration for enhanced interpretability
- Cross-scale information fusion for comprehensive modeling

Example:
    >>> from hiergwas.attention import HierarchicalMultiScaleAttention
    >>> 
    >>> # Initialize HMSA layer
    >>> hmsa = HierarchicalMultiScaleAttention(
    ...     in_channels=128,
    ...     out_channels=128,
    ...     num_scales=3,
    ...     heads=8
    ... )
    >>> 
    >>> # Apply to genomic features
    >>> output, attention_weights = hmsa(x, edge_index, return_attention_weights=True)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.typing import Adj, OptTensor, PairTensor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class HierarchicalMultiScaleAttention(MessagePassing):
    """
    Hierarchical Multi-Scale Attention (HMSA) Layer
    
    This layer implements the core innovation of HierGWAS: a hierarchical attention
    mechanism that captures genomic interactions at multiple biological scales
    simultaneously.
    
    The attention mechanism works by:
    1. Computing scale-specific attention patterns with different receptive fields
    2. Learning hierarchical weights to optimally combine scales
    3. Integrating biological priors for enhanced interpretability
    4. Enabling cross-scale information fusion
    
    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        num_scales (int): Number of biological scales (typically 3)
        heads (int): Number of attention heads per scale
        dropout (float): Dropout rate for attention weights
        bias (bool): Whether to use bias in linear transformations
        biological_prior (bool): Whether to integrate biological priors
        cross_scale_fusion (bool): Whether to enable cross-scale fusion
        scale_temperatures (List[float], optional): Temperature parameters for each scale
        
    Example:
        >>> # Basic usage
        >>> hmsa = HierarchicalMultiScaleAttention(
        ...     in_channels=128,
        ...     out_channels=128,
        ...     num_scales=3,
        ...     heads=8
        ... )
        >>> 
        >>> # Advanced usage with biological priors
        >>> hmsa = HierarchicalMultiScaleAttention(
        ...     in_channels=256,
        ...     out_channels=256,
        ...     num_scales=4,
        ...     heads=16,
        ...     biological_prior=True,
        ...     cross_scale_fusion=True,
        ...     scale_temperatures=[1.0, 0.8, 0.6, 0.4]
        ... )
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_scales: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        biological_prior: bool = False,
        cross_scale_fusion: bool = True,
        scale_temperatures: Optional[List[float]] = None,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales
        self.heads = heads
        self.dropout = dropout
        self.biological_prior = biological_prior
        self.cross_scale_fusion = cross_scale_fusion
        
        # Scale-specific temperature parameters
        if scale_temperatures is None:
            # Default: decreasing temperature for larger scales
            self.scale_temperatures = [1.0 - 0.2 * i for i in range(num_scales)]
        else:
            self.scale_temperatures = scale_temperatures
        
        assert len(self.scale_temperatures) == num_scales, \
            f"Number of temperatures ({len(self.scale_temperatures)}) must match num_scales ({num_scales})"
        
        # Ensure output channels are divisible by heads
        assert out_channels % heads == 0, \
            f"out_channels ({out_channels}) must be divisible by heads ({heads})"
        
        self.head_dim = out_channels // heads
        
        # Scale-specific transformations
        self.scale_transforms = nn.ModuleList([
            nn.Linear(in_channels, heads * self.head_dim, bias=bias)
            for _ in range(num_scales)
        ])
        
        # Scale-specific attention parameters
        self.scale_att_weights = nn.ModuleList([
            nn.Linear(2 * self.head_dim, 1, bias=False)
            for _ in range(num_scales)
        ])
        
        # Hierarchical weighting network
        self.hierarchical_weights = nn.Sequential(
            nn.Linear(heads * self.head_dim * num_scales, heads * self.head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(heads * self.head_dim, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # Cross-scale fusion if enabled
        if cross_scale_fusion:
            self.cross_scale_attention = nn.MultiheadAttention(
                embed_dim=heads * self.head_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Biological prior integration
        if biological_prior:
            self.bio_prior_mlp = nn.Sequential(
                nn.Linear(heads * self.head_dim, heads * self.head_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(heads * self.head_dim // 2, 1),
                nn.Sigmoid()
            )
            self.bio_prior_weight = nn.Parameter(torch.tensor(0.1))
        
        # Output projection
        self.output_proj = nn.Linear(heads * self.head_dim, out_channels, bias=bias)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(out_channels)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for transform in self.scale_transforms:
            nn.init.xavier_uniform_(transform.weight)
            if transform.bias is not None:
                nn.init.zeros_(transform.bias)
        
        for att_weight in self.scale_att_weights:
            nn.init.xavier_uniform_(att_weight.weight)
        
        # Initialize hierarchical weights
        for layer in self.hierarchical_weights:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
        
        # Initialize biological prior parameters
        if self.biological_prior:
            for layer in self.bio_prior_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x: Union[torch.Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Optional[Tuple[int, int]] = None,
        return_attention_weights: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass of the Hierarchical Multi-Scale Attention layer.
        
        Args:
            x (Tensor or PairTensor): Node features
            edge_index (Adj): Edge indices
            edge_attr (Tensor, optional): Edge attributes
            size (Tuple[int, int], optional): Size of the bipartite graph
            return_attention_weights (bool): Whether to return attention weights
            
        Returns:
            Tensor or Tuple: Output features, optionally with attention weights
            
        Example:
            >>> # Basic forward pass
            >>> output = hmsa(x, edge_index)
            >>> 
            >>> # Forward pass with attention analysis
            >>> output, attention_weights = hmsa(
            ...     x, edge_index, return_attention_weights=True
            ... )
            >>> 
            >>> # Access scale-specific attention
            >>> local_attention = attention_weights['scale_attention'][0]
            >>> hierarchical_weights = attention_weights['hierarchical_weights']
        """
        # Handle input format
        if isinstance(x, torch.Tensor):
            x_src = x_dst = x
        else:
            x_src, x_dst = x
        
        # Store original features for residual connection
        residual = x_dst
        
        # Compute scale-specific features and attention
        scale_outputs = []
        scale_attention_weights = []
        
        for scale_idx in range(self.num_scales):
            # Scale-specific transformation
            x_transformed = self.scale_transforms[scale_idx](x_src)
            x_transformed = x_transformed.view(-1, self.heads, self.head_dim)
            
            # Compute scale-specific attention
            scale_output, scale_attn = self._compute_scale_attention(
                x_transformed,
                x_dst,
                edge_index,
                scale_idx,
                size,
                return_attention_weights
            )
            
            scale_outputs.append(scale_output)
            if return_attention_weights:
                scale_attention_weights.append(scale_attn)
        
        # Concatenate all scale outputs for hierarchical weighting
        all_scale_features = torch.cat(scale_outputs, dim=-1)  # [N, heads * head_dim * num_scales]
        
        # Compute hierarchical weights
        hierarchical_weights = self.hierarchical_weights(all_scale_features)  # [N, num_scales]
        
        # Weighted combination of scales
        weighted_output = torch.zeros_like(scale_outputs[0])
        for scale_idx, scale_output in enumerate(scale_outputs):
            weight = hierarchical_weights[:, scale_idx:scale_idx+1].unsqueeze(-1)  # [N, 1, 1]
            weighted_output += weight * scale_output
        
        # Cross-scale fusion if enabled
        if self.cross_scale_fusion:
            # Prepare for cross-attention: [num_scales, N, heads * head_dim]
            scale_stack = torch.stack(scale_outputs, dim=0).transpose(0, 1)  # [N, num_scales, heads * head_dim]
            
            # Apply cross-scale attention
            fused_output, _ = self.cross_scale_attention(
                scale_stack, scale_stack, scale_stack
            )
            
            # Combine with weighted output
            fused_output = fused_output.mean(dim=1)  # Average across scales
            weighted_output = 0.7 * weighted_output + 0.3 * fused_output
        
        # Biological prior integration if enabled
        if self.biological_prior:
            bio_prior = self.bio_prior_mlp(weighted_output.mean(dim=1, keepdim=True))
            bio_weight = self.bio_prior_weight * bio_prior
            weighted_output = weighted_output * (1 + bio_weight)
        
        # Output projection
        output = self.output_proj(weighted_output)
        
        # Residual connection and layer normalization
        if output.shape == residual.shape:
            output = output + residual
        output = self.layer_norm(output)
        
        # Dropout
        output = self.dropout_layer(output)
        
        if return_attention_weights:
            attention_dict = {
                'scale_attention': scale_attention_weights,
                'hierarchical_weights': hierarchical_weights,
                'scale_outputs': scale_outputs
            }
            return output, attention_dict
        else:
            return output
    
    def _compute_scale_attention(
        self,
        x_transformed: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: Adj,
        scale_idx: int,
        size: Optional[Tuple[int, int]],
        return_attention_weights: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention for a specific biological scale.
        
        This method implements scale-specific attention computation with
        different receptive fields and temperature parameters for each scale.
        
        Args:
            x_transformed (Tensor): Scale-specific transformed features
            x_dst (Tensor): Destination node features
            edge_index (Adj): Edge indices
            scale_idx (int): Index of the current scale
            size (Tuple[int, int], optional): Size of the bipartite graph
            return_attention_weights (bool): Whether to return attention weights
            
        Returns:
            Tuple: Scale-specific output and attention weights (if requested)
        """
        # Get scale-specific temperature
        temperature = self.scale_temperatures[scale_idx]
        
        # Propagate messages with scale-specific attention
        output = self.propagate(
            edge_index,
            x=x_transformed,
            size=size,
            scale_idx=scale_idx,
            temperature=temperature,
            return_attention_weights=return_attention_weights
        )
        
        # Reshape output
        output = output.view(-1, self.heads * self.head_dim)
        
        # Return attention weights if requested
        if return_attention_weights and hasattr(self, '_current_attention_weights'):
            attention_weights = self._current_attention_weights
            delattr(self, '_current_attention_weights')
            return output, attention_weights
        else:
            return output, None
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_index_i: torch.Tensor,
        size_i: Optional[int],
        scale_idx: int,
        temperature: float,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Compute messages between nodes with scale-specific attention.
        
        This method implements the core message passing with attention weights
        that are specific to each biological scale.
        
        Args:
            x_i (Tensor): Features of target nodes
            x_j (Tensor): Features of source nodes
            edge_index_i (Tensor): Target node indices
            size_i (int, optional): Number of target nodes
            scale_idx (int): Index of the current scale
            temperature (float): Temperature parameter for this scale
            return_attention_weights (bool): Whether to store attention weights
            
        Returns:
            Tensor: Computed messages
        """
        # Compute attention scores
        # x_i and x_j have shape [E, heads, head_dim]
        
        # Concatenate source and target features for attention computation
        att_input = torch.cat([x_i, x_j], dim=-1)  # [E, heads, 2 * head_dim]
        
        # Reshape for attention computation
        att_input_flat = att_input.view(-1, 2 * self.head_dim)  # [E * heads, 2 * head_dim]
        
        # Compute attention scores
        att_scores = self.scale_att_weights[scale_idx](att_input_flat)  # [E * heads, 1]
        att_scores = att_scores.view(-1, self.heads)  # [E, heads]
        
        # Apply temperature scaling
        att_scores = att_scores / temperature
        
        # Apply softmax to get attention weights
        att_weights = softmax(att_scores, edge_index_i, num_nodes=size_i)  # [E, heads]
        
        # Store attention weights if requested
        if return_attention_weights:
            self._current_attention_weights = att_weights.detach()
        
        # Apply dropout to attention weights
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)
        
        # Apply attention weights to source features
        # x_j has shape [E, heads, head_dim]
        # att_weights has shape [E, heads]
        att_weights_expanded = att_weights.unsqueeze(-1)  # [E, heads, 1]
        
        # Compute weighted messages
        messages = att_weights_expanded * x_j  # [E, heads, head_dim]
        
        return messages
    
    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Update node features after message aggregation.
        
        Args:
            aggr_out (Tensor): Aggregated messages
            x (Tensor): Original node features
            
        Returns:
            Tensor: Updated node features
        """
        # aggr_out has shape [N, heads, head_dim]
        return aggr_out
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get the last computed attention weights.
        
        Returns:
            Tensor or None: Last computed attention weights
        """
        if hasattr(self, '_current_attention_weights'):
            return self._current_attention_weights
        else:
            return None
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'in_channels={self.in_channels}, '
            f'out_channels={self.out_channels}, '
            f'num_scales={self.num_scales}, '
            f'heads={self.heads}, '
            f'dropout={self.dropout}, '
            f'biological_prior={self.biological_prior}, '
            f'cross_scale_fusion={self.cross_scale_fusion}'
            f')'
        )


class ScaleSpecificAttention(nn.Module):
    """
    Scale-Specific Attention Module
    
    This module implements attention computation for a specific biological scale
    with customizable receptive fields and attention patterns.
    
    Args:
        hidden_dim (int): Hidden dimension size
        heads (int): Number of attention heads
        scale_type (str): Type of scale ('local', 'regional', 'global')
        temperature (float): Temperature parameter for attention softmax
        dropout (float): Dropout rate
        
    Example:
        >>> local_attention = ScaleSpecificAttention(
        ...     hidden_dim=128,
        ...     heads=8,
        ...     scale_type='local',
        ...     temperature=1.0
        ... )
    """
    
    def __init__(
        self,
        hidden_dim: int,
        heads: int = 8,
        scale_type: str = 'local',
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.scale_type = scale_type
        self.temperature = temperature
        self.head_dim = hidden_dim // heads
        
        assert hidden_dim % heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale-specific parameters
        self._setup_scale_parameters()
        
        self.reset_parameters()
    
    def _setup_scale_parameters(self):
        """Setup scale-specific parameters based on scale type."""
        if self.scale_type == 'local':
            # Local scale: sharp attention, small receptive field
            self.attention_kernel_size = 3
            self.attention_dilation = 1
        elif self.scale_type == 'regional':
            # Regional scale: medium attention, moderate receptive field
            self.attention_kernel_size = 7
            self.attention_dilation = 2
        elif self.scale_type == 'global':
            # Global scale: broad attention, large receptive field
            self.attention_kernel_size = 15
            self.attention_dilation = 4
        else:
            raise ValueError(f"Unknown scale_type: {self.scale_type}")
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of scale-specific attention.
        
        Args:
            x (Tensor): Input features [batch_size, seq_len, hidden_dim]
            mask (Tensor, optional): Attention mask
            return_attention (bool): Whether to return attention weights
            
        Returns:
            Tensor or Tuple: Output features, optionally with attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(self.head_dim) * self.temperature)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attended)
        
        if return_attention:
            return output, attention_weights
        else:
            return output


class BiologicalPriorModule(nn.Module):
    """
    Biological Prior Integration Module
    
    This module integrates biological prior knowledge into the attention mechanism
    to enhance interpretability and biological relevance.
    
    Args:
        hidden_dim (int): Hidden dimension size
        num_gene_ontology_terms (int): Number of Gene Ontology terms
        num_pathways (int): Number of biological pathways
        dropout (float): Dropout rate
        
    Example:
        >>> bio_prior = BiologicalPriorModule(
        ...     hidden_dim=128,
        ...     num_gene_ontology_terms=1000,
        ...     num_pathways=500
        ... )
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_gene_ontology_terms: int = 1000,
        num_pathways: int = 500,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_go_terms = num_gene_ontology_terms
        self.num_pathways = num_pathways
        
        # Gene Ontology embeddings
        self.go_embeddings = nn.Embedding(num_gene_ontology_terms, hidden_dim // 2)
        
        # Pathway embeddings
        self.pathway_embeddings = nn.Embedding(num_pathways, hidden_dim // 2)
        
        # Prior fusion network
        self.prior_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Attention weighting for priors
        self.prior_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.go_embeddings.weight)
        nn.init.xavier_uniform_(self.pathway_embeddings.weight)
        
        for layer in self.prior_fusion:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        genomic_features: torch.Tensor,
        go_term_ids: torch.Tensor,
        pathway_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate biological priors with genomic features.
        
        Args:
            genomic_features (Tensor): Genomic feature representations
            go_term_ids (Tensor): Gene Ontology term IDs
            pathway_ids (Tensor): Pathway IDs
            
        Returns:
            Tensor: Enhanced features with biological priors
        """
        # Get biological embeddings
        go_features = self.go_embeddings(go_term_ids)
        pathway_features = self.pathway_embeddings(pathway_ids)
        
        # Combine biological features
        bio_features = torch.cat([go_features, pathway_features], dim=-1)
        
        # Apply attention between genomic and biological features
        enhanced_features, _ = self.prior_attention(
            genomic_features,
            bio_features,
            bio_features
        )
        
        # Compute prior weights
        prior_weights = self.prior_fusion(enhanced_features)
        
        # Apply prior weighting
        output = genomic_features * (1 + prior_weights)
        
        return output

