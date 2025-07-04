"""
HierGWAS Model Architecture

This module implements the main HierGWAS model that combines hierarchical
multi-scale attention with heterogeneous graph neural networks for
genome-wide association studies.

The model architecture consists of:
1. Feature preprocessing for different node types (SNPs, genes, pathways)
2. Hierarchical multi-scale attention layers
3. Cross-modal fusion for heterogeneous data
4. Output prediction layers

Key Components:
- HierGWASModel: Main model class
- HeteroGNNEncoder: Heterogeneous graph encoder
- FeaturePreprocessor: Input feature processing
- OutputPredictor: Final prediction layers

Example:
    >>> from hiergwas.model import HierGWASModel
    >>> from hiergwas.config import HierGWASConfig
    >>> 
    >>> config = HierGWASConfig(num_scales=3, attention_heads=8)
    >>> model = HierGWASModel(data=gwas_data, config=config)
    >>> 
    >>> # Forward pass
    >>> predictions = model(x_dict, edge_index_dict, batch_size=32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, BatchNorm
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple, Union, Any

from .attention import HierarchicalMultiScaleAttention, BiologicalPriorModule
from .config import HierGWASConfig
from .data import GWASData


class HierGWASModel(nn.Module):
    """
    HierGWAS: Hierarchical Multi-Scale Attention Model for GWAS
    
    This is the main model class that implements the complete HierGWAS architecture.
    It combines hierarchical multi-scale attention with heterogeneous graph neural
    networks to capture genomic interactions at multiple biological scales.
    
    The model processes three types of nodes:
    - SNP nodes: Individual genetic variants
    - Gene nodes: Protein-coding genes
    - Pathway nodes: Biological pathways and Gene Ontology terms
    
    And three types of edges:
    - SNP-Gene: Variant-to-gene mappings
    - Gene-Pathway: Gene-to-pathway memberships
    - Gene-Gene: Protein-protein interactions
    
    Args:
        data (GWASData): Genomic data object
        config (HierGWASConfig): Model configuration
        
    Example:
        >>> # Basic model creation
        >>> model = HierGWASModel(data=gwas_data, config=config)
        >>> 
        >>> # Forward pass
        >>> predictions = model(x_dict, edge_index_dict, batch_size=32)
        >>> 
        >>> # Forward pass with attention analysis
        >>> predictions, attention = model(
        ...     x_dict, edge_index_dict, batch_size=32,
        ...     return_attention_weights=True
        ... )
    """
    
    def __init__(
        self,
        data: GWASData,
        config: HierGWASConfig
    ):
        super().__init__()
        
        self.data = data
        self.config = config
        
        # Extract configuration parameters
        self.num_scales = config.num_scales
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.attention_heads = config.attention_heads
        self.dropout = config.dropout
        self.biological_priors = config.biological_priors
        self.cross_scale_fusion = config.cross_scale_fusion
        
        # Node type information
        self.node_types = ['SNP', 'Gene', 'Pathway']
        self.edge_types = [
            ('SNP', 'maps_to', 'Gene'),
            ('Gene', 'belongs_to', 'Pathway'),
            ('Gene', 'interacts_with', 'Gene')
        ]
        
        # Feature dimensions for each node type
        self.feature_dims = {
            'SNP': data.snp_feature_dim,
            'Gene': data.gene_feature_dim,
            'Pathway': data.pathway_feature_dim
        }
        
        # Build model components
        self._build_feature_preprocessor()
        self._build_hierarchical_encoder()
        self._build_output_predictor()
        
        # Initialize biological prior module if enabled
        if self.biological_priors:
            self._build_biological_prior_module()
        
        # Initialize parameters
        self.reset_parameters()
    
    def _build_feature_preprocessor(self):
        """Build feature preprocessing layers for each node type."""
        self.feature_preprocessors = nn.ModuleDict()
        
        for node_type in self.node_types:
            input_dim = self.feature_dims[node_type]
            
            # Multi-layer feature preprocessor with residual connections
            preprocessor = FeaturePreprocessor(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                num_layers=2,
                dropout=self.dropout,
                use_batch_norm=True,
                use_residual=True
            )
            
            self.feature_preprocessors[node_type] = preprocessor
    
    def _build_hierarchical_encoder(self):
        """Build hierarchical multi-scale attention encoder."""
        self.hierarchical_layers = nn.ModuleList()
        
        for layer_idx in range(self.num_layers):
            # Create heterogeneous convolution layer with HMSA
            hetero_conv = HeteroConv({
                edge_type: HierarchicalMultiScaleAttention(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    num_scales=self.num_scales,
                    heads=self.attention_heads,
                    dropout=self.dropout,
                    biological_prior=self.biological_priors,
                    cross_scale_fusion=self.cross_scale_fusion
                )
                for edge_type in self.edge_types
            }, aggr='sum')
            
            self.hierarchical_layers.append(hetero_conv)
        
        # Layer normalization for each node type
        self.layer_norms = nn.ModuleDict({
            node_type: nn.LayerNorm(self.hidden_dim)
            for node_type in self.node_types
        })
        
        # Dropout layers
        self.dropout_layers = nn.ModuleDict({
            node_type: nn.Dropout(self.dropout)
            for node_type in self.node_types
        })
    
    def _build_output_predictor(self):
        """Build output prediction layers."""
        # We predict on SNP nodes for GWAS
        self.output_predictor = OutputPredictor(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            output_dim=1,  # Binary classification
            num_layers=3,
            dropout=self.dropout,
            use_batch_norm=True
        )
        
        # Global pooling for graph-level predictions if needed
        self.global_pooling = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def _build_biological_prior_module(self):
        """Build biological prior integration module."""
        self.biological_prior_module = BiologicalPriorModule(
            hidden_dim=self.hidden_dim,
            num_gene_ontology_terms=self.data.num_go_terms,
            num_pathways=self.data.num_pathways,
            dropout=self.dropout
        )
    
    def reset_parameters(self):
        """Initialize model parameters."""
        # Initialize feature preprocessors
        for preprocessor in self.feature_preprocessors.values():
            preprocessor.reset_parameters()
        
        # Initialize hierarchical layers
        for layer in self.hierarchical_layers:
            for conv in layer.convs.values():
                if hasattr(conv, 'reset_parameters'):
                    conv.reset_parameters()
        
        # Initialize output predictor
        self.output_predictor.reset_parameters()
        
        # Initialize global pooling
        for layer in self.global_pooling:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Initialize biological prior module
        if self.biological_priors:
            self.biological_prior_module.reset_parameters()
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        batch_size: Optional[int] = None,
        return_attention_weights: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass of the HierGWAS model.
        
        Args:
            x_dict (Dict): Node features for each node type
            edge_index_dict (Dict): Edge indices for each edge type
            batch_size (int, optional): Batch size for node-level predictions
            return_attention_weights (bool): Whether to return attention weights
            
        Returns:
            Tensor or Tuple: Predictions, optionally with attention weights
            
        Example:
            >>> # Basic prediction
            >>> predictions = model(x_dict, edge_index_dict, batch_size=32)
            >>> 
            >>> # Prediction with attention analysis
            >>> predictions, attention = model(
            ...     x_dict, edge_index_dict, batch_size=32,
            ...     return_attention_weights=True
            ... )
        """
        # Store attention weights if requested
        all_attention_weights = {
            'layer_attention': [],
            'scale_attention': [],
            'hierarchical_weights': []
        } if return_attention_weights else None
        
        # Feature preprocessing
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.feature_preprocessors[node_type](x)
        
        # Store original features for residual connections
        residual_dict = {node_type: h.clone() for node_type, h in h_dict.items()}
        
        # Hierarchical multi-scale attention layers
        for layer_idx, hetero_conv in enumerate(self.hierarchical_layers):
            # Apply hierarchical attention
            if return_attention_weights:
                h_dict_new = {}
                layer_attention = {}
                
                for edge_type in self.edge_types:
                    src_type, _, dst_type = edge_type
                    edge_index = edge_index_dict[edge_type]
                    
                    # Get source and destination features
                    x_src = h_dict[src_type]
                    x_dst = h_dict[dst_type]
                    
                    # Apply HMSA with attention weight extraction
                    conv_layer = hetero_conv.convs[edge_type]
                    output, attention = conv_layer(
                        (x_src, x_dst),
                        edge_index,
                        return_attention_weights=True
                    )
                    
                    # Store output and attention
                    if dst_type not in h_dict_new:
                        h_dict_new[dst_type] = output
                    else:
                        h_dict_new[dst_type] += output
                    
                    layer_attention[edge_type] = attention
                
                # Store layer attention weights
                all_attention_weights['layer_attention'].append(layer_attention)
                h_dict = h_dict_new
            else:
                h_dict = hetero_conv(h_dict, edge_index_dict)
            
            # Apply layer normalization and dropout
            for node_type in self.node_types:
                if node_type in h_dict:
                    # Residual connection
                    if h_dict[node_type].shape == residual_dict[node_type].shape:
                        h_dict[node_type] = h_dict[node_type] + residual_dict[node_type]
                    
                    # Layer normalization
                    h_dict[node_type] = self.layer_norms[node_type](h_dict[node_type])
                    
                    # Dropout
                    h_dict[node_type] = self.dropout_layers[node_type](h_dict[node_type])
                    
                    # Update residual for next layer
                    residual_dict[node_type] = h_dict[node_type].clone()
        
        # Biological prior integration if enabled
        if self.biological_priors and hasattr(self, 'biological_prior_module'):
            # This would require additional data about GO terms and pathways
            # For now, we'll skip this step in the basic implementation
            pass
        
        # Output prediction on SNP nodes
        snp_features = h_dict['SNP']
        
        if batch_size is not None:
            # Node-level predictions for the first batch_size nodes
            snp_features = snp_features[:batch_size]
        
        # Apply output predictor
        predictions = self.output_predictor(snp_features)
        
        if return_attention_weights:
            return predictions, all_attention_weights
        else:
            return predictions
    
    def get_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        node_type: str = 'SNP'
    ) -> torch.Tensor:
        """
        Get learned embeddings for a specific node type.
        
        Args:
            x_dict (Dict): Node features for each node type
            edge_index_dict (Dict): Edge indices for each edge type
            node_type (str): Type of nodes to get embeddings for
            
        Returns:
            Tensor: Learned embeddings for the specified node type
            
        Example:
            >>> snp_embeddings = model.get_embeddings(x_dict, edge_index_dict, 'SNP')
            >>> gene_embeddings = model.get_embeddings(x_dict, edge_index_dict, 'Gene')
        """
        with torch.no_grad():
            # Forward pass without predictions
            h_dict = {}
            for nt, x in x_dict.items():
                h_dict[nt] = self.feature_preprocessors[nt](x)
            
            # Apply hierarchical layers
            for hetero_conv in self.hierarchical_layers:
                h_dict = hetero_conv(h_dict, edge_index_dict)
                
                # Apply normalization
                for nt in self.node_types:
                    if nt in h_dict:
                        h_dict[nt] = self.layer_norms[nt](h_dict[nt])
            
            return h_dict[node_type]
    
    def compute_attention_statistics(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        num_samples: int = 100
    ) -> Dict:
        """
        Compute statistics about learned attention patterns.
        
        Args:
            x_dict (Dict): Node features for each node type
            edge_index_dict (Dict): Edge indices for each edge type
            num_samples (int): Number of samples to analyze
            
        Returns:
            Dict: Attention statistics and patterns
            
        Example:
            >>> stats = model.compute_attention_statistics(x_dict, edge_index_dict)
            >>> print(f"Average hierarchical weights: {stats['avg_hierarchical_weights']}")
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass with attention weights
            _, attention_weights = self.forward(
                x_dict, edge_index_dict,
                batch_size=num_samples,
                return_attention_weights=True
            )
            
            # Compute statistics
            stats = {
                'num_layers': len(attention_weights['layer_attention']),
                'attention_patterns': {},
                'hierarchical_weight_stats': {},
                'scale_preference_stats': {}
            }
            
            # Analyze attention patterns for each layer
            for layer_idx, layer_attn in enumerate(attention_weights['layer_attention']):
                layer_stats = {}
                
                for edge_type, attn_data in layer_attn.items():
                    if 'hierarchical_weights' in attn_data:
                        hier_weights = attn_data['hierarchical_weights']
                        layer_stats[edge_type] = {
                            'mean_weights': hier_weights.mean(dim=0).cpu().numpy(),
                            'std_weights': hier_weights.std(dim=0).cpu().numpy(),
                            'entropy': self._compute_attention_entropy(hier_weights)
                        }
                
                stats['attention_patterns'][f'layer_{layer_idx}'] = layer_stats
            
            return stats
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights."""
        # Normalize weights
        normalized_weights = F.softmax(attention_weights, dim=-1)
        
        # Compute entropy
        entropy = -(normalized_weights * torch.log(normalized_weights + 1e-8)).sum(dim=-1)
        
        return entropy.mean().item()


class FeaturePreprocessor(nn.Module):
    """
    Feature Preprocessing Module
    
    This module preprocesses input features for different node types with
    multi-layer transformations, batch normalization, and residual connections.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension
        output_dim (int): Output feature dimension
        num_layers (int): Number of preprocessing layers
        dropout (float): Dropout rate
        use_batch_norm (bool): Whether to use batch normalization
        use_residual (bool): Whether to use residual connections
        
    Example:
        >>> preprocessor = FeaturePreprocessor(
        ...     input_dim=100,
        ...     hidden_dim=128,
        ...     output_dim=128,
        ...     num_layers=2
        ... )
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Build layers
        layers = []
        layer_dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            
            # Linear layer
            layers.append(nn.Linear(in_dim, out_dim))
            
            # Batch normalization (except for last layer)
            if use_batch_norm and i < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_dim))
            
            # Activation (except for last layer)
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
        
        # Residual projection if needed
        if use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
    
    def reset_parameters(self):
        """Initialize parameters."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        if self.residual_proj is not None:
            nn.init.xavier_uniform_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feature preprocessor.
        
        Args:
            x (Tensor): Input features
            
        Returns:
            Tensor: Preprocessed features
        """
        # Apply layers
        out = self.layers(x)
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            
            if out.shape == residual.shape:
                out = out + residual
        
        return out


class OutputPredictor(nn.Module):
    """
    Output Prediction Module
    
    This module generates final predictions from learned node representations
    with multi-layer transformations and regularization.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension
        output_dim (int): Output dimension (1 for binary classification)
        num_layers (int): Number of prediction layers
        dropout (float): Dropout rate
        use_batch_norm (bool): Whether to use batch normalization
        
    Example:
        >>> predictor = OutputPredictor(
        ...     input_dim=128,
        ...     hidden_dim=64,
        ...     output_dim=1,
        ...     num_layers=3
        ... )
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Build layers
        layers = []
        layer_dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            
            # Linear layer
            layers.append(nn.Linear(in_dim, out_dim))
            
            # Batch normalization and activation (except for last layer)
            if i < num_layers - 1:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        
        self.layers = nn.Sequential(*layers)
    
    def reset_parameters(self):
        """Initialize parameters."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of output predictor.
        
        Args:
            x (Tensor): Input features
            
        Returns:
            Tensor: Predictions (logits)
        """
        return self.layers(x)

