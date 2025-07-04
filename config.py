"""
HierGWAS Configuration Management

This module provides configuration management for HierGWAS models,
including parameter validation, serialization, and default settings.

The configuration system supports:
- Model architecture parameters
- Training hyperparameters
- Data processing settings
- Biological prior configurations
- Attention mechanism settings

Example:
    >>> from hiergwas.config import HierGWASConfig
    >>> 
    >>> # Create default configuration
    >>> config = HierGWASConfig()
    >>> 
    >>> # Create custom configuration
    >>> config = HierGWASConfig(
    ...     num_scales=4,
    ...     attention_heads=16,
    ...     hidden_dim=256,
    ...     biological_priors=True
    ... )
    >>> 
    >>> # Save and load configuration
    >>> config.save('config.json')
    >>> loaded_config = HierGWASConfig.load('config.json')
"""

import json
import os
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict, field


@dataclass
class HierGWASConfig:
    """
    Configuration class for HierGWAS models.
    
    This class contains all configurable parameters for the HierGWAS model,
    including architecture settings, training parameters, and biological
    prior configurations.
    
    Architecture Parameters:
        num_scales (int): Number of biological scales (typically 3)
        attention_heads (int): Number of attention heads per scale
        hidden_dim (int): Hidden dimension size for feature representations
        num_layers (int): Number of hierarchical attention layers
        dropout (float): Dropout rate for regularization
        
    Attention Parameters:
        hierarchical_weighting (bool): Whether to use learnable hierarchical weights
        biological_priors (bool): Whether to integrate biological prior knowledge
        cross_scale_fusion (bool): Whether to enable cross-scale information fusion
        scale_temperatures (List[float]): Temperature parameters for each scale
        
    Training Parameters:
        learning_rate (float): Initial learning rate
        weight_decay (float): L2 regularization weight
        batch_size (int): Training batch size
        max_epochs (int): Maximum number of training epochs
        patience (int): Early stopping patience
        
    Data Parameters:
        snp_feature_dim (int): Dimension of SNP features
        gene_feature_dim (int): Dimension of gene features
        pathway_feature_dim (int): Dimension of pathway features
        
    Example:
        >>> # Default configuration
        >>> config = HierGWASConfig()
        >>> 
        >>> # Custom configuration for large-scale study
        >>> config = HierGWASConfig(
        ...     num_scales=4,
        ...     attention_heads=16,
        ...     hidden_dim=512,
        ...     num_layers=6,
        ...     biological_priors=True,
        ...     cross_scale_fusion=True
        ... )
    """
    
    # Architecture parameters
    num_scales: int = 3
    attention_heads: int = 8
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    
    # Attention mechanism parameters
    hierarchical_weighting: bool = True
    biological_priors: bool = False
    cross_scale_fusion: bool = True
    scale_temperatures: Optional[List[float]] = None
    attention_dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler: str = 'cosine_annealing'  # 'cosine_annealing', 'plateau', 'none'
    lr_scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data parameters
    snp_feature_dim: int = 64
    gene_feature_dim: int = 64
    pathway_feature_dim: int = 64
    
    # Biological prior parameters
    num_go_terms: int = 1000
    num_pathways: int = 500
    go_embedding_dim: int = 32
    pathway_embedding_dim: int = 32
    
    # Model regularization
    use_batch_norm: bool = True
    use_layer_norm: bool = True
    use_residual_connections: bool = True
    adaptive_regularization: bool = False
    
    # Evaluation parameters
    eval_every: int = 1
    save_best_model: bool = True
    compute_attention_stats: bool = False
    
    # Hardware and performance
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    num_workers: int = 4
    pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Logging and saving
    verbose: bool = True
    log_every: int = 10
    save_every: int = 50
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        self._validate_parameters()
        self._set_default_scale_temperatures()
        self._set_default_lr_scheduler_params()
    
    def _validate_parameters(self):
        """Validate configuration parameters."""
        # Architecture validation
        if self.num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {self.num_scales}")
        
        if self.attention_heads < 1:
            raise ValueError(f"attention_heads must be >= 1, got {self.attention_heads}")
        
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"attention_heads ({self.attention_heads})"
            )
        
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        
        # Dropout validation
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        
        if not 0 <= self.attention_dropout <= 1:
            raise ValueError(f"attention_dropout must be in [0, 1], got {self.attention_dropout}")
        
        # Training parameter validation
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        
        # Feature dimension validation
        if self.snp_feature_dim < 1:
            raise ValueError(f"snp_feature_dim must be >= 1, got {self.snp_feature_dim}")
        
        if self.gene_feature_dim < 1:
            raise ValueError(f"gene_feature_dim must be >= 1, got {self.gene_feature_dim}")
        
        if self.pathway_feature_dim < 1:
            raise ValueError(f"pathway_feature_dim must be >= 1, got {self.pathway_feature_dim}")
        
        # Learning rate scheduler validation
        valid_schedulers = ['cosine_annealing', 'plateau', 'none']
        if self.lr_scheduler not in valid_schedulers:
            raise ValueError(
                f"lr_scheduler must be one of {valid_schedulers}, got {self.lr_scheduler}"
            )
        
        # Device validation
        valid_devices = ['auto', 'cpu', 'cuda']
        if self.device not in valid_devices and not self.device.startswith('cuda:'):
            raise ValueError(
                f"device must be one of {valid_devices} or 'cuda:X', got {self.device}"
            )
    
    def _set_default_scale_temperatures(self):
        """Set default scale temperatures if not provided."""
        if self.scale_temperatures is None:
            # Default: decreasing temperature for larger scales
            self.scale_temperatures = [1.0 - 0.2 * i for i in range(self.num_scales)]
        
        # Validate scale temperatures
        if len(self.scale_temperatures) != self.num_scales:
            raise ValueError(
                f"Length of scale_temperatures ({len(self.scale_temperatures)}) "
                f"must match num_scales ({self.num_scales})"
            )
        
        for i, temp in enumerate(self.scale_temperatures):
            if temp <= 0:
                raise ValueError(f"scale_temperatures[{i}] must be > 0, got {temp}")
    
    def _set_default_lr_scheduler_params(self):
        """Set default learning rate scheduler parameters."""
        if not self.lr_scheduler_params:
            if self.lr_scheduler == 'cosine_annealing':
                self.lr_scheduler_params = {
                    'T_max': self.max_epochs,
                    'eta_min': self.learning_rate / 100
                }
            elif self.lr_scheduler == 'plateau':
                self.lr_scheduler_params = {
                    'mode': 'max',
                    'factor': 0.5,
                    'patience': 5,
                    'threshold': 1e-4
                }
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
            
        Example:
            >>> config.update(num_scales=4, attention_heads=16)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        # Re-validate after update
        self._validate_parameters()
        self._set_default_scale_temperatures()
        self._set_default_lr_scheduler_params()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict: Configuration as dictionary
            
        Example:
            >>> config_dict = config.to_dict()
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HierGWASConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict (Dict): Configuration dictionary
            
        Returns:
            HierGWASConfig: Configuration object
            
        Example:
            >>> config = HierGWASConfig.from_dict(config_dict)
        """
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath (str): Path to save configuration
            
        Example:
            >>> config.save('hiergwas_config.json')
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'HierGWASConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath (str): Path to configuration file
            
        Returns:
            HierGWASConfig: Loaded configuration
            
        Example:
            >>> config = HierGWASConfig.load('hiergwas_config.json')
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get parameters relevant for model initialization.
        
        Returns:
            Dict: Model parameters
            
        Example:
            >>> model_params = config.get_model_params()
            >>> model = HierGWASModel(**model_params)
        """
        return {
            'num_scales': self.num_scales,
            'attention_heads': self.attention_heads,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'hierarchical_weighting': self.hierarchical_weighting,
            'biological_priors': self.biological_priors,
            'cross_scale_fusion': self.cross_scale_fusion,
            'scale_temperatures': self.scale_temperatures,
            'attention_dropout': self.attention_dropout,
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm,
            'use_residual_connections': self.use_residual_connections
        }
    
    def get_training_params(self) -> Dict[str, Any]:
        """
        Get parameters relevant for training.
        
        Returns:
            Dict: Training parameters
            
        Example:
            >>> training_params = config.get_training_params()
            >>> trainer = HierGWASTrainer(**training_params)
        """
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'warmup_epochs': self.warmup_epochs,
            'gradient_clip': self.gradient_clip,
            'lr_scheduler': self.lr_scheduler,
            'lr_scheduler_params': self.lr_scheduler_params,
            'eval_every': self.eval_every,
            'save_best_model': self.save_best_model
        }
    
    def get_data_params(self) -> Dict[str, Any]:
        """
        Get parameters relevant for data processing.
        
        Returns:
            Dict: Data parameters
            
        Example:
            >>> data_params = config.get_data_params()
            >>> data_loader = GWASDataLoader(**data_params)
        """
        return {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'snp_feature_dim': self.snp_feature_dim,
            'gene_feature_dim': self.gene_feature_dim,
            'pathway_feature_dim': self.pathway_feature_dim
        }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"HierGWASConfig(\n"
            f"  Architecture: {self.num_scales} scales, {self.attention_heads} heads, "
            f"{self.hidden_dim}D, {self.num_layers} layers\n"
            f"  Training: lr={self.learning_rate}, batch_size={self.batch_size}, "
            f"epochs={self.max_epochs}\n"
            f"  Features: biological_priors={self.biological_priors}, "
            f"cross_scale_fusion={self.cross_scale_fusion}\n"
            f")"
        )


# Predefined configurations for common use cases
class HierGWASConfigs:
    """
    Predefined HierGWAS configurations for common use cases.
    
    This class provides several pre-configured setups that work well
    for different types of GWAS studies and computational resources.
    
    Available configurations:
    - small: For small datasets and limited computational resources
    - medium: Balanced configuration for typical GWAS studies
    - large: For large-scale studies with abundant computational resources
    - research: High-capacity configuration for research applications
    
    Example:
        >>> # Use predefined configuration
        >>> config = HierGWASConfigs.medium()
        >>> 
        >>> # Customize predefined configuration
        >>> config = HierGWASConfigs.large()
        >>> config.update(biological_priors=True, num_scales=4)
    """
    
    @staticmethod
    def small() -> HierGWASConfig:
        """
        Small configuration for limited computational resources.
        
        Suitable for:
        - Small datasets (< 10K samples)
        - Limited GPU memory
        - Quick prototyping
        
        Returns:
            HierGWASConfig: Small configuration
        """
        return HierGWASConfig(
            num_scales=2,
            attention_heads=4,
            hidden_dim=64,
            num_layers=2,
            batch_size=256,
            max_epochs=50,
            learning_rate=1e-3
        )
    
    @staticmethod
    def medium() -> HierGWASConfig:
        """
        Medium configuration for typical GWAS studies.
        
        Suitable for:
        - Medium datasets (10K-100K samples)
        - Standard GPU resources
        - Production use
        
        Returns:
            HierGWASConfig: Medium configuration
        """
        return HierGWASConfig(
            num_scales=3,
            attention_heads=8,
            hidden_dim=128,
            num_layers=3,
            batch_size=512,
            max_epochs=100,
            learning_rate=1e-4,
            biological_priors=True,
            cross_scale_fusion=True
        )
    
    @staticmethod
    def large() -> HierGWASConfig:
        """
        Large configuration for large-scale studies.
        
        Suitable for:
        - Large datasets (> 100K samples)
        - High-end GPU resources
        - Research applications
        
        Returns:
            HierGWASConfig: Large configuration
        """
        return HierGWASConfig(
            num_scales=4,
            attention_heads=16,
            hidden_dim=256,
            num_layers=4,
            batch_size=1024,
            max_epochs=200,
            learning_rate=5e-5,
            biological_priors=True,
            cross_scale_fusion=True,
            adaptive_regularization=True
        )
    
    @staticmethod
    def research() -> HierGWASConfig:
        """
        Research configuration for maximum performance.
        
        Suitable for:
        - Research applications
        - Maximum model capacity
        - Extensive computational resources
        
        Returns:
            HierGWASConfig: Research configuration
        """
        return HierGWASConfig(
            num_scales=5,
            attention_heads=32,
            hidden_dim=512,
            num_layers=6,
            batch_size=2048,
            max_epochs=500,
            learning_rate=1e-5,
            biological_priors=True,
            cross_scale_fusion=True,
            adaptive_regularization=True,
            compute_attention_stats=True,
            warmup_epochs=20
        )

