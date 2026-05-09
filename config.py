"""RIVER Configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "Norman"
    celltype: str = "K562"
    data_dir: Path = Path("rawdata")
    processed_dir: Path = Path("processed_data")
    n_hvg: int = 2000
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    split_seed: int = 42


@dataclass
class Stage1Config:
    """Stage 1: Causal graph learning."""
    n_epochs: int = 300
    lr: float = 3e-3
    hidden_dim: int = 64
    lambda_l1: float = 0.02
    lambda_l1_no_prior: float = 0.2
    prior_mask_mode: str = "adaptive"
    mu_init: float = 1.0
    mu_factor: float = 10.0
    dag_threshold: float = 1e-8
    adj_threshold: float = 0.3
    lambda_pert: float = 0.0


@dataclass
class Stage2Config:
    """Stage 2: Neural SCM."""
    n_epochs: int = 200
    lr: float = 1e-3
    hidden_dim: int = 64
    batch_size: int = 256
    lambda_pert_scm: float = 0.0
    scheduled_sampling_start: float = -1.0

@dataclass
class Stage3Config:
    """Stage 3: Greedy Causal Perturbation Search."""
    diff_threshold: float = 0.5
    convergence_threshold: float = 0.1
    max_interventions: int = 10
    score_mode: str = "raw"


@dataclass
class CPDConfig:
    """Full RIVER configuration."""
    exp_name: str = "cpd_v1"
    seed: int = 42
    device: str = "cuda"
    output_dir: Path = Path("results")
    
    data: DataConfig = field(default_factory=DataConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    
    prior_graph_path: Optional[Path] = None
    
    def get_data_path(self) -> Path:
        """Get processed data path."""
        return self.data.processed_dir / self.data.dataset / f"{self.data.celltype}.h5ad"
    
    def get_prior_graph_path(self) -> Path:
        """Get prior graph path."""
        if self.prior_graph_path is not None:
            return Path(self.prior_graph_path)
        return self.data.processed_dir / self.data.dataset / f"prior_graph_{self.data.celltype}.pt"
    
    def get_output_dir(self) -> Path:
        """Get experiment output directory."""
        return self.output_dir / f"{self.exp_name}_seed{self.seed}"
    
    @classmethod
    def from_yaml(cls, path: Path) -> "CPDConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            d = yaml.safe_load(f)
        
        config = cls()
        
        _path_fields = {"data_dir", "processed_dir"}
        
        if "data" in d:
            for k, v in d["data"].items():
                if hasattr(config.data, k):
                    if k in _path_fields:
                        v = Path(v)
                    setattr(config.data, k, v)
        
        if "stage1" in d:
            for k, v in d["stage1"].items():
                if hasattr(config.stage1, k):
                    setattr(config.stage1, k, v)
        
        if "stage2" in d:
            for k, v in d["stage2"].items():
                if hasattr(config.stage2, k):
                    setattr(config.stage2, k, v)
        
        if "stage3" in d:
            for k, v in d["stage3"].items():
                if hasattr(config.stage3, k):
                    setattr(config.stage3, k, v)
        
        _top_path_fields = {"output_dir", "prior_graph_path"}
        for k in ["exp_name", "seed", "device", "output_dir", "prior_graph_path"]:
            if k in d:
                v = d[k]
                if k in _top_path_fields and v is not None:
                    v = Path(v)
                setattr(config, k, v)
        
        return config
