"""CRISP Model: unified three-stage pipeline."""

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch

from .causal_graph import CausalGraphLearner
from .neural_scm import NeuralSCM
from .gcps import GreedyCausalPerturbationSearch


class CPDModel:
    """CRISP full pipeline: DAG learning -> Neural SCM -> Causal Perturbation Search."""
    
    def __init__(
        self,
        gene_names: list[str],
        prior_adj: Optional[np.ndarray] = None,
        device: str = "cuda",
        
        cg_hidden_dim: int = 64,
        cg_lambda_l1: float = 0.02,
        cg_lambda_l1_no_prior: float = 0.2,
        cg_prior_mask_mode: str = "adaptive",
        adj_threshold: float = 0.3,
        
        scm_hidden_dim: int = 64,
        
        diff_threshold: float = 0.5,
        convergence_threshold: float = 0.1,
        max_interventions: int = 10,
        score_mode: str = "raw",
    ):
        self.gene_names = gene_names
        self.n_genes = len(gene_names)
        self.device = device
        self.prior_adj = prior_adj
        
        
        self.cg_hidden_dim = cg_hidden_dim
        self.cg_lambda_l1 = cg_lambda_l1
        self.cg_lambda_l1_no_prior = cg_lambda_l1_no_prior
        self.cg_prior_mask_mode = cg_prior_mask_mode
        self.adj_threshold = adj_threshold
        self.scm_hidden_dim = scm_hidden_dim
        self.diff_threshold = diff_threshold
        self.convergence_threshold = convergence_threshold
        self.max_interventions = max_interventions
        self.score_mode = score_mode
        
        
        self.causal_graph_learner: Optional[CausalGraphLearner] = None
        self.learned_adj: Optional[np.ndarray] = None
        self.raw_W: Optional[np.ndarray] = None
        self.neural_scm: Optional[NeuralSCM] = None
        self.gcps: Optional[GreedyCausalPerturbationSearch] = None
    
    def learn_causal_graph(
        self,
        obs_data: torch.Tensor,
        n_epochs: int = 300,
        lr: float = 3e-3,
        mu_init: float = 0.1,
        mu_factor: float = 2.0,
        dag_threshold: float = 1e-6,
        start_epoch: int = 0,
        optimizer_state: dict = None,
        mu_start: float = None,
        epoch_callback: Optional[Callable] = None,
        pert_data: Optional[torch.Tensor] = None,
        pert_labels: Optional[torch.Tensor] = None,
        ctrl_mean: Optional[torch.Tensor] = None,
        lambda_pert: float = 0.1,
    ) -> np.ndarray:
        """Stage 1: Learn causal DAG from observational data."""
        if self.causal_graph_learner is None:
            self.causal_graph_learner = CausalGraphLearner(
                n_genes=self.n_genes,
                hidden_dim=self.cg_hidden_dim,
                lambda_l1=self.cg_lambda_l1,
                lambda_l1_no_prior=self.cg_lambda_l1_no_prior,
            ).to(self.device)
        
        if self.prior_adj is not None:
            prior_tensor = torch.from_numpy(self.prior_adj).float().to(self.device)
            self.causal_graph_learner.set_prior_mask(
                prior_tensor, mode=self.cg_prior_mask_mode
            )
        
        optimizer = torch.optim.Adam(
            self.causal_graph_learner.parameters(), lr=lr
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        
        obs_data = obs_data.cpu()
        if pert_data is not None:
            pert_data = pert_data.cpu()
            pert_labels = pert_labels.cpu()
            ctrl_mean_cpu = ctrl_mean.cpu() if ctrl_mean is not None else None
        mu = mu_start if mu_start is not None else mu_init
        
        for epoch in range(start_epoch, n_epochs):
            
            perm = torch.randperm(obs_data.shape[0])
            batch = obs_data[perm[:512]].to(self.device)
            
            
            pert_batch = None
            if pert_data is not None and pert_labels is not None:
                p_perm = torch.randperm(pert_data.shape[0])
                p_bs = min(64, pert_data.shape[0])
                p_expr = pert_data[p_perm[:p_bs]].to(self.device)
                p_label = pert_labels[p_perm[:p_bs]].to(self.device)
                p_ctrl = ctrl_mean_cpu[p_perm[:p_bs]].to(self.device) if ctrl_mean_cpu is not None else None
                if p_ctrl is None:
                    p_ctrl = torch.zeros_like(p_expr)
                pert_batch = (p_ctrl, p_expr, p_label)
            
            losses = self.causal_graph_learner.compute_loss(
                batch, mu=mu, pert_data=pert_batch, lambda_pert=lambda_pert
            )
            
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            
            
            h = losses["dag_constraint"].item()
            if h > dag_threshold and epoch > 0 and epoch % 100 == 0:
                mu = min(mu * mu_factor, 100.0)
            
            if epoch % 100 == 0:
                pert_str = ""
                if "pert_loss" in losses:
                    pert_str = f" pert={losses['pert_loss'].item():.4f}"
                print(f"[Stage1] Epoch {epoch}: "
                      f"recon={losses['recon'].item():.4f} "
                      f"h(W)={h:.6f} mu={mu:.1f}{pert_str}")
            
            if epoch_callback is not None:
                record = {
                    "stage": 1,
                    "epoch": epoch,
                    "loss": losses["total"].item(),
                    "recon_loss": losses["recon"].item(),
                    "dag_h": h,
                    "mu": mu,
                }
                epoch_callback(epoch, record, optimizer, mu)
        
        
        self.raw_W = self.causal_graph_learner.get_raw_W()
        self.learned_adj = self.causal_graph_learner.get_adjacency(
            threshold=self.adj_threshold)
        print(f"[Stage1] Learned DAG: {self.learned_adj.sum():.0f} edges "
              f"(threshold={self.adj_threshold})")
        
        
        self.causal_graph_learner.cpu()
        del self.causal_graph_learner
        self.causal_graph_learner = None
        torch.cuda.empty_cache()
        
        return self.learned_adj
    
    def train_neural_scm(
        self,
        obs_data: torch.Tensor,
        n_epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 256,
        start_epoch: int = 0,
        optimizer_state: dict = None,
        epoch_callback: Optional[Callable] = None,
        pert_data: Optional[torch.Tensor] = None,
        pert_labels: Optional[torch.Tensor] = None,
        ctrl_mean: Optional[torch.Tensor] = None,
        lambda_pert_scm: float = 0.0,
        scheduled_sampling_start: float = -1.0,
    ):
        """Stage 2: Train Neural SCM on observational and perturbation data."""
        if self.learned_adj is None:
            raise RuntimeError("Must run learn_causal_graph() first")
        
        if self.neural_scm is None:
            self.neural_scm = NeuralSCM(
                adjacency=self.learned_adj,
                gene_names=self.gene_names,
                hidden_dim=self.scm_hidden_dim,
            ).to(self.device)
        
        optimizer = torch.optim.Adam(self.neural_scm.parameters(), lr=lr)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        obs_data = obs_data.cpu()
        
        pert_gpu = None
        if lambda_pert_scm > 0 and pert_data is not None:
            pert_gpu = (
                ctrl_mean.to(self.device),
                pert_data.to(self.device),
                pert_labels.to(self.device),
            )
            
            n_nonleaf = sum(
                1 for b in range(pert_data.shape[0])
                if any(len(self.neural_scm._descendants[t.item()]) > 0
                       for t in (pert_labels[b] > 0.5).nonzero(as_tuple=True)[0])
            )
            print(f"[Stage2] Perturbation data pre-loaded to GPU "
                  f"({pert_data.shape[0]} total, {n_nonleaf} non-leaf)")
        
        for epoch in range(start_epoch, n_epochs):
            perm = torch.randperm(obs_data.shape[0])
            
            epoch_loss = 0.0
            epoch_pert_loss = 0.0
            n_batches = 0
            
            for i in range(0, obs_data.shape[0], batch_size):
                batch = obs_data[perm[i:i+batch_size]].to(self.device)
                
                sp = 0.0
                if scheduled_sampling_start >= 0:
                    progress = epoch / max(n_epochs - 1, 1)
                    if progress >= scheduled_sampling_start:
                        ramp = (progress - scheduled_sampling_start) / max(1.0 - scheduled_sampling_start, 1e-6)
                        sp = min(ramp * 0.5, 0.5)
                losses = self.neural_scm.compute_loss(batch, sampling_prob=sp)
                
                total = losses["total"]
                
                
                if pert_gpu is not None:
                    ctrl_all, pert_all, label_all = pert_gpu
                    n_pert = ctrl_all.shape[0]
                    pert_bs = min(batch_size // 4, n_pert)
                    pert_idx = torch.randperm(n_pert, device=self.device)[:pert_bs]
                    
                    interv_loss = self.neural_scm.compute_interventional_loss(
                        ctrl_all[pert_idx], pert_all[pert_idx], label_all[pert_idx]
                    )
                    total = total + lambda_pert_scm * interv_loss
                    epoch_pert_loss += interv_loss.item()
                
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
                
                epoch_loss += losses["total"].item()
                n_batches += 1
            
            if epoch % 10 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                sp_display = 0.0
                if scheduled_sampling_start >= 0:
                    progress = epoch / max(n_epochs - 1, 1)
                    if progress >= scheduled_sampling_start:
                        ramp = (progress - scheduled_sampling_start) / max(1.0 - scheduled_sampling_start, 1e-6)
                        sp_display = min(ramp * 0.5, 0.5)
                msg = f"[Stage2] Epoch {epoch}: recon_loss={avg_loss:.4f}"
                if lambda_pert_scm > 0:
                    avg_pert = epoch_pert_loss / max(n_batches, 1)
                    msg += f" pert_loss={avg_pert:.4f}"
                if sp_display > 0:
                    msg += f" sp={sp_display:.2f}"
                print(msg)
            
            if epoch_callback is not None:
                record = {
                    "stage": 2,
                    "epoch": epoch,
                    "loss": epoch_loss / max(n_batches, 1),
                }
                epoch_callback(epoch, record, optimizer, None)
        
        
        self.gcps = GreedyCausalPerturbationSearch(
            scm=self.neural_scm,
            adjacency=self.learned_adj,
            diff_threshold=self.diff_threshold,
            convergence_threshold=self.convergence_threshold,
            max_interventions=self.max_interventions,
            score_mode=self.score_mode,
        )
        
        print("[Stage2] Neural SCM trained. GCPS ready.")
    
    @torch.no_grad()
    def predict(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        weight_mode: str = "uniform",
    ) -> dict:
        """Stage 3: Predict intervention set via causal perturbation search."""
        if self.gcps is None:
            raise RuntimeError("Must run train_neural_scm() first")
        
        x_source = x_source.to(self.device)
        x_target = x_target.to(self.device)
        
        result = self.gcps.search(x_source, x_target, weight_mode)
        
        
        result["intervention_genes"] = [
            self.gene_names[i] for i in result["intervention_set"]
        ]
        
        return result
    
    def save(self, save_dir: Path):
        """Save model checkpoint."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": {
                "gene_names": self.gene_names,
                "n_genes": self.n_genes,
                "cg_hidden_dim": self.cg_hidden_dim,
                "scm_hidden_dim": self.scm_hidden_dim,
                "diff_threshold": self.diff_threshold,
                "convergence_threshold": self.convergence_threshold,
                "max_interventions": self.max_interventions,
                "adj_threshold": self.adj_threshold,
                "score_mode": self.score_mode,
            },
        }
        if self.learned_adj is not None:
            state["learned_adj"] = self.learned_adj
        if hasattr(self, 'raw_W') and self.raw_W is not None:
            state["raw_W"] = self.raw_W
        if self.causal_graph_learner is not None:
            state["cg_state"] = self.causal_graph_learner.state_dict()
        if self.neural_scm is not None:
            state["scm_state"] = self.neural_scm.state_dict()
        
        torch.save(state, save_dir / "best_model.pt")
        print(f"Model saved to {save_dir / 'best_model.pt'}")
    
    @classmethod
    def load(cls, save_dir: Path, device: str = "cuda") -> "CPDModel":
        """Load saved model."""
        save_dir = Path(save_dir)
        
        best_path = save_dir / "best_model.pt"
        ckpt_path = save_dir / "checkpoints" / "best_model.pt"
        legacy_path = save_dir / "config.pt"
        
        if best_path.exists():
            return cls._load_new_format(best_path, device)
        elif ckpt_path.exists():
            return cls._load_new_format(ckpt_path, device)
        elif legacy_path.exists():
            return cls._load_legacy_format(save_dir, device)
        else:
            raise FileNotFoundError(f"No model found in {save_dir}")
    
    @classmethod
    def _load_new_format(cls, path: Path, device: str) -> "CPDModel":
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = state["config"]
        
        model = cls(
            gene_names=config["gene_names"],
            device=device,
            cg_hidden_dim=config["cg_hidden_dim"],
            scm_hidden_dim=config["scm_hidden_dim"],
            diff_threshold=config["diff_threshold"],
            convergence_threshold=config["convergence_threshold"],
            max_interventions=config["max_interventions"],
            adj_threshold=config.get("adj_threshold", 0.3),
            score_mode=config.get("score_mode", "raw"),
        )
        
        if "learned_adj" in state:
            model.learned_adj = state["learned_adj"]
        if "raw_W" in state:
            model.raw_W = state["raw_W"]
        
        if "scm_state" in state and model.learned_adj is not None:
            model.neural_scm = NeuralSCM(
                adjacency=model.learned_adj,
                gene_names=model.gene_names,
                hidden_dim=config["scm_hidden_dim"],
            ).to(device)
            model.neural_scm.load_state_dict(state["scm_state"])
            
            model.gcps = GreedyCausalPerturbationSearch(
                scm=model.neural_scm,
                adjacency=model.learned_adj,
                diff_threshold=config["diff_threshold"],
                convergence_threshold=config["convergence_threshold"],
                max_interventions=config["max_interventions"],
                score_mode=config.get("score_mode", "raw"),
            )
        
        return model
    
    @classmethod
    def _load_legacy_format(cls, save_dir: Path, device: str) -> "CPDModel":
        config = torch.load(
            save_dir / "config.pt", map_location="cpu", weights_only=False
        )
        
        model = cls(
            gene_names=config["gene_names"],
            device=device,
            cg_hidden_dim=config["cg_hidden_dim"],
            scm_hidden_dim=config["scm_hidden_dim"],
            diff_threshold=config["diff_threshold"],
            convergence_threshold=config["convergence_threshold"],
            max_interventions=config["max_interventions"],
        )
        
        adj_path = save_dir / "learned_adj.npy"
        if adj_path.exists():
            model.learned_adj = np.load(adj_path)
        
        scm_path = save_dir / "neural_scm.pt"
        if scm_path.exists() and model.learned_adj is not None:
            model.neural_scm = NeuralSCM(
                adjacency=model.learned_adj,
                gene_names=model.gene_names,
                hidden_dim=config["scm_hidden_dim"],
            ).to(device)
            model.neural_scm.load_state_dict(
                torch.load(scm_path, map_location=device, weights_only=True)
            )
            model.gcps = GreedyCausalPerturbationSearch(
                scm=model.neural_scm,
                adjacency=model.learned_adj,
                diff_threshold=config["diff_threshold"],
                convergence_threshold=config["convergence_threshold"],
                max_interventions=config["max_interventions"],
                score_mode=config.get("score_mode", "raw"),
            )
        
        return model
