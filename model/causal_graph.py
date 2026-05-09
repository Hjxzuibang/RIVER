"""Stage 1: Prior-guided Sparse Causal Graph Learning."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, Tuple


class _SafeSlogdet(torch.autograd.Function):
    """slogdet with CPU fallback for backward.

    Workaround for cublasStrsm bug on Blackwell (sm_120) GPUs where
    the backward pass of torch.linalg.slogdet crashes with
    CUBLAS_STATUS_INVALID_VALUE.  Forward runs on the original device;
    the gradient (A^{-T}) is computed on CPU and moved back.
    """

    @staticmethod
    def forward(ctx, A):
        sign, logabsdet = torch.linalg.slogdet(A)
        ctx.save_for_backward(A)
        return logabsdet

    @staticmethod
    def backward(ctx, grad_output):
        A, = ctx.saved_tensors
        A_cpu = A.detach().cpu().to(torch.float64)
        A_inv_T = torch.linalg.inv(A_cpu).T.to(A.dtype)
        return (grad_output.cpu() * A_inv_T).to(A.device)


safe_logabsdet = _SafeSlogdet.apply


class CausalGraphLearner(nn.Module):
    """Prior-guided DAGMA causal graph learner (Stage 1)."""
    
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 64,
        lambda_l1: float = 0.02,
        lambda_l1_no_prior: float = 0.2,
        s: float = 1.0,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.lambda_l1 = lambda_l1
        self.lambda_l1_no_prior = lambda_l1_no_prior
        self.s = s
        
        self.W = nn.Parameter(0.01 * torch.randn(n_genes, n_genes))
        self.register_buffer("prior_mask", torch.ones(n_genes, n_genes))
        self.register_buffer("l1_weights", torch.ones(n_genes, n_genes) * lambda_l1)
        
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_genes, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_genes)
        ])
    
    def set_prior_mask(
        self,
        prior_adj: torch.Tensor,
        mode: str = "adaptive",
    ):
        """Set prior mask from biological prior graph."""
        prior_binary = (prior_adj > 0).float()
        
        prior_binary.fill_diagonal_(0)
        
        if mode == "hard":
            self.prior_mask.copy_(prior_binary)
            self.l1_weights.fill_(self.lambda_l1)
        elif mode == "adaptive":
            self.prior_mask.fill_(1.0)
            self.l1_weights.copy_(
                torch.where(
                    prior_binary > 0,
                    torch.full_like(prior_binary, self.lambda_l1),
                    torch.full_like(prior_binary, self.lambda_l1_no_prior),
                )
            )
        
        n_prior_edges = int(prior_binary.sum().item())
        n_total = self.n_genes * (self.n_genes - 1)
        print(f"  Prior mask: {n_prior_edges}/{n_total} edges "
              f"({100*n_prior_edges/n_total:.1f}%), mode={mode}")
    
    def _get_masked_W(self) -> torch.Tensor:
        """Get masked adjacency matrix."""
        W = self.W * self.prior_mask
        W = W * (1 - torch.eye(self.n_genes, device=W.device))
        return W
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass: X_hat = f(X, W) with gradient checkpointing."""
        W_masked = self._get_masked_W()
        W_abs = torch.abs(W_masked)

        chunk_size = 256
        X_hat_chunks = []
        for start in range(0, self.n_genes, chunk_size):
            end = min(start + chunk_size, self.n_genes)
            chunk = grad_checkpoint(
                self._forward_chunk, X, W_abs, start, end,
                use_reentrant=False,
            )
            X_hat_chunks.append(chunk)

        return torch.cat(X_hat_chunks, dim=1)

    def _forward_chunk(self, X: torch.Tensor, W_abs: torch.Tensor,
                       start: int, end: int) -> torch.Tensor:
        results = []
        for i in range(start, end):
            masked_input = X * W_abs[:, i].unsqueeze(0)
            x_i_hat = self.mlps[i](masked_input)
            results.append(x_i_hat)
        return torch.cat(results, dim=1)
    
    def dag_constraint(self) -> torch.Tensor:
        """DAGMA DAG constraint with adaptive s (Gershgorin bound)."""
        d = self.n_genes
        W_masked = self._get_masked_W()
        W_abs = torch.abs(W_masked)

        with torch.no_grad():
            row_sums = W_abs.sum(dim=1)
            s = max(row_sums.max().item() + 0.5, self.s)

        A = s * torch.eye(d, device=W_abs.device) - W_abs
        logabsdet = safe_logabsdet(A)
        h = -logabsdet + d * np.log(s)
        return h
    
    def l1_penalty(self) -> torch.Tensor:
        """Adaptive L1 sparsity penalty."""
        W_masked = self._get_masked_W()
        return (self.l1_weights * torch.abs(W_masked)).sum() / self.n_genes
    
    def compute_loss(
        self,
        X: torch.Tensor,
        mu: float = 1.0,
        pert_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        lambda_pert: float = 0.1,
    ) -> dict:
        """Compute total loss: reconstruction + DAG constraint + L1 + optional interventional loss."""
        X_hat = self.forward(X)
        
        recon_loss = nn.functional.mse_loss(X_hat, X)
        h = self.dag_constraint()
        l1 = self.l1_penalty()
        
        total = recon_loss + l1 + mu * h
        
        losses = {
            "total": total,
            "recon": recon_loss,
            "dag_constraint": h,
            "l1": l1,
        }
        
        if pert_data is not None:
            ctrl_mean, pert_expr, pert_label = pert_data
            pert_loss = self._compute_interventional_loss(
                ctrl_mean, pert_expr, pert_label
            )
            losses["pert_loss"] = pert_loss
            losses["total"] = losses["total"] + lambda_pert * pert_loss
        
        return losses
    
    def _compute_interventional_loss(
        self,
        ctrl_mean: torch.Tensor,
        pert_expr: torch.Tensor,
        pert_label: torch.Tensor,
    ) -> torch.Tensor:
        """Interventional consistency loss: simulate do(gene_i) and compare with observed."""
        x_do = ctrl_mean.clone()
        x_do = x_do * (1 - pert_label) + pert_expr * pert_label
        x_hat = self.forward(x_do)
        non_target_mask = (1 - pert_label)
        diff = (x_hat - pert_expr) * non_target_mask
        loss = (diff * diff).sum() / non_target_mask.sum().clamp(min=1)
        
        return loss
    
    def get_adjacency(self, threshold: float = 0.3) -> np.ndarray:
        """Get thresholded adjacency matrix."""
        W = self._get_masked_W().detach().cpu().numpy()
        W_abs = np.abs(W)
        nonzero = W_abs[W_abs > 1e-10]
        if len(nonzero) > 0:
            print(f"  [DAG] W abs stats: median={np.median(nonzero):.4f}, "
                  f"max={nonzero.max():.4f}, >0.01={np.sum(nonzero>0.01)}, "
                  f">0.1={np.sum(nonzero>0.1)}, >0.3={np.sum(nonzero>0.3)}")
        W_abs[W_abs < threshold] = 0.0
        np.fill_diagonal(W_abs, 0.0)
        adj = (W_abs > 0).astype(np.float32)
        print(f"  [DAG] threshold={threshold}, edges={int(adj.sum())}")
        return adj

    def get_raw_W(self) -> np.ndarray:
        """Get raw continuous weight matrix."""
        return self._get_masked_W().detach().cpu().numpy()
    
    def get_topological_order(self, adj: Optional[np.ndarray] = None) -> list[int]:
        """Get topological ordering of the DAG."""
        if adj is None:
            adj = self.get_adjacency()
        
        n = adj.shape[0]
        in_degree = adj.sum(axis=0).astype(int)
        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            for j in range(n):
                if adj[node, j] > 0:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)
        
        return order
