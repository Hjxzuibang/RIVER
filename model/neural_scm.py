"""Stage 2: Neural Structural Causal Model."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GeneSCM(nn.Module):
    """Per-gene structural equation: X_i = f(X_pa(i))."""
    
    def __init__(self, n_parents: int, hidden_dim: int = 64):
        super().__init__()
        if n_parents == 0:
            self.is_root = True
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.is_root = False
            self.net = nn.Sequential(
                nn.Linear(n_parents, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
    
    def forward(self, parent_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.is_root:
            batch = parent_values.shape[0] if parent_values is not None else 1
            return self.bias.expand(batch, 1)
        return self.net(parent_values)


class NeuralSCM(nn.Module):
    """Neural Structural Causal Model (Stage 2)."""
    
    def __init__(
        self,
        adjacency: np.ndarray,
        gene_names: list[str],
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_genes = len(gene_names)
        self.gene_names = gene_names
        self.adjacency = adjacency
        self.topo_order = self._topological_sort()
        
        self.parent_indices = {}
        for j in range(self.n_genes):
            parents = np.where(adjacency[:, j] > 0)[0].tolist()
            self.parent_indices[j] = parents
        
        
        self._children = [[] for _ in range(self.n_genes)]
        for j in range(self.n_genes):
            for p in self.parent_indices[j]:
                self._children[p].append(j)
        self._descendants = self._compute_descendants()
        self._topo_pos = {g: i for i, g in enumerate(self.topo_order)}
        
        self.gene_scms = nn.ModuleList()
        for j in range(self.n_genes):
            n_parents = len(self.parent_indices[j])
            self.gene_scms.append(GeneSCM(n_parents, hidden_dim))
        
        
        self._root_genes = [j for j in self.topo_order if len(self.parent_indices[j]) == 0]
        self._nonroot_topo = [j for j in self.topo_order if len(self.parent_indices[j]) > 0]
        
        self._build_batch_groups()
    
    def _build_batch_groups(self):
        """Group non-root genes by parent count for batched forward."""
        from collections import defaultdict
        groups = defaultdict(list)  # parent_count -> [gene_indices]
        for j in self._nonroot_topo:
            k = len(self.parent_indices[j])
            groups[k].append(j)
        
        self._batch_groups: list[tuple] = []
        for k in sorted(groups.keys()):
            gene_list = groups[k]
            parent_idx = [self.parent_indices[j] for j in gene_list]
            parent_idx_t = torch.tensor(parent_idx, dtype=torch.long)
            w1_refs = [self.gene_scms[j].net[0].weight for j in gene_list]
            b1_refs = [self.gene_scms[j].net[0].bias for j in gene_list]
            w2_refs = [self.gene_scms[j].net[2].weight for j in gene_list]
            b2_refs = [self.gene_scms[j].net[2].bias for j in gene_list]
            self._batch_groups.append(
                (gene_list, parent_idx_t, w1_refs, b1_refs, w2_refs, b2_refs)
            )

    def _compute_descendants(self) -> list[set[int]]:
        """Precompute descendant sets via BFS."""
        from collections import deque
        descendants = []
        for g in range(self.n_genes):
            visited = set()
            q = deque(self._children[g])
            while q:
                c = q.popleft()
                if c not in visited:
                    visited.add(c)
                    for cc in self._children[c]:
                        if cc not in visited:
                            q.append(cc)
            descendants.append(visited)
        return descendants
    
    def _topological_sort(self) -> list[int]:
        """Topological sort via Kahn's algorithm."""
        adj = self.adjacency.copy()
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
        
        
        if len(order) < n:
            remaining = [i for i in range(n) if i not in order]
            order.extend(remaining)
        
        return order
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass: reconstruct expression from observed data."""
        batch = X.shape[0]
        X_hat = torch.zeros_like(X)
        
        # Process root nodes (87.6%): each returns bias, no computation needed
        if self._root_genes:
            root_biases = torch.cat(
                [self.gene_scms[j].bias for j in self._root_genes]
            )  # [n_roots]
            X_hat[:, self._root_genes] = root_biases.unsqueeze(0).expand(batch, -1)
        
        
        for gene_list, parent_idx_t, w1_refs, b1_refs, w2_refs, b2_refs in self._batch_groups:
            pidx = parent_idx_t.to(X.device)
            parent_vals = X[:, pidx]
            
            for i, j in enumerate(gene_list):
                h = F.linear(parent_vals[:, i, :], w1_refs[i], b1_refs[i])
                h = torch.relu(h)
                out = F.linear(h, w2_refs[i], b2_refs[i])
                X_hat[:, j] = out.squeeze(-1)
        
        return X_hat
    
    def cascade_forward(
        self, X: torch.Tensor, sampling_prob: float = 0.0
    ) -> torch.Tensor:
        """Cascade forward with scheduled sampling."""
        if sampling_prob <= 0.0:
            return self.forward(X)
        
        batch = X.shape[0]
        X_hat = torch.zeros_like(X)
        
        
        if self._root_genes:
            root_biases = torch.cat(
                [self.gene_scms[j].bias for j in self._root_genes]
            )
            X_hat[:, self._root_genes] = root_biases.unsqueeze(0).expand(batch, -1)
        
        
        use_pred = torch.rand(self.n_genes, device=X.device) < sampling_prob
        
        for j in self._root_genes:
            use_pred[j] = True
        
        
        X_mix = X.clone()
        for j in self._root_genes:
            X_mix[:, j] = X_hat[:, j]
        
        
        for j in self._nonroot_topo:
            parents = self.parent_indices[j]
            parent_vals = X_mix[:, parents]  # [batch, k]
            scm_j = self.gene_scms[j]
            pred_j = scm_j(parent_vals).squeeze(-1)  # [batch]
            X_hat[:, j] = pred_j
            if use_pred[j]:
                X_mix[:, j] = pred_j
        
        return X_hat

    def do_intervention(
        self,
        X_source: torch.Tensor,
        intervention_indices: list[int],
        intervention_values: torch.Tensor,
        n_iterations: int = 1,
    ) -> torch.Tensor:
        """Simulate do-intervention: fix intervened genes and propagate downstream."""
        intervention_set = set(intervention_indices)
        X_do = X_source.clone()
        
        
        for k, idx in enumerate(intervention_indices):
            X_do[:, idx] = intervention_values[:, k]
        
        
        desc_set = set()
        for idx in intervention_indices:
            desc_set.update(self._descendants[idx])
        desc_set -= intervention_set
        desc_topo = sorted(desc_set, key=lambda g: self._topo_pos[g])
        
        for _ in range(n_iterations):
            for j in desc_topo:
                parents = self.parent_indices[j]
                if len(parents) == 0:
                    X_do[:, j:j+1] = self.gene_scms[j](X_do[:, :1])
                else:
                    parent_vals = X_do[:, parents]
                    X_do[:, j:j+1] = self.gene_scms[j](parent_vals)
        
        return X_do
    
    def do_intervention_differentiable(
        self,
        X_source: torch.Tensor,
        intervention_indices: list[int],
        intervention_values: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable do-intervention (autograd safe)."""
        device = X_source.device

        new_cols: dict[int, torch.Tensor] = {}
        for k, idx in enumerate(intervention_indices):
            new_cols[idx] = intervention_values[:, k:k+1]

        
        desc_set = set()
        for idx in intervention_indices:
            desc_set.update(self._descendants[idx])
        desc_set -= set(intervention_indices)

        if desc_set:
            desc_topo = sorted(desc_set, key=lambda g: self._topo_pos[g])
            for j in desc_topo:
                parents = self.parent_indices[j]
                if len(parents) == 0:
                    col0 = new_cols.get(0, X_source[:, 0:1])
                    new_cols[j] = self.gene_scms[j](col0)
                else:
                    parent_vals = torch.cat([
                        new_cols[p] if p in new_cols else X_source[:, p:p+1]
                        for p in parents
                    ], dim=1)
                    new_cols[j] = self.gene_scms[j](parent_vals)

        if not new_cols:
            return X_source

        mod_indices = sorted(new_cols.keys())
        mod_vals = torch.cat([new_cols[j] for j in mod_indices], dim=1)
        mask = torch.zeros(self.n_genes, dtype=torch.bool, device=device)
        mod_idx_t = torch.tensor(mod_indices, dtype=torch.long, device=device)
        mask[mod_idx_t] = True
        mod_full = torch.zeros_like(X_source)
        mod_full[:, mod_idx_t] = mod_vals
        return torch.where(mask.unsqueeze(0), mod_full, X_source)

    def cascade_do_intervention_differentiable(
        self,
        X_source: torch.Tensor,
        intervention_indices: list[int],
        intervention_values: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable full-graph cascade do-intervention."""
        intervention_set = set(intervention_indices)
        new_cols: dict[int, torch.Tensor] = {}
        batch = X_source.shape[0]

        for k, idx in enumerate(intervention_indices):
            new_cols[idx] = intervention_values[:, k:k+1]

        if self._root_genes:
            for j in self._root_genes:
                if j not in intervention_set:
                    new_cols[j] = self.gene_scms[j].bias.unsqueeze(0).expand(batch, -1)

        for j in self._nonroot_topo:
            if j in intervention_set:
                continue
            parents = self.parent_indices[j]
            parent_vals = torch.cat([
                new_cols[p] if p in new_cols else X_source[:, p:p+1]
                for p in parents
            ], dim=1)
            new_cols[j] = self.gene_scms[j](parent_vals)

        all_cols = []
        for j in range(self.n_genes):
            if j in new_cols:
                all_cols.append(new_cols[j])
            else:
                all_cols.append(X_source[:, j:j+1])
        return torch.cat(all_cols, dim=1)

    def compute_loss(
        self, X: torch.Tensor, sampling_prob: float = 0.0
    ) -> dict:
        """Training loss: observation reconstruction."""
        if sampling_prob > 0.0:
            X_hat = self.cascade_forward(X, sampling_prob)
        else:
            X_hat = self.forward(X)
        recon_loss = nn.functional.mse_loss(X_hat, X)
        
        return {
            "total": recon_loss,
            "recon": recon_loss,
        }

    def compute_interventional_loss(
        self,
        ctrl_batch: torch.Tensor,
        pert_batch: torch.Tensor,
        label_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Interventional consistency loss: compare do-intervention predictions with observed."""
        batch_size = ctrl_batch.shape[0]

        
        groups: dict[tuple, list[int]] = {}
        for b in range(batch_size):
            targets = torch.where(label_batch[b] > 0.5)[0]
            if len(targets) == 0:
                continue
            key = tuple(targets.tolist())
            groups.setdefault(key, []).append(b)

        if not groups:
            return torch.tensor(0.0, device=ctrl_batch.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=ctrl_batch.device)
        count = 0

        for int_indices_tuple, sample_ids in groups.items():
            int_indices = list(int_indices_tuple)

            
            has_desc = any(len(self._descendants[g]) > 0 for g in int_indices)
            if not has_desc:
                continue

            idx = sample_ids
            n = len(idx)

            ctrl_g = ctrl_batch[idx]
            pert_g = pert_batch[idx]
            int_values = pert_g[:, int_indices]

            pred = self.do_intervention_differentiable(
                ctrl_g, int_indices, int_values
            )

            mask = label_batch[idx[0]] < 0.5
            loss = nn.functional.mse_loss(pred[:, mask], pert_g[:, mask])
            total_loss = total_loss + loss * n
            count += n

        if count == 0:
            return torch.tensor(0.0, device=ctrl_batch.device, requires_grad=True)

        return total_loss / count
