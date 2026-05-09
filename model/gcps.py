"""Stage 3: Greedy Causal Perturbation Search (GCPS)."""

import numpy as np
import torch

from .neural_scm import NeuralSCM


class GreedyCausalPerturbationSearch:
    """Greedy search for minimal intervention set using Neural SCM."""
    
    def __init__(
        self,
        scm: NeuralSCM,
        adjacency: np.ndarray,
        diff_threshold: float = 0.5,
        convergence_threshold: float = 0.1,
        max_interventions: int = 10,
        score_mode: str = "raw",
        pairwise_k: int = 50,
    ):
        self.scm = scm
        self.adjacency = adjacency
        self.n_genes = adjacency.shape[0]
        self.diff_threshold = diff_threshold
        self.convergence_threshold = convergence_threshold
        self.max_interventions = max_interventions
        self.score_mode = score_mode
        self.pairwise_k = pairwise_k
        self.train_signatures: dict[int, np.ndarray] | None = None
    
    def find_differential_genes(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> list[int]:
        """Identify differential gene set D."""
        diff = (x_target - x_source).abs()
        return torch.where(diff > self.diff_threshold)[0].tolist()
    
    def find_causal_roots(self, diff_genes: list[int]) -> list[int]:
        """Find causal roots: differential genes with no differential parents."""
        diff_set = set(diff_genes)
        roots = []
        
        for g in diff_genes:
            parents = set(np.where(self.adjacency[:, g] > 0)[0].tolist())
            if not parents.intersection(diff_set):
                roots.append(g)
        
        return roots
    
    def get_candidate_genes(self, diff_genes: list[int]) -> list[int]:
        """Get candidate intervention genes: causal roots + ancestors."""
        roots = self.find_causal_roots(diff_genes)
        
        
        ancestors = set()
        for g in diff_genes:
            self._find_ancestors(g, ancestors)
        
        candidates = set(roots) | ancestors
        return sorted(candidates)
    
    def _find_ancestors(self, gene: int, visited: set, max_depth: int = 5):
        """Recursively find ancestor genes."""
        if max_depth <= 0:
            return
        
        parents = np.where(self.adjacency[:, gene] > 0)[0].tolist()
        for p in parents:
            if p not in visited:
                visited.add(p)
                self._find_ancestors(p, visited, max_depth - 1)
    
    @torch.no_grad()
    def search(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        weight_mode: str = "uniform",
    ) -> dict:
        """Greedy search for minimal intervention set (vectorized)."""
        self.scm.eval()
        device = x_source.device

        
        diff_genes = self.find_differential_genes(x_source, x_target)
        if not diff_genes:
            return {
                "intervention_set": [],
                "intervention_values": torch.tensor([]),
                "scores": [],
                "gene_scores": torch.zeros(self.n_genes, device=device),
                "final_distance": 0.0,
            }

        
        candidates = self.get_candidate_genes(diff_genes)

        
        weights = self._compute_weights(x_source, x_target, diff_genes, weight_mode)
        diff_idx = torch.tensor(diff_genes, device=device)

        
        gene_scores = self._compute_all_gene_scores(
            x_source, x_target, diff_idx, weights
        )

        
        intervention_set: list[int] = []
        intervention_values: list[float] = []
        scores: list[float] = []
        remaining = sorted(set(candidates))

        prev_dist = self._weighted_distance(
            x_source, x_target, diff_genes, weights
        )

        for step in range(self.max_interventions):
            if not remaining:
                break

            n_cand = len(remaining)

            
            x_batch = x_source.unsqueeze(0).expand(n_cand, -1).clone()

            frozen_set = set(intervention_set)
            for k, idx in enumerate(intervention_set):
                x_batch[:, idx] = intervention_values[k]

            cand_set = {}
            for b, g in enumerate(remaining):
                x_batch[b, g] = x_target[g]
                cand_set[g] = b

            
            nfr = [j for j in self.scm._root_genes if j not in frozen_set]
            if nfr:
                biases = torch.cat([self.scm.gene_scms[j].bias for j in nfr])
                x_batch[:, nfr] = biases.unsqueeze(0).expand(n_cand, -1)
                
                for g, b in cand_set.items():
                    if g in set(nfr):
                        x_batch[b, g] = x_target[g]

            for j in self.scm._nonroot_topo:
                if j in frozen_set:
                    continue

                parents = self.scm.parent_indices[j]
                parent_vals = x_batch[:, parents]
                x_batch[:, j:j+1] = self.scm.gene_scms[j](parent_vals)

                if j in cand_set:
                    b = cand_set[j]
                    x_batch[b, j] = x_target[j]

            
            dists = (x_batch[:, diff_idx] - x_target[diff_idx].unsqueeze(0)).abs()
            dists = (dists * weights.unsqueeze(0)).sum(dim=1)  # [n_cand]
            gains = prev_dist - dists

            best_idx = gains.argmax().item()
            best_score = gains[best_idx].item()

            if best_score <= 0:
                break

            best_gene = remaining[best_idx]
            intervention_set.append(best_gene)
            intervention_values.append(x_target[best_gene].item())
            scores.append(best_score)
            remaining = [g for g in remaining if g != best_gene]

            
            prev_dist = dists[best_idx].item()

            if prev_dist < self.convergence_threshold:
                break

        final_values_tensor = torch.tensor(intervention_values, device=device)

        
        if self.score_mode == "pairwise":
            gene_scores, intervention_set, final_values_tensor = (
                self._apply_pairwise_scoring(
                    gene_scores, x_source, x_target, diff_genes, weights, device
                )
            )
        elif self.score_mode == "signature":
            gene_scores, intervention_set, final_values_tensor = (
                self._apply_signature_scoring(
                    x_source, x_target, device
                )
            )

        return {
            "intervention_set": intervention_set,
            "intervention_values": final_values_tensor,
            "scores": scores,
            "gene_scores": gene_scores,
            "final_distance": prev_dist if intervention_set else float('inf'),
        }
    
    @torch.no_grad()
    def _compute_all_gene_scores(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        diff_idx: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute single-gene marginal gain for all genes."""
        device = x_source.device
        n = self.n_genes
        prev_dist = (x_source[diff_idx] - x_target[diff_idx]).abs()
        prev_dist = (prev_dist * weights).sum()

        x_batch = self._batch_do_forward(x_source, x_target)

        
        dists = (x_batch[:, diff_idx] - x_target[diff_idx].unsqueeze(0)).abs()
        dists = (dists * weights.unsqueeze(0)).sum(dim=1)  # [n]
        gains = prev_dist - dists  # [n]

        return gains

    @torch.no_grad()
    def _batch_do_forward(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        """Batch do-intervention: row i = do(gene_i = x_target[i])."""
        n = self.n_genes
        device = x_source.device
        x_batch = x_source.unsqueeze(0).expand(n, -1).clone()

        for i in range(n):
            x_batch[i, i] = x_target[i]

        if self.scm._root_genes:
            root_biases = torch.cat(
                [self.scm.gene_scms[j].bias for j in self.scm._root_genes]
            )  # [n_roots]
            x_batch[:, self.scm._root_genes] = root_biases.unsqueeze(0).expand(n, -1)
            
            root_idx = torch.tensor(self.scm._root_genes, device=device)
            x_batch[root_idx, root_idx] = x_target[root_idx]

        for j in self.scm._nonroot_topo:
            parents = self.scm.parent_indices[j]
            parent_vals = x_batch[:, parents]
            x_batch[:, j:j+1] = self.scm.gene_scms[j](parent_vals)
            x_batch[j, j] = x_target[j]

        return x_batch

    def _compute_weights(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        diff_genes: list[int],
        mode: str,
    ) -> torch.Tensor:
        """Compute weights for differential genes."""
        n = len(diff_genes)
        
        if mode == "uniform":
            return torch.ones(n, device=x_source.device)
        elif mode == "diff":
            
            diffs = (x_target[diff_genes] - x_source[diff_genes]).abs()
            return diffs / diffs.sum()
        else:
            return torch.ones(n, device=x_source.device)
    
    def _weighted_distance(
        self,
        x_pred: torch.Tensor,
        x_target: torch.Tensor,
        diff_genes: list[int],
        weights: torch.Tensor,
    ) -> float:
        """Weighted distance on differential genes."""
        diff = (x_pred[diff_genes] - x_target[diff_genes]).abs()
        return (diff * weights).sum().item()

    # ------------------------------------------------------------------
    # Two-Stage Pairwise Scoring (TSPS) for multi-gene perturbations
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _batch_pairwise_do_forward(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        pairs: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Batch pairwise do-intervention forward pass."""
        n_pairs = len(pairs)
        device = x_source.device
        x_batch = x_source.unsqueeze(0).expand(n_pairs, -1).clone()

        intervention_map: dict[int, list[int]] = {}
        for b, (gi, gj) in enumerate(pairs):
            x_batch[b, gi] = x_target[gi]
            x_batch[b, gj] = x_target[gj]
            intervention_map.setdefault(gi, []).append(b)
            intervention_map.setdefault(gj, []).append(b)

        root_set = set(self.scm._root_genes)
        if self.scm._root_genes:
            root_biases = torch.cat(
                [self.scm.gene_scms[j].bias for j in self.scm._root_genes]
            )
            x_batch[:, self.scm._root_genes] = root_biases.unsqueeze(0).expand(
                n_pairs, -1
            )
            for j in self.scm._root_genes:
                if j in intervention_map:
                    batch_indices = intervention_map[j]
                    x_batch[batch_indices, j] = x_target[j]

        for j in self.scm._nonroot_topo:
            parents = self.scm.parent_indices[j]
            parent_vals = x_batch[:, parents]
            x_batch[:, j : j + 1] = self.scm.gene_scms[j](parent_vals)
            if j in intervention_map:
                batch_indices = intervention_map[j]
                x_batch[batch_indices, j] = x_target[j]

        return x_batch

    @torch.no_grad()
    def _apply_pairwise_scoring(
        self,
        gene_scores: torch.Tensor,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        diff_genes: list[int],
        weights: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        """Two-stage pairwise scoring (TSPS)."""
        from itertools import combinations

        n = self.n_genes
        k = min(self.pairwise_k, n)

        
        top_k_indices = gene_scores.argsort(descending=True)[:k].tolist()

        if len(top_k_indices) < 2:
            return gene_scores, top_k_indices, torch.tensor(
                [x_target[g].item() for g in top_k_indices], device=device
            )

        
        pairs = list(combinations(top_k_indices, 2))
        x_batch = self._batch_pairwise_do_forward(x_source, x_target, pairs)

        
        diff_idx = torch.tensor(diff_genes, device=device)
        dists = (x_batch[:, diff_idx] - x_target[diff_idx].unsqueeze(0)).abs()
        dists = (dists * weights.unsqueeze(0)).sum(dim=1)  # [n_pairs]
        pair_scores = -dists

        
        best_pair_idx = pair_scores.argmax().item()
        best_gi, best_gj = pairs[best_pair_idx]

        
        new_gene_scores = torch.full((n,), float("-inf"), device=device)
        for pidx, (gi, gj) in enumerate(pairs):
            s = pair_scores[pidx]
            if s > new_gene_scores[gi]:
                new_gene_scores[gi] = s
            if s > new_gene_scores[gj]:
                new_gene_scores[gj] = s

        
        pairwise_min = new_gene_scores[new_gene_scores > float("-inf")].min().item()
        for i in range(n):
            if new_gene_scores[i] == float("-inf"):
                new_gene_scores[i] = pairwise_min - 1.0 + gene_scores[i] / (
                    gene_scores.max().item() + 1e-8
                )

        
        max_score = new_gene_scores.max().item()
        new_gene_scores[best_gi] = max_score + 2.0
        new_gene_scores[best_gj] = max_score + 1.0

        intervention_set = [best_gi, best_gj]
        intervention_values = torch.tensor(
            [x_target[best_gi].item(), x_target[best_gj].item()], device=device
        )

        return new_gene_scores, intervention_set, intervention_values

    # ------------------------------------------------------------------
    # Training Signature Matching for multi-gene perturbations
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_signature_scoring(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, list[int], torch.Tensor]:
        """Training signature matching for multi-gene perturbation identification."""
        from itertools import combinations

        n = self.n_genes
        k = min(self.pairwise_k, n)

        if self.train_signatures is None or len(self.train_signatures) == 0:
            # Fallback: DGE top-2
            delta = (x_target - x_source).abs()
            gene_scores = delta.clone()
            top2 = delta.argsort(descending=True)[:2].tolist()
            return gene_scores, top2, torch.tensor(
                [x_target[g].item() for g in top2], device=device)

        # Observed expression change
        delta_obs = (x_target - x_source).cpu().numpy()

        # DGE ranking for candidate selection
        dge = np.abs(delta_obs)
        dge_ranked = np.argsort(-dge)

        sig_candidates = []
        for g in dge_ranked:
            g = int(g)
            if g in self.train_signatures:
                sig_candidates.append(g)
            if len(sig_candidates) >= k:
                break

        if len(sig_candidates) < 2:
            # Not enough candidates with signatures, fallback
            delta = (x_target - x_source).abs()
            gene_scores = delta.clone()
            top2 = delta.argsort(descending=True)[:2].tolist()
            return gene_scores, top2, torch.tensor(
                [x_target[g].item() for g in top2], device=device)

        # Enumerate all candidate pairs and compute Pearson correlation
        pairs = list(combinations(sig_candidates, 2))

        # Vectorized computation
        # Build signature matrix for all candidates
        sig_matrix = np.stack([self.train_signatures[g] for g in sig_candidates])  # [K, n_genes]

        # Compute pairwise sums and correlations
        best_pair_idx = -1
        best_corr = -np.inf
        pair_scores_list = np.empty(len(pairs), dtype=np.float64)

        # Pre-center delta_obs for faster Pearson
        obs_mean = delta_obs.mean()
        obs_centered = delta_obs - obs_mean
        obs_std = np.sqrt(np.sum(obs_centered ** 2))

        # Map candidate gene to index in sig_matrix
        cand_to_idx = {g: i for i, g in enumerate(sig_candidates)}

        for pidx, (gi, gj) in enumerate(pairs):
            expected = sig_matrix[cand_to_idx[gi]] + sig_matrix[cand_to_idx[gj]]
            exp_mean = expected.mean()
            exp_centered = expected - exp_mean
            exp_std = np.sqrt(np.sum(exp_centered ** 2))
            if exp_std < 1e-12 or obs_std < 1e-12:
                corr = 0.0
            else:
                corr = np.sum(exp_centered * obs_centered) / (exp_std * obs_std)
            pair_scores_list[pidx] = corr
            if corr > best_corr:
                best_corr = corr
                best_pair_idx = pidx

        best_gi, best_gj = pairs[best_pair_idx]

        # Map pair_scores to gene_scores
        new_gene_scores = torch.full((n,), float("-inf"), device=device)
        for pidx, (gi, gj) in enumerate(pairs):
            s = pair_scores_list[pidx]
            if s > new_gene_scores[gi].item():
                new_gene_scores[gi] = s
            if s > new_gene_scores[gj].item():
                new_gene_scores[gj] = s

        # Non-candidate genes: use DGE score scaled below pairwise min
        pairwise_valid = new_gene_scores[new_gene_scores > float("-inf")]
        if len(pairwise_valid) > 0:
            pairwise_min = pairwise_valid.min().item()
        else:
            pairwise_min = 0.0
        dge_t = torch.from_numpy(dge).to(device)
        dge_max = dge_t.max().item() + 1e-8
        for i in range(n):
            if new_gene_scores[i] == float("-inf"):
                new_gene_scores[i] = pairwise_min - 1.0 + dge_t[i] / dge_max

        # Ensure best pair is top-2
        max_score = new_gene_scores.max().item()
        new_gene_scores[best_gi] = max_score + 2.0
        new_gene_scores[best_gj] = max_score + 1.0

        intervention_set = [best_gi, best_gj]
        intervention_values = torch.tensor(
            [x_target[best_gi].item(), x_target[best_gj].item()], device=device)

        return new_gene_scores, intervention_set, intervention_values
