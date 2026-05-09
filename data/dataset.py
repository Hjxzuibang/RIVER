"""Perturbation Dataset."""

from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader


class PerturbationDataset(Dataset):
    """Single-cell perturbation dataset from preprocessed h5ad files."""
    
    def __init__(
        self,
        data_path: Path,
        split: str = "train",
        split_seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        cell_subsample_ratio: float = 1.0,
        cell_subsample_seed: int = 0,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.cell_subsample_ratio = cell_subsample_ratio
        self.cell_subsample_seed = cell_subsample_seed
        
        
        self.adata = sc.read_h5ad(self.data_path)
        self.gene_names = list(self.adata.var_names)
        self.n_genes = len(self.gene_names)
        
        
        self._prepare_data(split_seed, train_ratio, val_ratio)
    
    def _prepare_data(self, seed: int, train_ratio: float, val_ratio: float):
        adata = self.adata
        subsample = self.cell_subsample_ratio < 1.0
        ss_rng = np.random.RandomState(self.cell_subsample_seed) if subsample else None
        
        
        ctrl_mask = adata.obs["target_gene"] == "non-targeting"
        ctrl_cells = adata[ctrl_mask]
        ctrl_expr_full = np.array(ctrl_cells.X.todense() 
                                  if hasattr(ctrl_cells.X, 'todense') 
                                  else ctrl_cells.X)
        if subsample:
            n_ctrl = ctrl_expr_full.shape[0]
            n_keep = max(50, int(n_ctrl * self.cell_subsample_ratio))
            n_keep = min(n_keep, n_ctrl)
            idx = ss_rng.choice(n_ctrl, n_keep, replace=False)
            self.ctrl_expr = ctrl_expr_full[idx]
        else:
            self.ctrl_expr = ctrl_expr_full
        self.ctrl_mean = self.ctrl_expr.mean(axis=0)
        
        
        pert_mask = ~ctrl_mask
        pert_adata = adata[pert_mask]
        
        
        all_perts = sorted(pert_adata.obs["target_gene"].unique().tolist())
        
        
        rng = np.random.RandomState(seed)
        n = len(all_perts)
        indices = rng.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * (train_ratio + val_ratio))
        
        shuffled_perts = [all_perts[i] for i in indices]
        split_map = {
            "train": shuffled_perts[:n_train],
            "val": shuffled_perts[n_train:n_val],
            "test": shuffled_perts[n_val:],
        }
        
        selected_perts = set(split_map[self.split])
        
        
        self.samples = []
        for pert in selected_perts:
            mask = pert_adata.obs["target_gene"] == pert
            pert_cells = pert_adata[mask]
            expr = np.array(pert_cells.X.todense() 
                           if hasattr(pert_cells.X, 'todense') 
                           else pert_cells.X)
            
            # Cell sub-sampling (CausalDN uncertainty protocol)
            if subsample:
                n_cells = expr.shape[0]
                n_keep = max(50, int(n_cells * self.cell_subsample_ratio))
                n_keep = min(n_keep, n_cells)
                idx = ss_rng.choice(n_cells, n_keep, replace=False)
                expr = expr[idx]
            
            pert_mean = expr.mean(axis=0).flatten()
            
            
            pert_genes = pert.split("+")
            pert_label = np.zeros(self.n_genes, dtype=np.float32)
            for g in pert_genes:
                if g in self.gene_names:
                    pert_label[self.gene_names.index(g)] = 1.0
            
            self.samples.append({
                "pert_expr": pert_mean.astype(np.float32),
                "ctrl_mean": self.ctrl_mean.flatten().astype(np.float32),
                "pert_label": pert_label,
                "pert_name": pert,
                "n_cells": int(mask.sum()),
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "pert_expr": torch.from_numpy(s["pert_expr"]),
            "ctrl_mean": torch.from_numpy(s["ctrl_mean"]),
            "pert_label": torch.from_numpy(s["pert_label"]),
        }
    
    def get_gene_names(self):
        return self.gene_names
    
    def get_ctrl_mean(self):
        return torch.from_numpy(self.ctrl_mean.flatten().astype(np.float32))

    def compute_per_pert_degs(self, n_top: int = 1000, n_min: int = 100,
                              pval_threshold: float = 0.05):
        """Compute per-perturbation top DEGs (CausalDN evaluation protocol).

        Uses Wilcoxon signed-rank test + BH correction via scanpy.
        Results stored as ``sample["deg_indices"]`` (np.ndarray of gene indices).
        """
        adata = self.adata
        pert_names_in_split = {s["pert_name"] for s in self.samples}

        # Run DE for all perturbation groups vs control in one pass
        groups = [p for p in adata.obs["target_gene"].unique()
                  if p != "non-targeting" and p in pert_names_in_split]
        sc.tl.rank_genes_groups(
            adata, groupby="target_gene", groups=groups,
            reference="non-targeting", method="wilcoxon",
            corr_method="benjamini-hochberg",
        )

        gene_name_to_idx = {g: i for i, g in enumerate(self.gene_names)}

        for sample in self.samples:
            pert_name = sample["pert_name"]
            try:
                df = sc.get.rank_genes_groups_df(adata, group=pert_name)
            except KeyError:
                # Perturbation not found in DE results (too few cells, etc.)
                sample["deg_indices"] = np.arange(self.n_genes, dtype=np.int64)
                continue

            df["abs_lfc"] = df["logfoldchanges"].abs()
            sig = df[df["pvals_adj"] < pval_threshold].sort_values(
                "abs_lfc", ascending=False)

            if len(sig) >= n_top:
                top_genes = sig["names"].head(n_top).tolist()
            else:
                # Fill with top abs_lfc genes until n_min or n_top
                remaining = df[df["pvals_adj"] >= pval_threshold].sort_values(
                    "abs_lfc", ascending=False)
                top_genes = sig["names"].tolist()
                needed = max(n_min, n_top) - len(top_genes)
                top_genes += remaining["names"].head(needed).tolist()

            indices = [gene_name_to_idx[g] for g in top_genes
                       if g in gene_name_to_idx]
            sample["deg_indices"] = np.array(indices, dtype=np.int64)

    def build_lfc_lookup(self) -> dict:
        """Build gene_index -> LFC_profile lookup for CausalDN-style Spearman/Pearson.

        Returns dict mapping gene index to its LFC profile (pert_mean - ctrl_mean).
        Used only for evaluation metrics (not for training or prediction).
        Uses cell_subsample_ratio if set (consistent with sample pert_mean).
        """
        adata = self.adata
        ctrl_mask = adata.obs["target_gene"] == "non-targeting"
        ctrl_mean = self.ctrl_mean.flatten()
        subsample = self.cell_subsample_ratio < 1.0
        ss_rng = np.random.RandomState(self.cell_subsample_seed + 10000) if subsample else None

        gene_name_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        lfc_lookup = {}  # gene_index -> LFC profile [n_genes]

        pert_mask = ~ctrl_mask
        pert_adata = adata[pert_mask]
        all_perts = pert_adata.obs["target_gene"].unique().tolist()

        for pert_name in all_perts:
            # Only handle single-gene perturbations for LFC lookup
            pert_genes = pert_name.split("+")
            if len(pert_genes) != 1:
                continue
            gene = pert_genes[0]
            if gene not in gene_name_to_idx:
                continue
            gene_idx = gene_name_to_idx[gene]

            mask = pert_adata.obs["target_gene"] == pert_name
            cells = pert_adata[mask]
            expr = np.array(cells.X.todense()
                           if hasattr(cells.X, 'todense')
                           else cells.X)
            if subsample:
                n_cells = expr.shape[0]
                n_keep = max(50, int(n_cells * self.cell_subsample_ratio))
                n_keep = min(n_keep, n_cells)
                idx = ss_rng.choice(n_cells, n_keep, replace=False)
                expr = expr[idx]
            pert_mean = expr.mean(axis=0).flatten()
            lfc = (pert_mean - ctrl_mean).astype(np.float32)
            lfc_lookup[gene_idx] = lfc

        return lfc_lookup


def create_dataloader(
    data_path: Path,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for a given split."""
    dataset = PerturbationDataset(data_path, split=split, **kwargs)
    shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
