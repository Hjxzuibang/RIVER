"""Preprocess Module

Convert raw single-cell perturbation data into model-ready format.

Pipeline:
1. Backup raw counts -> layers["counts"]
2. Gene filtering (min_cells=3)
3. HVG selection + perturbation gene union
4. Library size -> obs["ncounts"] = X.sum()/1e4
5. Normalize + log1p -> X
6. Perturbation efficiency filtering (Z-score based)
7. Rare perturbation filtering (>=5 cells)
8. Final gene filtering
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


# ============================================================================
# Configuration
# ============================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = _PROJECT_ROOT / "rawdata"
OUTPUT_DIR = _PROJECT_ROOT / "processed_data"

MIN_CELLS = 3
TARGET_SUM = 1e4
N_HVG = 2000
PERT_EFFICIENCY_THRESHOLD = 1.0
MIN_CELLS_PER_PERT = 5

PERTURBATION_COL = "target_gene"
CONTROL_LABEL = "non-targeting"
SKIP_DIRS = {"demo"}


# ============================================================================
# Core functions
# ============================================================================

def preprocess_single_file(
    input_path: Path,
    output_path: Path,
    n_hvg: int = N_HVG,
    min_cells: int = MIN_CELLS,
    target_sum: float = TARGET_SUM,
    pert_efficiency_threshold: float = PERT_EFFICIENCY_THRESHOLD,
    min_cells_per_pert: int = MIN_CELLS_PER_PERT,
    filter_noncoding: bool = False,
) -> dict:
    """Preprocess a single h5ad file."""
    stats = {
        "input_path": str(input_path),
        "n_cells_input": 0,
        "n_genes_input": 0,
        "n_cells_output": 0,
        "n_genes_output": 0,
        "n_perturbations_output": 0,
        "n_control_cells": 0,
        "n_removed_lfc_filter": 0,
        "n_removed_pert_efficiency": 0,
        "n_removed_rare_perts": 0,
    }
    
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"{'='*60}")
    
    import sys
    print("\n[Step 1] Loading data...", end=" ", flush=True)
    sys.stdout.flush()
    adata = sc.read_h5ad(input_path)
    print("Done!")
    stats["n_cells_input"] = adata.n_obs
    stats["n_genes_input"] = adata.n_vars
    print(f"  Input shape: {adata.shape}")
    
    print("\n[Step 2] Backing up raw counts to layers['counts']...")
    adata.layers["counts"] = adata.X.copy()
    
    print(f"\n[Step 3] Filtering genes (min_cells={min_cells})...")
    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"  Genes: {n_genes_before} -> {adata.n_vars}")
    
    print(f"\n[Step 4] Selecting genes: HVG ({n_hvg}) + perturbation targets union...")
    perturbation_col = PERTURBATION_COL
    control_label = CONTROL_LABEL
    
    all_perturbations = set(adata.obs[perturbation_col].unique()) - {control_label}
    pert_genes_in_data = all_perturbations & set(adata.var_names)
    
    import re
    _NONCODING_PATTERN = re.compile(
        r'^(RP[0-9]|AC[0-9]|AF[0-9]|AL[0-9]|AP[0-9]|CH[0-9]|'
        r'LINC|MIR[0-9]|CTD-|CTC-|CTB-|XXbac)'
    )
    if filter_noncoding:
        all_gene_names = set(adata.var_names)
        symbol_col = None
        for col in ['gene_name', 'gene_symbol', 'symbol']:
            if col in adata.var.columns:
                symbol_col = col
                break
        
        if symbol_col:
            name_map = dict(zip(adata.var_names, adata.var[symbol_col]))
        else:
            name_map = {g: g for g in all_gene_names}
        
        noncoding = {g for g in all_gene_names if _NONCODING_PATTERN.match(str(name_map.get(g, g)))}
        
        noncoding -= pert_genes_in_data
        
        if noncoding:
            coding_genes = sorted(all_gene_names - noncoding)
            n_before = adata.n_vars
            adata = adata[:, coding_genes].copy()
            print(f"  [Filter] Removed {len(noncoding)} non-coding genes ({len(noncoding)/n_before*100:.1f}%)")
            print(f"  Genes: {n_before} -> {adata.n_vars}")
    
    adata_tmp = adata.copy()
    sc.pp.normalize_total(adata_tmp, target_sum=target_sum)
    sc.pp.log1p(adata_tmp)
    n_hvg_actual = min(n_hvg, adata_tmp.n_vars)
    sc.pp.highly_variable_genes(
        adata_tmp, n_top_genes=n_hvg_actual, subset=False, flavor="seurat_v3",
        layer="counts" if "counts" in adata_tmp.layers else None,
    )
    hvg_genes = set(adata_tmp.var_names[adata_tmp.var.highly_variable])
    del adata_tmp
    
    selected_genes = hvg_genes | pert_genes_in_data
    selected_genes = sorted(selected_genes & set(adata.var_names))
    
    n_genes_before = adata.n_vars
    adata = adata[:, selected_genes].copy()
    print(f"  HVG: {len(hvg_genes)}, Perturbation genes in data: {len(pert_genes_in_data)}")
    print(f"  Union (deduplicated): {len(selected_genes)}")
    print(f"  Genes: {n_genes_before} -> {adata.n_vars}")
    
    print(f"\n[Step 5] Normalizing + log1p (CASCADE design)...")
    sc.pp.normalize_total(adata, target_sum=target_sum, key_added="ncounts")
    sc.pp.log1p(adata)
    print(f"  ncounts = X.sum()/1e4 (CASCADE design)")
    print(f"  ncounts range: {adata.obs['ncounts'].min():.4f} - {adata.obs['ncounts'].max():.4f}")
    print(f"  X now in normalized_log1p space")
    
    print(f"\n[Step 7] Filtering perturbations by LFC (counts<0 AND log1p<0)...")
    n_cells_before = adata.n_obs
    n_perts_before = adata.obs[perturbation_col].nunique() - 1
    
    gene_to_idx = pd.Series(range(len(adata.var_names)), index=adata.var_names)
    pert_genes = adata.obs[perturbation_col]
    gene_indices = pert_genes.map(gene_to_idx).fillna(-1).astype(int).values
    gene_indices[pert_genes.values == control_label] = -1
    valid_mask = gene_indices >= 0
    
    if valid_mask.any():
        unique_gene_indices = np.unique(gene_indices[valid_mask])
        X_log1p = adata.X
        X_counts = adata.layers["counts"]
        
        X_log1p_subset = X_log1p[:, unique_gene_indices]
        X_counts_subset = X_counts[:, unique_gene_indices]
        
        ctrl_mask = (adata.obs[perturbation_col] == control_label).values
        ctrl_log1p_sparse = X_log1p_subset[ctrl_mask]
        ctrl_counts_sparse = X_counts_subset[ctrl_mask]
        if hasattr(ctrl_log1p_sparse, 'toarray'):
            ctrl_mu_log1p = np.asarray(ctrl_log1p_sparse.mean(axis=0)).flatten()
            ctrl_mu_counts = np.asarray(ctrl_counts_sparse.mean(axis=0)).flatten()
        else:
            ctrl_mu_log1p = ctrl_log1p_sparse.mean(axis=0).flatten()
            ctrl_mu_counts = ctrl_counts_sparse.mean(axis=0).flatten()
        
        row_indices = np.where(valid_mask)[0]
        subset_indices = np.searchsorted(unique_gene_indices, gene_indices[valid_mask])
        
        def extract_sparse_diag(sparse_mat, local_col_idx):
            """Extract mat[i, col_idx[i]] from a sparse matrix."""
            if not hasattr(sparse_mat, 'toarray'):
                return sparse_mat[np.arange(sparse_mat.shape[0]), local_col_idx]
            
            from scipy.sparse import csr_matrix as csr_mat
            n = sparse_mat.shape[0]
            selector = csr_mat(
                (np.ones(n, dtype=np.float64), (np.arange(n), local_col_idx)),
                shape=(n, sparse_mat.shape[1])
            )
            result = np.asarray(sparse_mat.multiply(selector).sum(axis=1)).flatten()
            return result
        
        temp_log1p = X_log1p_subset[row_indices]
        temp_counts = X_counts_subset[row_indices]
        cell_log1p = extract_sparse_diag(temp_log1p, subset_indices)
        cell_counts = extract_sparse_diag(temp_counts, subset_indices)
        
        df = pd.DataFrame({
            'pert': pert_genes.values[valid_mask],
            'gene_subset_idx': subset_indices,
            'log1p': cell_log1p,
            'counts': cell_counts,
        })
        pert_stats_df = df.groupby('pert', observed=True).agg({
            'gene_subset_idx': 'first',
            'log1p': 'mean',
            'counts': 'mean',
        })
        
        gene_idx = pert_stats_df['gene_subset_idx'].values.astype(int)
        ctrl_mu_log1p_pert = ctrl_mu_log1p[gene_idx]
        ctrl_mu_counts_pert = ctrl_mu_counts[gene_idx]
        lfc_log1p = pert_stats_df['log1p'].values - ctrl_mu_log1p_pert
        lfc_counts = pert_stats_df['counts'].values - ctrl_mu_counts_pert
        
        invalid_mask = (lfc_counts >= 0) | (lfc_log1p >= 0)
        invalid_perts = pert_stats_df.index[invalid_mask].tolist()
        
        if invalid_perts:
            mask = ~adata.obs[perturbation_col].isin(invalid_perts)
            n_cells_removed = (~mask).sum()
            adata = adata[mask].copy()
            n_perts_after = adata.obs[perturbation_col].nunique() - 1
            
            stats["n_removed_lfc_filter"] = len(invalid_perts)
            print(f"  Removed {len(invalid_perts)} perturbations with LFC >= 0")
            print(f"  Cells: {n_cells_before} -> {adata.n_obs} (removed {n_cells_removed} cells)")
            print(f"  Target genes: {n_perts_before} -> {n_perts_after}")
        else:
            print(f"  No perturbations filtered (all have LFC < 0)")
    else:
        print(f"  No perturbations to filter")
    
    if pert_efficiency_threshold > 0:
        print(f"\n[Step 8] Filtering by perturbation efficiency (|Z| >= {pert_efficiency_threshold})...")
        n_cells_before = adata.n_obs
        
        gene_to_idx = pd.Series(range(len(adata.var_names)), index=adata.var_names)
        
        pert_genes = adata.obs[perturbation_col]
        gene_indices = pert_genes.map(gene_to_idx).fillna(-1).astype(int).values
        gene_indices[pert_genes.values == control_label] = -1
        
        valid_mask = gene_indices >= 0
        keep_mask = ~valid_mask.copy()
        
        if valid_mask.any():
            unique_gene_indices = np.unique(gene_indices[valid_mask])
            
            X = adata.X
            X_subset = X[:, unique_gene_indices]
            
            ctrl_mask_bool = (adata.obs[perturbation_col] == control_label).values
            ctrl_X_subset = X_subset[ctrl_mask_bool]
            if hasattr(ctrl_X_subset, 'toarray'):
                ctrl_mu = np.asarray(ctrl_X_subset.mean(axis=0)).flatten()
                ctrl_sq = ctrl_X_subset.power(2).mean(axis=0)
                ctrl_sigma = np.sqrt(np.asarray(ctrl_sq).flatten() - ctrl_mu**2) + 1e-8
            else:
                ctrl_mu = ctrl_X_subset.mean(axis=0).flatten()
                ctrl_sigma = ctrl_X_subset.std(axis=0).flatten() + 1e-8
            
            valid_indices = gene_indices[valid_mask]
            row_indices = np.where(valid_mask)[0]
            subset_indices = np.searchsorted(unique_gene_indices, valid_indices)
            
            def extract_sparse_diag(sparse_mat, local_col_idx):
                """Extract mat[i, col_idx[i]] from a sparse matrix."""
                if not hasattr(sparse_mat, 'toarray'):
                    return sparse_mat[np.arange(sparse_mat.shape[0]), local_col_idx]
                from scipy.sparse import csr_matrix as csr_mat
                n = sparse_mat.shape[0]
                selector = csr_mat(
                    (np.ones(n, dtype=np.float64), (np.arange(n), local_col_idx)),
                    shape=(n, sparse_mat.shape[1])
                )
                result = np.asarray(sparse_mat.multiply(selector).sum(axis=1)).flatten()
                return result
            
            temp_X = X_subset[row_indices]
            x_int_values = extract_sparse_diag(temp_X, subset_indices)
            
            mu_values = ctrl_mu[subset_indices]
            sigma_values = ctrl_sigma[subset_indices]
            
            z_scores = (x_int_values - mu_values) / sigma_values
            efficiency_ok = z_scores <= -pert_efficiency_threshold
            keep_mask[valid_mask] = efficiency_ok
        
        n_removed = (~keep_mask).sum()
        perts_before = set(adata.obs[perturbation_col].unique()) - {control_label}
        adata = adata[keep_mask].copy()
        perts_after = set(adata.obs[perturbation_col].unique()) - {control_label}
        
        stats["n_removed_pert_efficiency"] = int(n_removed)
        print(f"  Cells: {n_cells_before} -> {adata.n_obs} (removed {n_removed} low-efficiency cells)")
        print(f"  Target genes: {len(perts_before)} -> {len(perts_after)}")
    else:
        print("\n[Step 8] Skipping perturbation efficiency filter (threshold=0)")
    
    if min_cells_per_pert > 0:
        print(f"\n[Step 9] Filtering rare perturbations (cells < {min_cells_per_pert})...")
        n_cells_before = adata.n_obs
        n_perts_before = adata.obs[perturbation_col].nunique() - 1
        
        pert_counts = adata.obs[perturbation_col].value_counts()
        rare_perts = pert_counts[
            (pert_counts < min_cells_per_pert) & 
            (pert_counts.index != control_label)
        ].index.tolist()
        
        if rare_perts:
            mask = ~adata.obs[perturbation_col].isin(rare_perts)
            n_cells_removed = (~mask).sum()
            adata = adata[mask].copy()
            n_perts_after = adata.obs[perturbation_col].nunique() - 1
            
            stats["n_removed_rare_perts"] = len(rare_perts)
            print(f"  Removed {len(rare_perts)} rare perturbations: {rare_perts[:5]}{'...' if len(rare_perts) > 5 else ''}")
            print(f"  Cells: {n_cells_before} -> {adata.n_obs} (removed {n_cells_removed} cells)")
            print(f"  Target genes: {n_perts_before} -> {n_perts_after}")
        else:
            print(f"  No rare perturbations found (all have >= {min_cells_per_pert} cells)")
    else:
        print("\n[Step 9] Skipping rare perturbation filter (threshold=0)")
    
    print("\n[Step 10] Final gene filtering...")
    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=1)
    print(f"  Genes: {n_genes_before} -> {adata.n_vars}")
    
    stats["n_cells_output"] = adata.n_obs
    stats["n_genes_output"] = adata.n_vars
    stats["n_perturbations_output"] = adata.obs[perturbation_col].nunique()
    stats["n_control_cells"] = (adata.obs[perturbation_col] == control_label).sum()
    
    print(f"\n[Step 11] Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(output_path, compression="gzip")
    
    gene_list_path = output_path.parent / f"gene_list_{output_path.stem}.csv"
    print(f"\n[Step 12] Saving gene list to {gene_list_path}...")
    gene_df = pd.DataFrame({
        "gene_id": adata.var_names.tolist(),
        "gene_name": adata.var["gene_symbol"].tolist() if "gene_symbol" in adata.var.columns else [""] * adata.n_vars,
    })
    gene_df.to_csv(gene_list_path, index=False)
    print(f"  {len(gene_df)} genes exported")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Output shape: {adata.shape}")
    print(f"  Perturbations: {stats['n_perturbations_output']}")
    print(f"  Control cells: {stats['n_control_cells']}")
    print(f"  layers['counts']: {adata.layers['counts'].shape}")
    print(f"  obs['ncounts']: min={adata.obs['ncounts'].min():.4f}, max={adata.obs['ncounts'].max():.4f}")
    print(f"{'='*60}")
    
    return stats


def preprocess_large_file_batched(
    input_path: Path,
    output_path: Path,
    n_batches: int = 3,
    n_hvg: int = N_HVG,
    min_cells: int = MIN_CELLS,
    target_sum: float = TARGET_SUM,
    pert_efficiency_threshold: float = PERT_EFFICIENCY_THRESHOLD,
    min_cells_per_pert: int = MIN_CELLS_PER_PERT,
) -> dict:
    """Batched preprocessing for large h5ad files to avoid OOM."""
    import gc
    
    stats = {
        "input_path": str(input_path),
        "n_cells_input": 0,
        "n_genes_input": 0,
        "n_cells_output": 0,
        "n_genes_output": 0,
        "n_perturbations_output": 0,
        "n_control_cells": 0,
        "n_batches": n_batches,
    }
    
    print(f"\n{'='*60}")
    print(f"Processing (BATCHED MODE): {input_path}")
    print(f"{'='*60}")
    
    # =========================================================================
    # Phase 1: Read metadata in backed mode
    # =========================================================================
    print("\n[Phase 1] Reading metadata (backed mode)...")
    adata_backed = sc.read_h5ad(input_path, backed='r')
    stats["n_cells_input"] = adata_backed.n_obs
    stats["n_genes_input"] = adata_backed.n_vars
    print(f"  Input shape: {adata_backed.shape}")
    
    perturbation_col = PERTURBATION_COL
    control_label = CONTROL_LABEL
    obs_df = adata_backed.obs.copy()
    var_names = adata_backed.var_names.tolist()
    var_df_full = adata_backed.var.copy()
    
    ctrl_mask = obs_df[perturbation_col] == control_label
    ctrl_indices = np.where(ctrl_mask)[0]
    pert_indices = np.where(~ctrl_mask)[0]
    
    print(f"  Control cells: {len(ctrl_indices)}")
    print(f"  Perturbation cells: {len(pert_indices)}")
    
    all_perturbations = set(obs_df[perturbation_col].unique()) - {control_label}
    print(f"  Total perturbation types: {len(all_perturbations)}")
    
    valid_pert_indices = pert_indices
    print(f"  Perturbation cells: {len(valid_pert_indices)}")
    
    adata_backed.file.close()
    del adata_backed
    gc.collect()
    
    # =========================================================================
    # Phase 2: Load control group in backed mode
    # =========================================================================
    print("\n[Phase 2] Loading control group (backed mode)...")
    
    adata_backed = sc.read_h5ad(input_path, backed='r')
    
    print(f"  Computing gene filter mask (min_cells={min_cells})...")
    X_backed = adata_backed.X
    n_genes = X_backed.shape[1]
    
    chunk_size = 100000
    gene_nonzero_counts = np.zeros(n_genes, dtype=np.int64)
    for start in range(0, adata_backed.n_obs, chunk_size):
        end = min(start + chunk_size, adata_backed.n_obs)
        chunk = X_backed[start:end]
        if hasattr(chunk, 'toarray'):
            gene_nonzero_counts += (chunk != 0).sum(axis=0).A1
        else:
            gene_nonzero_counts += (chunk != 0).sum(axis=0)
    
    gene_filter_mask = gene_nonzero_counts >= min_cells
    filtered_var_names = [var_names[i] for i in range(n_genes) if gene_filter_mask[i]]
    filtered_var_indices = np.where(gene_filter_mask)[0]
    filtered_var_df = var_df_full.iloc[filtered_var_indices].copy()
    print(f"  Genes: {n_genes} -> {len(filtered_var_names)}")
    
    print(f"  Extracting control cells ({len(ctrl_indices)} cells)...")
    ctrl_X_raw = X_backed[ctrl_indices][:, filtered_var_indices]
    if hasattr(ctrl_X_raw, 'toarray'):
        ctrl_X_raw = ctrl_X_raw.tocsr()
    
    ctrl_obs = obs_df.iloc[ctrl_indices].copy()
    
    adata_backed.file.close()
    del adata_backed, X_backed
    gc.collect()
    
    ctrl_adata = ad.AnnData(
        X=ctrl_X_raw.copy(),
        obs=ctrl_obs,
        var=filtered_var_df.copy(),
    )
    ctrl_adata.layers["counts"] = ctrl_X_raw.copy()
    del ctrl_X_raw
    gc.collect()
    
    print("  Normalizing control group...")
    sc.pp.normalize_total(ctrl_adata, target_sum=target_sum, key_added="ncounts")
    sc.pp.log1p(ctrl_adata)
    
    print("  Caching control mean (log1p space)...")
    X_ctrl_log1p = ctrl_adata.X
    X_ctrl_counts = ctrl_adata.layers["counts"]
    
    if hasattr(X_ctrl_log1p, 'toarray'):
        ctrl_mu_log1p = np.asarray(X_ctrl_log1p.mean(axis=0)).flatten()
        ctrl_mu_counts = np.asarray(X_ctrl_counts.mean(axis=0)).flatten()
        ctrl_sq = X_ctrl_log1p.power(2).mean(axis=0)
        ctrl_sigma_log1p = np.sqrt(np.asarray(ctrl_sq).flatten() - ctrl_mu_log1p**2) + 1e-8
    else:
        ctrl_mu_log1p = X_ctrl_log1p.mean(axis=0).flatten()
        ctrl_mu_counts = X_ctrl_counts.mean(axis=0).flatten()
        ctrl_sigma_log1p = X_ctrl_log1p.std(axis=0).flatten() + 1e-8
    
    print(f"  Control mean cached: shape={ctrl_mu_log1p.shape}")
    
    processed_ctrl = ctrl_adata.copy()
    del ctrl_adata, X_ctrl_log1p, X_ctrl_counts
    gc.collect()
    
    # =========================================================================
    # Phase 3: Process perturbation cells in batches
    # =========================================================================
    print(f"\n[Phase 3] Processing perturbation cells in {n_batches} batches...")
    
    np.random.seed(42)
    np.random.shuffle(valid_pert_indices)
    batch_indices_list = np.array_split(valid_pert_indices, n_batches)
    
    processed_batches = []
    
    for batch_idx, batch_cell_indices in enumerate(batch_indices_list):
        print(f"\n  --- Batch {batch_idx + 1}/{n_batches} ({len(batch_cell_indices)} cells) ---")
        
        print(f"    Loading batch (backed mode)...")
        adata_backed = sc.read_h5ad(input_path, backed='r')
        
        batch_X_raw = adata_backed.X[batch_cell_indices][:, filtered_var_indices]
        if hasattr(batch_X_raw, 'toarray'):
            batch_X_raw = batch_X_raw.tocsr()
        
        batch_obs = obs_df.iloc[batch_cell_indices].copy()
        
        adata_backed.file.close()
        del adata_backed
        gc.collect()
        
        adata_batch = ad.AnnData(
            X=batch_X_raw.copy(),
            obs=batch_obs,
            var=filtered_var_df.copy(),
        )
        adata_batch.layers["counts"] = batch_X_raw.copy()
        del batch_X_raw
        gc.collect()
        
        print(f"    Normalizing...")
        sc.pp.normalize_total(adata_batch, target_sum=target_sum, key_added="ncounts")
        sc.pp.log1p(adata_batch)
        
        gene_to_idx = pd.Series(range(len(adata_batch.var_names)), index=adata_batch.var_names)
        pert_genes = adata_batch.obs[perturbation_col].astype(str)
        gene_indices = pert_genes.map(gene_to_idx).fillna(-1).astype(int).values
        valid_mask = gene_indices >= 0
        
        print(f"    LFC filtering...")
        if valid_mask.any():
            unique_gene_indices = np.unique(gene_indices[valid_mask])
            
            X_log1p = adata_batch.X
            X_counts = adata_batch.layers["counts"]
            
            X_log1p_subset = X_log1p[:, unique_gene_indices]
            X_counts_subset = X_counts[:, unique_gene_indices]
            
            ctrl_mu_log1p_subset = ctrl_mu_log1p[unique_gene_indices]
            ctrl_mu_counts_subset = ctrl_mu_counts[unique_gene_indices]
            
            row_indices = np.where(valid_mask)[0]
            subset_indices = np.searchsorted(unique_gene_indices, gene_indices[valid_mask])
            
            def extract_sparse_diag(sparse_mat, local_col_idx):
                if not hasattr(sparse_mat, 'toarray'):
                    return sparse_mat[np.arange(sparse_mat.shape[0]), local_col_idx]
                from scipy.sparse import csr_matrix as csr_mat
                n = sparse_mat.shape[0]
                selector = csr_mat(
                    (np.ones(n, dtype=np.float64), (np.arange(n), local_col_idx)),
                    shape=(n, sparse_mat.shape[1])
                )
                return np.asarray(sparse_mat.multiply(selector).sum(axis=1)).flatten()
            
            temp_log1p = X_log1p_subset[row_indices]
            temp_counts = X_counts_subset[row_indices]
            cell_log1p = extract_sparse_diag(temp_log1p, subset_indices)
            cell_counts = extract_sparse_diag(temp_counts, subset_indices)
            
            df = pd.DataFrame({
                'pert': pert_genes.values[valid_mask],
                'gene_subset_idx': subset_indices,
                'log1p': cell_log1p,
                'counts': cell_counts,
            })
            pert_stats_df = df.groupby('pert', observed=True).agg({
                'gene_subset_idx': 'first',
                'log1p': 'mean',
                'counts': 'mean',
            })
            
            gene_idx = pert_stats_df['gene_subset_idx'].values.astype(int)
            lfc_log1p = pert_stats_df['log1p'].values - ctrl_mu_log1p_subset[gene_idx]
            lfc_counts = pert_stats_df['counts'].values - ctrl_mu_counts_subset[gene_idx]
            
            invalid_lfc_mask = (lfc_counts >= 0) | (lfc_log1p >= 0)
            invalid_perts = pert_stats_df.index[invalid_lfc_mask].tolist()
            
            if invalid_perts:
                mask = ~adata_batch.obs[perturbation_col].isin(invalid_perts)
                adata_batch = adata_batch[mask].copy()
                print(f"    Removed {len(invalid_perts)} perturbations with LFC >= 0")
        
        if pert_efficiency_threshold > 0 and adata_batch.n_obs > 0:
            print(f"    Z-score efficiency filtering...")
            
            gene_to_idx = pd.Series(range(len(adata_batch.var_names)), index=adata_batch.var_names)
            pert_genes = adata_batch.obs[perturbation_col].astype(str)
            gene_indices = pert_genes.map(gene_to_idx).fillna(-1).astype(int).values
            valid_mask = gene_indices >= 0
            
            keep_mask = np.zeros(len(adata_batch), dtype=bool)
            
            if valid_mask.any():
                unique_gene_indices = np.unique(gene_indices[valid_mask])
                X = adata_batch.X
                X_subset = X[:, unique_gene_indices]
                
                ctrl_mu_subset = ctrl_mu_log1p[unique_gene_indices]
                ctrl_sigma_subset = ctrl_sigma_log1p[unique_gene_indices]
                
                valid_indices = gene_indices[valid_mask]
                row_indices = np.where(valid_mask)[0]
                subset_indices = np.searchsorted(unique_gene_indices, valid_indices)
                
                temp_X = X_subset[row_indices]
                x_int_values = extract_sparse_diag(temp_X, subset_indices)
                
                mu_values = ctrl_mu_subset[subset_indices]
                sigma_values = ctrl_sigma_subset[subset_indices]
                
                z_scores = (x_int_values - mu_values) / sigma_values
                efficiency_ok = z_scores <= -pert_efficiency_threshold
                keep_mask[valid_mask] = efficiency_ok
            
            n_removed = (~keep_mask).sum()
            adata_batch = adata_batch[keep_mask].copy()
            print(f"    Removed {n_removed} low-efficiency cells")
        
        if adata_batch.n_obs > 0:
            processed_batches.append(adata_batch)
            print(f"    Batch result: {adata_batch.n_obs} cells retained")
        else:
            print(f"    Batch result: 0 cells (all filtered)")
        
        del adata_batch
        gc.collect()
    
    # =========================================================================
    # Phase 4: Merge all batches with control group
    # =========================================================================
    print(f"\n[Phase 4] Merging {len(processed_batches)} batches with control group...")
    
    all_adatas = [processed_ctrl] + processed_batches
    adata_merged = sc.concat(all_adatas, join='inner')
    
    del processed_ctrl, processed_batches, all_adatas
    gc.collect()
    
    print(f"  Merged shape: {adata_merged.shape}")
    
    if 'gene_symbol' not in adata_merged.var.columns and 'gene_symbol' in filtered_var_df.columns:
        common = adata_merged.var_names.intersection(filtered_var_df.index)
        adata_merged.var.loc[common, 'gene_symbol'] = filtered_var_df.loc[common, 'gene_symbol']
        print(f"  Restored gene_symbol column ({len(common)} genes)")
    
    # =========================================================================
    # Phase 4.5: HVG selection (HVG + perturbation targets union)
    # =========================================================================
    print(f"\n[Phase 4.5] Selecting genes: HVG ({n_hvg}) + perturbation targets union...")
    gene_symbol_backup = adata_merged.var[['gene_symbol']].copy() if 'gene_symbol' in adata_merged.var.columns else None
    sc.pp.highly_variable_genes(adata_merged, n_top_genes=min(n_hvg, adata_merged.n_vars), flavor='seurat_v3', layer='counts')
    hvg_mask = adata_merged.var['highly_variable'].values.copy()
    n_hvg_selected = int(hvg_mask.sum())
    
    pert_genes_in_var = set()
    for g in adata_merged.obs[perturbation_col].unique():
        if g != control_label and g in adata_merged.var_names:
            pert_genes_in_var.add(g)
    pert_mask = np.array([g in pert_genes_in_var for g in adata_merged.var_names])
    
    union_mask = hvg_mask | pert_mask
    n_union = int(union_mask.sum())
    print(f"  HVG: {n_hvg_selected}, Perturbation genes in data: {len(pert_genes_in_var)}")
    print(f"  Union (deduplicated): {n_union}")
    
    adata_merged = adata_merged[:, union_mask].copy()
    if gene_symbol_backup is not None:
        common = adata_merged.var_names.intersection(gene_symbol_backup.index)
        adata_merged.var.loc[common, 'gene_symbol'] = gene_symbol_backup.loc[common, 'gene_symbol']
    print(f"  Genes: {len(union_mask)} -> {adata_merged.n_vars}")
    
    # =========================================================================
    # Phase 5: Rare perturbation filtering + final save
    # =========================================================================
    if min_cells_per_pert > 0:
        print(f"\n[Phase 5] Filtering rare perturbations (cells < {min_cells_per_pert})...")
        pert_counts = adata_merged.obs[perturbation_col].value_counts()
        rare_perts = pert_counts[
            (pert_counts < min_cells_per_pert) & 
            (pert_counts.index != control_label)
        ].index.tolist()
        
        if rare_perts:
            mask = ~adata_merged.obs[perturbation_col].isin(rare_perts)
            adata_merged = adata_merged[mask].copy()
            print(f"  Removed {len(rare_perts)} rare perturbations")
    
    print("\n[Phase 6] Final gene filtering...")
    n_genes_before = adata_merged.n_vars
    sc.pp.filter_genes(adata_merged, min_cells=1)
    print(f"  Genes: {n_genes_before} -> {adata_merged.n_vars}")
    
    stats["n_cells_output"] = adata_merged.n_obs
    stats["n_genes_output"] = adata_merged.n_vars
    stats["n_perturbations_output"] = adata_merged.obs[perturbation_col].nunique()
    stats["n_control_cells"] = (adata_merged.obs[perturbation_col] == control_label).sum()
    
    print(f"\n[Phase 7] Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_merged.write(output_path, compression="gzip")
    
    gene_list_path = output_path.parent / f"gene_list_{output_path.stem}.csv"
    print(f"\n[Phase 7b] Saving gene list to {gene_list_path}...")
    gene_df = pd.DataFrame({
        "gene_id": adata_merged.var_names.tolist(),
        "gene_name": adata_merged.var["gene_symbol"].tolist() if "gene_symbol" in adata_merged.var.columns else [""] * adata_merged.n_vars,
    })
    gene_df.to_csv(gene_list_path, index=False)
    print(f"  {len(gene_df)} genes exported")
    
    print(f"\n{'='*60}")
    print(f"Summary (BATCHED):")
    print(f"  Output shape: {adata_merged.shape}")
    print(f"  Perturbations: {stats['n_perturbations_output']}")
    print(f"  Control cells: {stats['n_control_cells']}")
    print(f"{'='*60}")
    
    return stats


def preprocess_all(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    skip_dirs: set = SKIP_DIRS,
    skip_existing: bool = False,
    n_hvg: int = N_HVG,
    pert_efficiency_threshold: float = PERT_EFFICIENCY_THRESHOLD,
    min_cells_per_pert: int = MIN_CELLS_PER_PERT,
) -> pd.DataFrame:
    """Preprocess all datasets."""
    print(f"Gene selection: top-{n_hvg} HVG + perturbation targets union")
    
    all_stats = []
    
    for dataset in sorted(os.listdir(input_dir)):
        dataset_path = input_dir / dataset
        if not dataset_path.is_dir() or dataset in skip_dirs:
            continue
        
        for fname in sorted(os.listdir(dataset_path)):
            if not fname.endswith(".h5ad"):
                continue
            
            input_path = dataset_path / fname
            output_path = output_dir / dataset / fname
            
            if skip_existing and output_path.exists():
                print(f"\nSkipping (already exists): {output_path}")
                continue
            
            try:
                stats = preprocess_single_file(
                    input_path=input_path,
                    output_path=output_path,
                    n_hvg=n_hvg,
                    pert_efficiency_threshold=pert_efficiency_threshold,
                    min_cells_per_pert=min_cells_per_pert,
                )
                stats["status"] = "success"
            except Exception as e:
                print(f"\nERROR processing {input_path}: {e}")
                stats = {"input_path": str(input_path), "status": f"error: {e}"}
            
            all_stats.append(stats)
    
    stats_df = pd.DataFrame(all_stats)
    generate_coverage_report(output_dir)
    
    return stats_df


def generate_coverage_report(output_dir: Path) -> pd.DataFrame:
    """Generate perturbation gene coverage report."""
    all_perturbations = set()
    file_info = []
    
    for dataset_dir in sorted(output_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for fpath in sorted(dataset_dir.glob("*.h5ad")):
            adata = sc.read_h5ad(fpath, backed='r')
            perts = set(adata.obs[PERTURBATION_COL].unique()) - {CONTROL_LABEL}
            n_genes = adata.n_vars
            n_cells = adata.n_obs
            adata.file.close()
            all_perturbations.update(perts)
            file_info.append({
                "dataset": dataset_dir.name,
                "celltype": fpath.stem,
                "n_cells": n_cells,
                "n_genes": n_genes,
                "n_perturbations": len(perts),
            })
    
    df = pd.DataFrame(file_info)
    
    print("\n" + "="*80)
    print("Perturbation Coverage Report")
    print("="*80)
    print(df.to_string(index=False))
    print(f"\nTotal unique perturbations across all files: {len(all_perturbations)}")
    print("="*80)
    
    report_path = output_dir / "coverage_report.csv"
    df.to_csv(report_path, index=False)
    print(f"\nReport saved to: {report_path}")
    
    return df


# ============================================================================
# Verification
# ============================================================================

def verify_preprocessed_file(path: Path) -> dict:
    """Verify a preprocessed file."""
    adata = sc.read_h5ad(path)
    
    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        x_min = float(X.min())
        x_max = float(X.max())
    else:
        x_min = float(X.min())
        x_max = float(X.max())
    
    results = {
        "path": str(path),
        "shape": adata.shape,
        "has_counts_layer": "counts" in adata.layers,
        "has_ncounts": "ncounts" in adata.obs.columns,
        "X_is_log1p": bool(x_min >= 0 and x_max < 20),
        "counts_are_integers": False,
        "ncounts_positive": False,
        "control_exists": False,
    }
    
    if results["has_counts_layer"]:
        counts = adata.layers["counts"]
        import scipy.sparse as sp
        if sp.issparse(counts):
            data = counts.data
            results["counts_are_integers"] = bool(np.allclose(data, np.round(data)))
        else:
            results["counts_are_integers"] = bool(np.allclose(counts, np.round(counts)))
    
    if results["has_ncounts"]:
        results["ncounts_positive"] = bool((adata.obs["ncounts"] > 0).all())
        results["ncounts_range"] = f"{adata.obs['ncounts'].min():.4f} - {adata.obs['ncounts'].max():.4f}"
    
    results["control_exists"] = CONTROL_LABEL in adata.obs[PERTURBATION_COL].values
    
    # Verify reconstruction: expm1(X_log1p) * ncounts ~= raw_counts
    if results["has_counts_layer"] and results["has_ncounts"]:
        import scipy.sparse as sp
        raw_row = adata.layers["counts"][0]
        log1p_row = adata.X[0]
        if sp.issparse(raw_row):
            raw_row = raw_row.toarray()
        if sp.issparse(log1p_row):
            log1p_row = log1p_row.toarray()
        raw = np.array(raw_row).flatten()
        log1p = np.array(log1p_row).flatten()
        ncounts = adata.obs["ncounts"].iloc[0]
        reconstructed = np.expm1(log1p) * ncounts
        
        nonzero_idx = np.where(raw > 0)[0]
        if len(nonzero_idx) > 0:
            idx = nonzero_idx[0]
            results["reconstruction_check"] = f"raw={raw[idx]:.0f}, reconstructed={reconstructed[idx]:.1f}"
            results["reconstruction_ok"] = bool(np.abs(raw[idx] - reconstructed[idx]) < 1.0)
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess single-cell perturbation data")
    parser.add_argument(
        "--input-dir", 
        type=Path, 
        default=INPUT_DIR,
        help="Input directory"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=OUTPUT_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--single-file",
        type=Path,
        default=None,
        help="Process a single file instead of all"
    )
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="Verify a preprocessed file"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in output directory"
    )
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=N_HVG,
        help=f"Number of highly variable genes to select (default: {N_HVG}). Perturbation target genes are always included."
    )
    parser.add_argument(
        "--pert-efficiency-threshold",
        type=float,
        default=PERT_EFFICIENCY_THRESHOLD,
        help=f"Perturbation efficiency Z-score threshold (default: {PERT_EFFICIENCY_THRESHOLD}). Set to 0 to skip."
    )
    parser.add_argument(
        "--min-cells-per-pert",
        type=int,
        default=MIN_CELLS_PER_PERT,
        help=f"Minimum cells per perturbation (default: {MIN_CELLS_PER_PERT}). Perturbations with fewer cells will be removed. Set to 0 to skip."
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Use batched mode for large files (avoids OOM). Requires --single-file."
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=3,
        help="Number of batches for batched mode (default: 3)"
    )
    parser.add_argument(
        "--filter-noncoding",
        action="store_true",
        help="Filter out non-coding RNA genes (lncRNA etc.) before HVG selection. Improves prior graph density for datasets with many lncRNAs."
    )
    
    args = parser.parse_args()
    
    if args.verify:
        results = verify_preprocessed_file(args.verify)
        print("\nVerification Results:")
        for k, v in results.items():
            status = "OK" if v not in [False, None] else "FAIL"
            print(f"  {status} {k}: {v}")
        return
    
    if args.single_file:
        output_path = args.output_dir / args.single_file.parent.name / args.single_file.name
        
        if args.batched:
            print(f"Using BATCHED mode with {args.n_batches} batches...")
            preprocess_large_file_batched(
                input_path=args.single_file,
                output_path=output_path,
                n_batches=args.n_batches,
                n_hvg=args.n_hvg,
                pert_efficiency_threshold=args.pert_efficiency_threshold,
                min_cells_per_pert=args.min_cells_per_pert,
            )
        else:
            preprocess_single_file(
                input_path=args.single_file,
                output_path=output_path,
                n_hvg=args.n_hvg,
                pert_efficiency_threshold=args.pert_efficiency_threshold,
                min_cells_per_pert=args.min_cells_per_pert,
                filter_noncoding=args.filter_noncoding,
            )
    else:
        stats_df = preprocess_all(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
            n_hvg=args.n_hvg,
            pert_efficiency_threshold=args.pert_efficiency_threshold,
            min_cells_per_pert=args.min_cells_per_pert,
        )
        
        print("\n" + "="*60)
        print("All Processing Complete")
        print("="*60)
        print(stats_df.to_string())


if __name__ == "__main__":
    main()
