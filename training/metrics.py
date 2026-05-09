"""Evaluation metrics for CRISP."""

import numpy as np
import torch
from typing import Optional


def _compute_idcg(num_correct: int, num_nodes: int) -> float:
    """Ideal DCG assuming correct genes ranked first."""
    idcg = 0.0
    for rank in range(1, num_correct + 1):
        gain = 1 - rank / num_nodes
        discount = 1 / np.log2(rank + 1)
        idcg += gain * discount
    return idcg


class PerturbationMetrics:
    """Reverse perturbation prediction ranking metrics collector."""

    RECALL_K_VALUES = [1, 2, 10, 100]

    def __init__(self):
        self.results = []

    def update(
        self,
        pred_scores: torch.Tensor,
        true_label: torch.Tensor,
        pert_name: str = "",
        forward_consistency: Optional[float] = None,
        deg_indices: Optional[np.ndarray] = None,
        lfc_corr: Optional[dict] = None,
    ):
        scores = pred_scores.detach().cpu().numpy() if hasattr(pred_scores, 'detach') else np.asarray(pred_scores)
        true = true_label.detach().cpu().numpy().astype(bool) if hasattr(true_label, 'detach') else np.asarray(true_label, dtype=bool)

        num_nodes = len(scores)
        correct_set = set(np.where(true)[0].tolist())
        num_correct = len(correct_set)

        if num_correct == 0:
            
            entry = {
                "pert_name": pert_name,
                "true_gene_ranks": [],
                "ndcg": 0.0,
                "mean_rank_score": 0.0,
                "jaccard": 0.0,
                "auroc": 0.0,
                "auprc": 0.0,
                "n_correct": 0,
            }
            for k in self.RECALL_K_VALUES:
                entry[f"recall@{k}"] = 0.0
            entry["exact_match"] = 0.0
            for k in self.RECALL_K_VALUES:
                entry[f"all_in_top@{k}"] = 0.0
            if forward_consistency is not None:
                entry["forward_consistency"] = forward_consistency
            self.results.append(entry)
            return

        
        ranked_genes = np.argsort(scores)[::-1].tolist()

        # --- Recall@K ---
        recall_at_k = {}
        for k in self.RECALL_K_VALUES:
            top_k = set(ranked_genes[:k])
            recall_at_k[k] = len(top_k & correct_set) / num_correct

        # --- True gene ranks (for recall@k curve at arbitrary k) ---
        true_gene_ranks = []
        for ci in sorted(correct_set):
            true_gene_ranks.append(ranked_genes.index(ci) + 1)  # 1-based

        # --- nDCG ---
        dcg = 0.0
        for ci in correct_set:
            rank = ranked_genes.index(ci) + 1  # 1-based
            gain = 1 - rank / num_nodes
            discount = 1 / np.log2(rank + 1)
            dcg += gain * discount
        idcg = _compute_idcg(num_correct, num_nodes)
        ndcg = dcg / idcg if idcg > 0 else 0.0

        # --- Mean Ranking Score ---
        ranking_scores = []
        for ci in correct_set:
            rank_0based = ranked_genes.index(ci)
            ranking_scores.append(1 - rank_0based / num_nodes)
        mean_rank_score = float(np.mean(ranking_scores))

        # --- Jaccard (pred top-|correct| vs correct) ---
        pred_top = set(ranked_genes[:num_correct])
        union = pred_top | correct_set
        jaccard = len(pred_top & correct_set) / len(union) if union else 0.0

        # --- Exact Match (top-|correct| == correct_set) ---
        exact_match = 1.0 if pred_top == correct_set else 0.0

        # --- All-in-Top-K (all correct genes within top K) ---
        all_in_top_k = {}
        for k in self.RECALL_K_VALUES:
            top_k = set(ranked_genes[:k])
            all_in_top_k[k] = 1.0 if correct_set <= top_k else 0.0

        # --- AUROC & AUPRC ---
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            auroc = float(roc_auc_score(true.astype(int), scores))
        except ValueError:
            auroc = 0.0
        try:
            auprc = float(average_precision_score(true.astype(int), scores))
        except ValueError:
            auprc = 0.0

        entry = {
            "pert_name": pert_name,
            "true_gene_ranks": true_gene_ranks,
            "ndcg": ndcg,
            "mean_rank_score": mean_rank_score,
            "jaccard": jaccard,
            "exact_match": exact_match,
            "auroc": auroc,
            "auprc": auprc,
            "n_correct": num_correct,
        }
        for k in self.RECALL_K_VALUES:
            entry[f"all_in_top@{k}"] = all_in_top_k[k]
        for k in self.RECALL_K_VALUES:
            entry[f"recall@{k}"] = recall_at_k[k]
        if forward_consistency is not None:
            entry["forward_consistency"] = forward_consistency

        # --- DEG-restricted metrics (CausalDN evaluation protocol) ---
        if deg_indices is not None and len(deg_indices) > 0:
            d_scores = scores[deg_indices]
            d_true = true[deg_indices]
            d_num = int(d_true.sum())
            if d_num > 0 and d_num < len(d_true):
                d_ranked = np.argsort(d_scores)[::-1].tolist()
                d_correct = set(np.where(d_true)[0].tolist())
                # Rank (CausalDN primary metric)
                d_ranks = []
                for ci in d_correct:
                    d_ranks.append(1.0 - d_ranked.index(ci) / len(d_ranked))
                entry["deg_rank"] = float(np.mean(d_ranks))
                # Recall@K on restricted pool
                for k in self.RECALL_K_VALUES:
                    top_k = set(d_ranked[:k])
                    entry[f"deg_recall@{k}"] = len(top_k & d_correct) / d_num
                # AUROC / AUPRC on restricted pool
                try:
                    entry["deg_auroc"] = float(roc_auc_score(d_true.astype(int), d_scores))
                except ValueError:
                    entry["deg_auroc"] = 0.0
                try:
                    entry["deg_auprc"] = float(average_precision_score(d_true.astype(int), d_scores))
                except ValueError:
                    entry["deg_auprc"] = 0.0
                entry["deg_n_candidates"] = len(deg_indices)

        # LFC-based Spearman/Pearson (CausalDN primary metric)
        if lfc_corr is not None:
            entry["lfc_spearman"] = lfc_corr["spearman"]
            entry["lfc_pearson"] = lfc_corr["pearson"]

        self.results.append(entry)

    def compute(self) -> dict:
        """Compute aggregate metrics."""
        if not self.results:
            return {}

        metrics: dict = {"n_samples": len(self.results)}

        for key in ["ndcg", "mean_rank_score", "jaccard", "auroc", "auprc"]:
            values = [r[key] for r in self.results]
            metrics[f"mean_{key}"] = float(np.mean(values))
            metrics[f"std_{key}"] = float(np.std(values))

        for k in self.RECALL_K_VALUES:
            values = [r[f"recall@{k}"] for r in self.results]
            metrics[f"mean_recall@{k}"] = float(np.mean(values))
            metrics[f"std_recall@{k}"] = float(np.std(values))

        # Exact Match %
        em_values = [r["exact_match"] for r in self.results]
        metrics["exact_match_pct"] = 100.0 * float(np.mean(em_values))

        # All-in-Top-K %
        for k in self.RECALL_K_VALUES:
            ait_values = [r[f"all_in_top@{k}"] for r in self.results]
            metrics[f"all_in_top@{k}_pct"] = 100.0 * float(np.mean(ait_values))

        # Partially Correct %
        n_partial = sum(1 for r in self.results if r["jaccard"] > 0)
        metrics["partially_correct_pct"] = 100.0 * n_partial / len(self.results)

        
        fc_values = [r["forward_consistency"] for r in self.results
                     if "forward_consistency" in r]
        if fc_values:
            metrics["mean_forward_consistency"] = float(np.mean(fc_values))
            metrics["std_forward_consistency"] = float(np.std(fc_values))

        # DEG-restricted metrics (CausalDN protocol)
        deg_keys = ["deg_rank", "deg_auroc", "deg_auprc"]
        for key in deg_keys:
            values = [r[key] for r in self.results if key in r]
            if values:
                metrics[f"mean_{key}"] = float(np.mean(values))
                metrics[f"std_{key}"] = float(np.std(values))
        for k in self.RECALL_K_VALUES:
            values = [r[f"deg_recall@{k}"] for r in self.results
                      if f"deg_recall@{k}" in r]
            if values:
                metrics[f"mean_deg_recall@{k}"] = float(np.mean(values))
                metrics[f"std_deg_recall@{k}"] = float(np.std(values))
        cand_values = [r["deg_n_candidates"] for r in self.results
                       if "deg_n_candidates" in r]
        if cand_values:
            metrics["mean_deg_n_candidates"] = float(np.mean(cand_values))

        # LFC-based Spearman/Pearson (CausalDN primary)
        for key in ["lfc_spearman", "lfc_pearson"]:
            values = [r[key] for r in self.results if key in r]
            if values:
                metrics[f"mean_{key}"] = float(np.mean(values))
                metrics[f"std_{key}"] = float(np.std(values))
                metrics[f"n_{key}"] = len(values)

        return metrics

    def print_summary(self, summary: Optional[dict] = None):
        """Print evaluation summary."""
        if summary is None:
            summary = self.compute()

        print("\n" + "=" * 55)
        print("Reverse Perturbation Prediction - Evaluation Summary")
        print("=" * 55)
        print(f"  Samples:              {summary.get('n_samples', 0)}")
        for k in self.RECALL_K_VALUES:
            m = summary.get(f'mean_recall@{k}', 0)
            s = summary.get(f'std_recall@{k}', 0)
            print(f"  Recall@{k:<4d}           {m:.4f} +/- {s:.4f}")
        print(f"  nDCG:                 {summary.get('mean_ndcg', 0):.4f} +/- {summary.get('std_ndcg', 0):.4f}")
        print(f"  Mean Ranking Score:   {summary.get('mean_mean_rank_score', 0):.4f} +/- {summary.get('std_mean_rank_score', 0):.4f}")
        print(f"  AUROC:                {summary.get('mean_auroc', 0):.4f} +/- {summary.get('std_auroc', 0):.4f}")
        print(f"  AUPRC:                {summary.get('mean_auprc', 0):.4f} +/- {summary.get('std_auprc', 0):.4f}")
        print(f"  Jaccard:              {summary.get('mean_jaccard', 0):.4f} +/- {summary.get('std_jaccard', 0):.4f}")
        print(f"  Exact Match:          {summary.get('exact_match_pct', 0):.1f}%")
        print(f"  Partially Correct:    {summary.get('partially_correct_pct', 0):.1f}%")
        # All-in-Top-K (meaningful for multi-gene perturbations)
        has_multi = any(r.get("n_correct", 1) > 1 for r in self.results)
        if has_multi:
            for k in self.RECALL_K_VALUES:
                v = summary.get(f'all_in_top@{k}_pct', 0)
                print(f"  All-in-Top-{k:<4d}       {v:.1f}%")
        if "mean_forward_consistency" in summary:
            print(f"  Forward Consistency:  {summary.get('mean_forward_consistency', 0):.4f} +/- {summary.get('std_forward_consistency', 0):.4f}")
        # DEG-restricted (CausalDN protocol)
        if "mean_deg_rank" in summary:
            print("-" * 55)
            print(f"  [DEG-1000] Candidates:  {summary.get('mean_deg_n_candidates', 0):.0f}")
            print(f"  [DEG-1000] Rank:        {summary.get('mean_deg_rank', 0):.4f} +/- {summary.get('std_deg_rank', 0):.4f}")
            for k in self.RECALL_K_VALUES:
                m = summary.get(f'mean_deg_recall@{k}', 0)
                s = summary.get(f'std_deg_recall@{k}', 0)
                print(f"  [DEG-1000] Recall@{k:<4d}  {m:.4f} +/- {s:.4f}")
            print(f"  [DEG-1000] AUROC:       {summary.get('mean_deg_auroc', 0):.4f} +/- {summary.get('std_deg_auroc', 0):.4f}")
            print(f"  [DEG-1000] AUPRC:       {summary.get('mean_deg_auprc', 0):.4f} +/- {summary.get('std_deg_auprc', 0):.4f}")
        if "mean_lfc_spearman" in summary:
            n_lfc = summary.get('n_lfc_spearman', 0)
            print(f"  [LFC] Spearman rho:       {summary.get('mean_lfc_spearman', 0):.4f} +/- {summary.get('std_lfc_spearman', 0):.4f}  (n={n_lfc})")
            print(f"  [LFC] Pearson r:        {summary.get('mean_lfc_pearson', 0):.4f} +/- {summary.get('std_lfc_pearson', 0):.4f}")
        print("=" * 55)


def compute_reverse_metrics(test_dataset, pred_dict: dict) -> dict:
    """Convenience wrapper: compute reverse metrics from pred_dict.

    Args:
        test_dataset: PerturbationDataset (test split, with deg_indices if available)
        pred_dict: {pert_name: scores_array} mapping

    Returns:
        Flat dict with keys: recall_at_1, ndcg, auprc, auroc, deg_recall_at_1, etc.
    """
    pm = PerturbationMetrics()
    for sample in test_dataset.samples:
        name = sample["pert_name"]
        if name not in pred_dict:
            continue
        scores = pred_dict[name]
        label = sample["pert_label"]
        deg_indices = sample.get("deg_indices", None)
        pm.update(
            pred_scores=torch.from_numpy(np.asarray(scores, dtype=np.float32)),
            true_label=torch.from_numpy(np.asarray(label, dtype=np.float32)),
            pert_name=name,
            deg_indices=deg_indices,
        )
    summary = pm.compute()
    result = {
        "recall_at_1": summary.get("mean_recall@1", 0.0),
        "recall_at_2": summary.get("mean_recall@2", 0.0),
        "recall_at_10": summary.get("mean_recall@10", 0.0),
        "recall_at_100": summary.get("mean_recall@100", 0.0),
        "ndcg": summary.get("mean_ndcg", 0.0),
        "auprc": summary.get("mean_auprc", 0.0),
        "auroc": summary.get("mean_auroc", 0.0),
        "exact_match_pct": summary.get("exact_match_pct", 0.0),
        "deg_recall_at_1": summary.get("mean_deg_recall@1", 0.0),
        "deg_recall_at_2": summary.get("mean_deg_recall@2", 0.0),
        "deg_rank": summary.get("mean_deg_rank", 0.0),
    }
    for k in PerturbationMetrics.RECALL_K_VALUES:
        key = f"all_in_top@{k}_pct"
        if key in summary:
            result[key] = summary[key]
    return result


class ForwardMetrics:
    """Forward prediction metrics collector (Stage 2: Neural SCM)."""

    def __init__(self):
        self.results = []

    def update(
        self,
        pred_expr: np.ndarray,
        true_expr: np.ndarray,
        ctrl_expr: np.ndarray,
        pert_name: str = "",
        de_idx: Optional[np.ndarray] = None,
    ):
        from scipy.stats import pearsonr

        
        r_all, _ = pearsonr(pred_expr, true_expr)
        if not np.isfinite(r_all):
            r_all = 0.0

        
        mse_all = float(np.mean((pred_expr - true_expr) ** 2))

        entry = {
            "pert_name": pert_name,
            "pearson_all": r_all,
            "mse_all": mse_all,
        }

        if de_idx is not None and len(de_idx) > 0:
            pred_de = pred_expr[de_idx]
            true_de = true_expr[de_idx]
            ctrl_de = ctrl_expr[de_idx]

            # Pearson on Top-20 DEG (delta from control)
            pred_delta = pred_de - ctrl_de
            true_delta = true_de - ctrl_de
            r_de, _ = pearsonr(pred_delta, true_delta)
            entry["pearson_delta_top20"] = r_de if np.isfinite(r_de) else 0.0

            # Direction Accuracy on Top-20 DEGs
            pred_sign = np.sign(pred_delta)
            true_sign = np.sign(true_delta)
            nonzero_mask = true_sign != 0
            if nonzero_mask.any():
                entry["direction_accuracy_top20"] = float(
                    np.mean(pred_sign[nonzero_mask] == true_sign[nonzero_mask])
                )
            else:
                entry["direction_accuracy_top20"] = 0.0

        self.results.append(entry)

    def compute(self) -> dict:
        """Compute aggregate metrics."""
        if not self.results:
            return {}

        metrics: dict = {"n_samples": len(self.results)}

        for key in ["pearson_all", "mse_all"]:
            values = [r[key] for r in self.results]
            metrics[f"mean_{key}"] = float(np.mean(values))
            metrics[f"std_{key}"] = float(np.std(values))

        
        for key in ["pearson_delta_top20", "direction_accuracy_top20"]:
            values = [r[key] for r in self.results if key in r]
            if values:
                metrics[f"mean_{key}"] = float(np.mean(values))
                metrics[f"std_{key}"] = float(np.std(values))

        return metrics

    def print_summary(self, summary: Optional[dict] = None):
        """Print evaluation summary."""
        if summary is None:
            summary = self.compute()

        print("\n" + "=" * 55)
        print("Forward Prediction (Stage 2) - Evaluation Summary")
        print("=" * 55)
        print(f"  Samples:              {summary.get('n_samples', 0)}")
        print(f"  Pearson (all):        {summary.get('mean_pearson_all', 0):.4f} +/- {summary.get('std_pearson_all', 0):.4f}")
        print(f"  MSE (all):            {summary.get('mean_mse_all', 0):.6f} +/- {summary.get('std_mse_all', 0):.6f}")
        if "mean_pearson_delta_top20" in summary:
            print(f"  Pearson (DEG delta):      {summary.get('mean_pearson_delta_top20', 0):.4f} +/- {summary.get('std_pearson_delta_top20', 0):.4f}")
        if "mean_direction_accuracy_top20" in summary:
            print(f"  Direction Acc (top20): {summary.get('mean_direction_accuracy_top20', 0):.4f} +/- {summary.get('std_direction_accuracy_top20', 0):.4f}")
        print("=" * 55)
