"""RIVER Trainer: three-stage training pipeline with checkpointing."""

from pathlib import Path
from typing import Optional
import csv

import numpy as np
import torch
import yaml

from model.cpd import CPDModel
from data.dataset import PerturbationDataset
from data.graph_builder import load_prior_graph
from .metrics import PerturbationMetrics


class CPDTrainer:
    """Three-stage training pipeline with checkpoint resume."""

    CKPT_INTERVAL = 50

    def __init__(
        self,
        data_path: Path,
        output_dir: Path,
        device: str = "cuda",
        seed: int = 42,
        resume: bool = False,
        
        cg_n_epochs: int = 300,
        cg_lr: float = 3e-3,
        cg_hidden_dim: int = 64,
        cg_lambda_l1: float = 0.02,
        cg_lambda_l1_no_prior: float = 0.2,
        cg_prior_mask_mode: str = "adaptive",
        
        scm_n_epochs: int = 200,
        scm_lr: float = 1e-3,
        scm_hidden_dim: int = 64,
        scm_batch_size: int = 256,
        
        diff_threshold: float = 0.5,
        max_interventions: int = 10,
        score_mode: str = "raw",
        
        adj_threshold: float = 0.3,
        
        lambda_pert: float = 0.0,
        
        lambda_pert_scm: float = 0.0,
        
        scheduled_sampling_start: float = -1.0,
        
        prior_graph_path: Optional[Path] = None,
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.device = device
        self.seed = seed
        self.resume = resume

        
        self.cg_n_epochs = cg_n_epochs
        self.cg_lr = cg_lr
        self.cg_hidden_dim = cg_hidden_dim
        self.cg_lambda_l1 = cg_lambda_l1
        self.cg_lambda_l1_no_prior = cg_lambda_l1_no_prior
        self.cg_prior_mask_mode = cg_prior_mask_mode
        self.scm_n_epochs = scm_n_epochs
        self.scm_lr = scm_lr
        self.scm_hidden_dim = scm_hidden_dim
        self.scm_batch_size = scm_batch_size
        self.diff_threshold = diff_threshold
        self.max_interventions = max_interventions
        self.score_mode = score_mode
        self.adj_threshold = adj_threshold
        self.lambda_pert = lambda_pert
        self.lambda_pert_scm = lambda_pert_scm
        self.scheduled_sampling_start = scheduled_sampling_start
        self.prior_graph_path = prior_graph_path

        
        self.history: list[dict] = []

        
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self):
        """Run full training pipeline."""
        print("=" * 60)
        print("CPD Training Pipeline")
        print("=" * 60)

        
        print("\n[Data] Loading training dataset...")
        train_dataset = PerturbationDataset(self.data_path, split="train", split_seed=self.seed)

        gene_names = train_dataset.get_gene_names()
        n_genes = len(gene_names)
        print(f"[Data] {n_genes} genes, train={len(train_dataset)} perturbations")

        prior_adj = self._build_prior_graph(gene_names)
        obs_data = torch.from_numpy(train_dataset.ctrl_expr).float()
        print(f"[Data] Control cells: {obs_data.shape[0]}")

        
        self._dump_config()

        
        start_stage = 1
        s1_start_epoch = 0
        s2_start_epoch = 0
        s1_optim_state = None
        s2_optim_state = None
        mu_start = None
        cpd = None

        if self.resume:
            ckpt = self._load_checkpoint()
            if ckpt is not None:
                self.history = ckpt.get("training_history", [])
                completed = ckpt.get("completed_stage", 0)
                if completed >= 2:
                    print("[Resume] Training complete, proceeding to evaluation")
                    start_stage = 3
                elif completed >= 1:
                    start_stage = 2
                    if ckpt.get("current_stage", 1) >= 2:
                        
                        s2_start_epoch = ckpt.get("current_epoch", 0) + 1
                        if s2_start_epoch >= self.scm_n_epochs:
                            start_stage = 3
                        else:
                            s2_optim_state = ckpt.get("optimizer_state")
                    
                else:
                    start_stage = 1
                    s1_start_epoch = ckpt.get("current_epoch", 0) + 1
                    s1_optim_state = ckpt.get("optimizer_state")
                    mu_start = ckpt.get("mu")

                
                cpd = CPDModel(
                    gene_names=gene_names,
                    prior_adj=prior_adj,
                    device=self.device,
                    cg_hidden_dim=self.cg_hidden_dim,
                    cg_lambda_l1=self.cg_lambda_l1,
                    cg_lambda_l1_no_prior=self.cg_lambda_l1_no_prior,
                    cg_prior_mask_mode=self.cg_prior_mask_mode,
                    adj_threshold=self.adj_threshold,
                    scm_hidden_dim=self.scm_hidden_dim,
                    diff_threshold=self.diff_threshold,
                    max_interventions=self.max_interventions,
                    score_mode=self.score_mode,
                )
                self._restore_model_state(cpd, ckpt)
                print(f"[Resume] Resuming from stage {start_stage}")

        
        if cpd is None:
            cpd = CPDModel(
                gene_names=gene_names,
                prior_adj=prior_adj,
                device=self.device,
                cg_hidden_dim=self.cg_hidden_dim,
                cg_lambda_l1=self.cg_lambda_l1,
                cg_lambda_l1_no_prior=self.cg_lambda_l1_no_prior,
                cg_prior_mask_mode=self.cg_prior_mask_mode,
                adj_threshold=self.adj_threshold,
                scm_hidden_dim=self.scm_hidden_dim,
                diff_threshold=self.diff_threshold,
                max_interventions=self.max_interventions,
                score_mode=self.score_mode,
            )

        # ------ Stage 1 ------
        if start_stage <= 1:
            print("\n" + "=" * 60)
            print("Stage 1: Causal Graph Learning")
            print("=" * 60)

            
            pert_data_t = None
            pert_labels_t = None
            ctrl_mean_t = None
            if self.lambda_pert > 0 and len(train_dataset.samples) > 0:
                pert_data_t = torch.tensor(
                    np.array([s["pert_expr"] for s in train_dataset.samples]),
                    dtype=torch.float32,
                )
                pert_labels_t = torch.tensor(
                    np.array([s["pert_label"] for s in train_dataset.samples]),
                    dtype=torch.float32,
                )
                ctrl_mean_t = torch.tensor(
                    np.array([s["ctrl_mean"] for s in train_dataset.samples]),
                    dtype=torch.float32,
                )
                print(f"[Data] Interventional data: {pert_data_t.shape[0]} perturbations, "
                      f"lambda_pert={self.lambda_pert}")

            cpd.learn_causal_graph(
                obs_data,
                n_epochs=self.cg_n_epochs,
                lr=self.cg_lr,
                start_epoch=s1_start_epoch,
                optimizer_state=s1_optim_state,
                mu_start=mu_start,
                pert_data=pert_data_t,
                pert_labels=pert_labels_t,
                ctrl_mean=ctrl_mean_t,
                lambda_pert=self.lambda_pert,
                epoch_callback=lambda e, r, opt, mu: self._epoch_callback(
                    cpd, stage=1, epoch=e, record=r, optimizer=opt,
                    n_epochs=self.cg_n_epochs, mu=mu,
                ),
            )
            
            self._save_checkpoint(
                cpd, completed_stage=1, current_stage=1,
                current_epoch=self.cg_n_epochs - 1,
                optimizer=None, mu=None,
            )

        # ------ Stage 2 ------
        if start_stage <= 2:
            print("\n" + "=" * 60)
            print("Stage 2: Neural SCM Training")
            print("=" * 60)

            
            s2_pert_data = None
            s2_pert_labels = None
            s2_ctrl_mean = None
            if self.lambda_pert_scm > 0 and len(train_dataset.samples) > 0:
                s2_pert_data = torch.tensor(
                    np.array([s["pert_expr"] for s in train_dataset.samples]),
                    dtype=torch.float32,
                )
                s2_pert_labels = torch.tensor(
                    np.array([s["pert_label"] for s in train_dataset.samples]),
                    dtype=torch.float32,
                )
                s2_ctrl_mean = torch.tensor(
                    np.array([s["ctrl_mean"] for s in train_dataset.samples]),
                    dtype=torch.float32,
                )
                print(f"[Data] Interventional SCM data: {s2_pert_data.shape[0]} perturbations, "
                      f"lambda_pert_scm={self.lambda_pert_scm}")

            cpd.train_neural_scm(
                obs_data,
                n_epochs=self.scm_n_epochs,
                lr=self.scm_lr,
                batch_size=self.scm_batch_size,
                start_epoch=s2_start_epoch,
                optimizer_state=s2_optim_state,
                epoch_callback=lambda e, r, opt, _mu: self._epoch_callback(
                    cpd, stage=2, epoch=e, record=r, optimizer=opt,
                    n_epochs=self.scm_n_epochs,
                ),
                pert_data=s2_pert_data,
                pert_labels=s2_pert_labels,
                ctrl_mean=s2_ctrl_mean,
                lambda_pert_scm=self.lambda_pert_scm,
                scheduled_sampling_start=self.scheduled_sampling_start,
            )
            self._save_checkpoint(
                cpd, completed_stage=2, current_stage=2,
                current_epoch=self.scm_n_epochs - 1,
                optimizer=None,
            )

        
        cpd.save(self.ckpt_dir)

        
        print("\n" + "=" * 60)
        print("Stage 3: Evaluation (GCPS Inference)")
        print("=" * 60)
        
        del train_dataset, obs_data
        print("[Data] Loading test dataset...")
        test_dataset = PerturbationDataset(self.data_path, split="test", split_seed=self.seed)
        print(f"[Data] test={len(test_dataset)} perturbations")
        self._evaluate(cpd, test_dataset)

        
        self._save_history()

        print(f"\nTraining complete. Results saved to {self.output_dir}")

    # ------------------------------------------------------------------
    # Prior graph
    # ------------------------------------------------------------------

    def _build_prior_graph(self, gene_names: list[str]) -> Optional[np.ndarray]:
        """Load pre-built prior graph."""
        if self.prior_graph_path is None:
            print("[Prior] No prior graph path specified, skipping")
            return None

        path = Path(self.prior_graph_path)
        if not path.exists():
            print(f"[Prior] Prior graph not found: {path}, skipping")
            return None

        prior_adj, _ = load_prior_graph(path, gene_names)
        n_edges = int(prior_adj.sum())
        density = n_edges / (len(gene_names) ** 2) * 100
        print(f"[Prior] Loaded prior graph: {n_edges} edges, density={density:.2f}%")
        return prior_adj

    # ------------------------------------------------------------------
    # Epoch callback / checkpoint
    # ------------------------------------------------------------------

    def _epoch_callback(
        self, cpd, *, stage, epoch, record, optimizer, n_epochs, mu=None,
    ):
        """Per-epoch callback: record history + periodic checkpoint."""
        self.history.append(record)

        
        is_last = (epoch == n_epochs - 1)
        if (epoch + 1) % self.CKPT_INTERVAL == 0 or is_last:
            self._save_checkpoint(
                cpd,
                completed_stage=stage - 1,
                current_stage=stage,
                current_epoch=epoch,
                optimizer=optimizer,
                mu=mu,
            )
            self._save_history()

    def _save_checkpoint(
        self, cpd, *, completed_stage, current_stage, current_epoch,
        optimizer=None, mu=None,
    ):
        """Save last.ckpt for resume."""
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "completed_stage": completed_stage,
            "current_stage": current_stage,
            "current_epoch": current_epoch,
            "training_history": self.history,
        }
        if cpd.causal_graph_learner is not None:
            ckpt["cg_state"] = cpd.causal_graph_learner.state_dict()
        if cpd.learned_adj is not None:
            ckpt["learned_adj"] = cpd.learned_adj
        if cpd.neural_scm is not None:
            ckpt["scm_state"] = cpd.neural_scm.state_dict()
        if optimizer is not None:
            ckpt["optimizer_state"] = optimizer.state_dict()
        if mu is not None:
            ckpt["mu"] = mu

        torch.save(ckpt, self.ckpt_dir / "last.ckpt")

    def _load_checkpoint(self) -> Optional[dict]:
        """Load last.ckpt."""
        path = self.ckpt_dir / "last.ckpt"
        if not path.exists():
            print("[Resume] No checkpoint found, starting from scratch")
            return None
        print(f"[Resume] Loading {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    @staticmethod
    def _restore_model_state(cpd: CPDModel, ckpt: dict):
        """Restore CPDModel weights from checkpoint."""
        from model.causal_graph import CausalGraphLearner
        from model.neural_scm import NeuralSCM

        if "learned_adj" in ckpt:
            cpd.learned_adj = ckpt["learned_adj"]

        if "cg_state" in ckpt:
            cpd.causal_graph_learner = CausalGraphLearner(
                n_genes=cpd.n_genes,
                hidden_dim=cpd.cg_hidden_dim,
                lambda_l1=cpd.cg_lambda_l1,
                lambda_l1_no_prior=cpd.cg_lambda_l1_no_prior,
            ).to(cpd.device)
            cpd.causal_graph_learner.load_state_dict(ckpt["cg_state"])

        if "scm_state" in ckpt and cpd.learned_adj is not None:
            cpd.neural_scm = NeuralSCM(
                adjacency=cpd.learned_adj,
                gene_names=cpd.gene_names,
                hidden_dim=cpd.scm_hidden_dim,
            ).to(cpd.device)
            cpd.neural_scm.load_state_dict(ckpt["scm_state"])

    # ------------------------------------------------------------------
    # History / evaluation persistence
    # ------------------------------------------------------------------

    def _save_history(self):
        """Write training history to CSV."""
        if not self.history:
            return
        path = self.output_dir / "training_history.csv"
        fieldnames = list(self.history[0].keys())
        
        all_keys = set()
        for r in self.history:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.history:
                writer.writerow(r)

    def _dump_config(self):
        """Write experiment config snapshot to YAML."""
        cfg = {
            "data_path": str(self.data_path),
            "output_dir": str(self.output_dir),
            "device": self.device,
            "seed": self.seed,
            "stage1": {
                "n_epochs": self.cg_n_epochs,
                "lr": self.cg_lr,
                "hidden_dim": self.cg_hidden_dim,
                "lambda_l1": self.cg_lambda_l1,
                "lambda_l1_no_prior": self.cg_lambda_l1_no_prior,
                "prior_mask_mode": self.cg_prior_mask_mode,
            },
            "stage2": {
                "n_epochs": self.scm_n_epochs,
                "lr": self.scm_lr,
                "hidden_dim": self.scm_hidden_dim,
                "batch_size": self.scm_batch_size,
            },
            "stage3": {
                "diff_threshold": self.diff_threshold,
                "max_interventions": self.max_interventions,
            },
            "prior_graph_path": str(self.prior_graph_path) if self.prior_graph_path else None,
        }
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, cpd: CPDModel, test_dataset: PerturbationDataset):
        """Evaluate on test set and save metrics."""
        from scipy.stats import spearmanr, pearsonr

        # Compute per-perturbation DEGs for CausalDN-aligned evaluation
        print("[Eval] Computing per-perturbation DEGs (top-1000) ...")
        test_dataset.compute_per_pert_degs(n_top=1000, n_min=100)

        # Build LFC lookup for CausalDN-style Spearman/Pearson
        print("[Eval] Building LFC lookup ...")
        lfc_lookup = test_dataset.build_lfc_lookup()

        metrics = PerturbationMetrics()
        ctrl_mean = test_dataset.get_ctrl_mean()

        per_pert_rows = []

        for i in range(len(test_dataset)):
            sample = test_dataset.samples[i]
            x_source = ctrl_mean
            x_target = torch.from_numpy(sample["pert_expr"])
            true_label = torch.from_numpy(sample["pert_label"])

            result = cpd.predict(x_source, x_target)

            
            pred_scores = result["gene_scores"]

            deg_indices = sample.get("deg_indices")

            # LFC-based Spearman/Pearson (CausalDN primary)
            lfc_corr = None
            scores_np = pred_scores.detach().cpu().numpy() if hasattr(pred_scores, 'detach') else np.asarray(pred_scores)
            top1_idx = int(np.argmax(scores_np))
            true_idx = np.where(true_label.numpy())[0]
            if len(true_idx) == 1 and true_idx[0] in lfc_lookup and top1_idx in lfc_lookup:
                true_lfc = lfc_lookup[true_idx[0]]
                pred_lfc = lfc_lookup[top1_idx]
                sp, _ = spearmanr(pred_lfc, true_lfc)
                pe, _ = pearsonr(pred_lfc, true_lfc)
                lfc_corr = {
                    "spearman": float(sp) if np.isfinite(sp) else 0.0,
                    "pearson": float(pe) if np.isfinite(pe) else 0.0,
                }

            metrics.update(pred_scores, true_label, sample["pert_name"],
                           deg_indices=deg_indices, lfc_corr=lfc_corr)

            
            true_genes = set(np.where(true_label.numpy())[0].tolist())
            intervention_set = result.get("intervention_set", [])
            pred_genes = set(intervention_set)
            per_pert_rows.append({
                "pert_name": sample["pert_name"],
                "n_true": len(true_genes),
                "n_pred": len(pred_genes),
                "overlap": len(true_genes & pred_genes),
                "jaccard": len(true_genes & pred_genes) / max(len(true_genes | pred_genes), 1),
            })

        summary = metrics.compute()
        metrics.print_summary(summary)

        # --- metrics_summary.csv ---
        summary_path = self.output_dir / "metrics_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    writer.writerow([k, f"{v:.6f}"])

        # --- metrics_per_pert.csv ---
        per_pert_path = self.output_dir / "metrics_per_pert.csv"
        if per_pert_rows:
            with open(per_pert_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=per_pert_rows[0].keys())
                writer.writeheader()
                writer.writerows(per_pert_rows)
