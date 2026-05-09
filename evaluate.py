"""RIVER Unified Evaluation Entry Point."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, pearsonr

from data.dataset import PerturbationDataset
from training.metrics import PerturbationMetrics


# ------------------------------------------------------------------
# Core evaluation
# ------------------------------------------------------------------

def evaluate_predictions(
    dataset: PerturbationDataset,
    pred_dict: dict,
    output_dir: Path,
    method_name: str = "unknown",
    forward_consistency_map: dict = None,
):
    """Unified evaluation entry point."""
    lfc_lookup = dataset.build_lfc_lookup()
    print(f"  LFC lookup: {len(lfc_lookup)} genes")

    metrics = PerturbationMetrics()
    n_evaluated, n_zero_pred = 0, 0

    for sample in dataset.samples:
        pert_name = sample["pert_name"]
        true_label = torch.from_numpy(sample["pert_label"])

        if pert_name in pred_dict:
            pred_scores = pred_dict[pert_name]
        else:
            pred_scores = np.zeros(dataset.n_genes, dtype=np.float32)
            n_zero_pred += 1

        if not isinstance(pred_scores, np.ndarray):
            pred_scores = np.asarray(pred_scores, dtype=np.float32)

        deg_indices = sample.get("deg_indices")

        
        lfc_corr = None
        top1_idx = int(np.argmax(pred_scores))
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

        fc = None
        if forward_consistency_map and pert_name in forward_consistency_map:
            fc = forward_consistency_map[pert_name]

        metrics.update(
            pred_scores=torch.from_numpy(pred_scores),
            true_label=true_label,
            pert_name=pert_name,
            forward_consistency=fc,
            deg_indices=deg_indices,
            lfc_corr=lfc_corr,
        )
        n_evaluated += 1

    print(f"  Evaluated: {n_evaluated} perts ({n_zero_pred} zero predictions)")

    summary = metrics.compute()
    summary["method"] = method_name
    metrics.print_summary(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = output_dir / "eval.json"
    with open(eval_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {eval_path}")

    
    per_pert_path = output_dir / "per_pert_results.json"
    with open(per_pert_path, "w") as f:
        json.dump(metrics.results, f, indent=2)
    print(f"Per-perturbation results: {per_pert_path}")

    return summary


# ------------------------------------------------------------------
# predictions.npz loader
# ------------------------------------------------------------------

def load_predictions_npz(npz_path: str) -> dict:
    """Load predictions.npz -> {pert_name: scores_array}."""
    data = np.load(npz_path, allow_pickle=True)
    pert_names = data["pert_names"]
    pred_scores = data["pred_scores"]
    return {str(n): pred_scores[i] for i, n in enumerate(pert_names)}


# ------------------------------------------------------------------
# Training Signature Builder (for score_mode="signature")
# ------------------------------------------------------------------

def build_train_signatures(data_path: str | Path) -> dict[int, np.ndarray]:
    """Build per-gene average perturbation signatures from training data."""
    train_ds = PerturbationDataset(Path(data_path), split="train")
    ctrl_mean = train_ds.get_ctrl_mean().numpy()

    train_deltas: dict[int, list[np.ndarray]] = {}
    for s in train_ds.samples:
        pert_idx = np.where(s["pert_label"])[0]
        delta = s["pert_expr"] - ctrl_mean
        for l in pert_idx:
            l = int(l)
            if l not in train_deltas:
                train_deltas[l] = []
            train_deltas[l].append(delta)

    signatures = {g: np.mean(ds, axis=0) for g, ds in train_deltas.items()}
    print(f"  [Signatures] Built {len(signatures)} gene signatures from "
          f"{len(train_ds.samples)} training perturbations")
    return signatures


def build_model_signatures(model_dir: str | Path, data_path: str | Path,
                           device: str = "cuda") -> dict[int, np.ndarray]:
    """Build model-predicted single-gene perturbation signatures via SCM."""
    from model.cpd import CPDModel

    cpd = CPDModel.load(Path(model_dir), device=device)
    train_ds = PerturbationDataset(Path(data_path), split="train")
    ctrl_mean = train_ds.get_ctrl_mean()

    
    gene_pert_vals: dict[int, list[float]] = {}
    for s in train_ds.samples:
        pert_idx = np.where(s["pert_label"])[0]
        for l in pert_idx:
            l = int(l)
            if l not in gene_pert_vals:
                gene_pert_vals[l] = []
            gene_pert_vals[l].append(float(s["pert_expr"][l]))

    avg_pert_vals = {g: np.mean(vs) for g, vs in gene_pert_vals.items()}

    
    x_source = ctrl_mean.unsqueeze(0).to(device)
    signatures = {}
    with torch.no_grad():
        for gene_idx, pert_val in avg_pert_vals.items():
            iv = torch.tensor([[pert_val]], device=device)
            x_do = cpd.neural_scm.do_intervention(x_source, [gene_idx], iv)
            delta = (x_do - x_source).squeeze(0).cpu().numpy()
            signatures[gene_idx] = delta

    print(f"  [ModelSig] Built {len(signatures)} model-predicted signatures "
          f"via SCM do-intervention")
    return signatures


# ------------------------------------------------------------------
# CRISP model inference
# ------------------------------------------------------------------

def crisp_inference(model_dir: str, dataset: PerturbationDataset,
                    device: str = "cuda", score_mode: str = None,
                    diff_threshold: float = None,
                    max_interventions: int = None,
                    mask_self_effect: bool = False,
                    pairwise_k: int = None,
                    train_signatures: dict = None) -> tuple:
    """CRISP model inference, returns (pred_dict, fc_map)."""
    from model.cpd import CPDModel

    cpd = CPDModel.load(Path(model_dir), device=device)

    if cpd.gcps is not None:
        if score_mode is not None:
            cpd.gcps.score_mode = score_mode
            print(f"  [Override] score_mode = {score_mode}")
        if diff_threshold is not None:
            cpd.gcps.diff_threshold = diff_threshold
            print(f"  [Override] diff_threshold = {diff_threshold}")
        if max_interventions is not None:
            cpd.gcps.max_interventions = max_interventions
            print(f"  [Override] max_interventions = {max_interventions}")
        if pairwise_k is not None:
            cpd.gcps.pairwise_k = pairwise_k
            print(f"  [Override] pairwise_k = {pairwise_k}")
        if train_signatures is not None:
            cpd.gcps.train_signatures = train_signatures
            print(f"  [Override] train_signatures = {len(train_signatures)} genes")
    ctrl_mean = dataset.get_ctrl_mean()
    pred_dict = {}
    fc_map = {}

    if mask_self_effect:
        print("  [MaskSelfEffect] Target gene expression replaced with control values")

    for i, sample in enumerate(dataset.samples):
        x_source = ctrl_mean
        x_target = torch.from_numpy(sample["pert_expr"])

        if mask_self_effect:
            pert_mask = sample["pert_label"]
            pert_idx = np.where(np.array(pert_mask) > 0.5)[0]
            x_target = x_target.clone()
            x_target[pert_idx] = x_source[pert_idx]

        result = cpd.predict(x_source, x_target)

        if "gene_scores" in result:
            scores = result["gene_scores"].cpu().numpy()
        else:
            scores = np.zeros(cpd.n_genes, dtype=np.float32)
            for rank, gene_idx in enumerate(result["intervention_set"]):
                scores[gene_idx] = float(len(result["intervention_set"]) - rank)

        pred_dict[sample["pert_name"]] = scores

        # Forward consistency
        intervention_set = result["intervention_set"]
        if intervention_set:
            x_s = x_source.unsqueeze(0).to(cpd.device)
            vals = result["intervention_values"].unsqueeze(0).to(cpd.device)
            x_do = cpd.neural_scm.do_intervention(x_s, intervention_set, vals)
            x_do_np = x_do.squeeze(0).detach().cpu().numpy()
            x_t_np = x_target.numpy()
            corr = np.corrcoef(x_do_np, x_t_np)[0, 1]
            fc_map[sample["pert_name"]] = float(corr) if np.isfinite(corr) else 0.0

        if (i + 1) % 10 == 0:
            print(f"  Inference {i+1}/{len(dataset.samples)}")

    return pred_dict, fc_map


# ------------------------------------------------------------------
# CRISP Per-Perturbation Fine-Tuning  (following CausalDN protocol)
# ------------------------------------------------------------------

def crisp_finetune(model_dir: str, dataset: PerturbationDataset,
                   per_sample_interv: list,
                   device: str = "cuda", score_mode: str = None,
                   ft_steps: int = 10, ft_lr: float = 1e-3,
                   ft_wd: float = 0.0,
                   ft_grad_clip: float = 1.0,
                   mask_self_effect: bool = False,
                   train_signatures: dict = None) -> tuple:
    """CRISP per-perturbation fine-tuning (CausalDN protocol).

    Args:
        per_sample_interv: list of intervention index arrays, one per sample.
    """
    from model.cpd import CPDModel

    cpd = CPDModel.load(Path(model_dir), device=device)
    if score_mode is not None and cpd.gcps is not None:
        cpd.gcps.score_mode = score_mode
        print(f"  [Override] score_mode = {score_mode}")
    if train_signatures is not None and cpd.gcps is not None:
        cpd.gcps.train_signatures = train_signatures
        print(f"  [Override] train_signatures = {len(train_signatures)} genes")

    ctrl_mean = dataset.get_ctrl_mean()
    pred_dict = {}
    fc_map = {}

    
    original_scm_state = copy.deepcopy(cpd.neural_scm.state_dict())

    for i, sample in enumerate(dataset.samples):
        
        cpd.neural_scm.load_state_dict(copy.deepcopy(original_scm_state))

        x_source = ctrl_mean.to(device)
        x_target = torch.from_numpy(sample["pert_expr"]).to(device)
        interv_idx = per_sample_interv[i]

        
        if len(interv_idx) >= 1 and ft_steps > 0:
            cpd.neural_scm.train()
            
            regularized = []
            not_regularized = []
            for name, param in cpd.neural_scm.named_parameters():
                if not param.requires_grad:
                    continue
                if name.endswith(".bias") or len(param.shape) == 1:
                    not_regularized.append(param)
                else:
                    regularized.append(param)
            param_groups = [
                {"params": regularized, "weight_decay": ft_wd, "lr": ft_lr},
                {"params": not_regularized, "weight_decay": 0., "lr": ft_lr},
            ]
            optimizer = torch.optim.AdamW(param_groups)

            x_src_b = x_source.unsqueeze(0)  # [1, n_genes]
            interv_vals = x_target[interv_idx].unsqueeze(0)  # [1, k]

            for step in range(ft_steps):
                optimizer.zero_grad()

                x_pred = cpd.neural_scm.cascade_do_intervention_differentiable(
                    x_src_b, interv_idx, interv_vals)
                loss = F.mse_loss(x_pred.squeeze(0), x_target)

                loss.backward()
                if ft_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(cpd.neural_scm.parameters(),
                                                   ft_grad_clip)
                optimizer.step()

            cpd.neural_scm.eval()

        
        x_source_cpu = ctrl_mean
        x_target_cpu = torch.from_numpy(sample["pert_expr"])

        if mask_self_effect:
            x_target_cpu = x_target_cpu.clone()
            x_target_cpu[interv_idx] = x_source_cpu[interv_idx]

        result = cpd.predict(x_source_cpu, x_target_cpu)

        if "gene_scores" in result:
            scores = result["gene_scores"].cpu().numpy()
        else:
            scores = np.zeros(cpd.n_genes, dtype=np.float32)
            for rank, gene_idx in enumerate(result["intervention_set"]):
                scores[gene_idx] = float(len(result["intervention_set"]) - rank)

        pred_dict[sample["pert_name"]] = scores

        # Forward consistency
        intervention_set = result["intervention_set"]
        if intervention_set:
            x_s = x_source.unsqueeze(0)
            vals = result["intervention_values"].unsqueeze(0).to(device)
            x_do = cpd.neural_scm.do_intervention(x_s, intervention_set, vals)
            x_do_np = x_do.squeeze(0).detach().cpu().numpy()
            x_t_np = x_target.cpu().numpy()
            corr = np.corrcoef(x_do_np, x_t_np)[0, 1]
            fc_map[sample["pert_name"]] = float(corr) if np.isfinite(corr) else 0.0

        if (i + 1) % 10 == 0:
            print(f"  Inference {i+1}/{len(dataset.samples)}")

    return pred_dict, fc_map


# ------------------------------------------------------------------
# CRISP forward evaluation
# ------------------------------------------------------------------

def _evaluate_forward(model_dir, dataset, device):
    """Stage 2 forward prediction evaluation."""
    from model.cpd import CPDModel
    from training.metrics import ForwardMetrics

    cpd = CPDModel.load(Path(model_dir), device=device)
    ctrl_mean = dataset.get_ctrl_mean()
    metrics = ForwardMetrics()
    ctrl_np = ctrl_mean.numpy()

    for i in range(len(dataset)):
        sample = dataset.samples[i]
        x_source = ctrl_mean.unsqueeze(0).to(cpd.device)
        pert_expr = sample["pert_expr"]
        pert_mask = sample["pert_label"]

        intervention_set = np.where(pert_mask)[0].tolist()
        if not intervention_set:
            continue

        interv_vals = torch.from_numpy(
            pert_expr[intervention_set]
        ).unsqueeze(0).to(cpd.device)
        x_do = cpd.neural_scm.do_intervention(
            x_source, intervention_set, interv_vals
        )
        pred_expr = x_do.squeeze(0).detach().cpu().numpy()
        metrics.update(pred_expr, pert_expr, ctrl_np, sample["pert_name"])

        if (i + 1) % 10 == 0:
            print(f"  [Forward] {i+1}/{len(dataset)}")

    summary = metrics.compute()
    metrics.print_summary(summary)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RIVER Unified Evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_dir", type=str,
                       help="RIVER model directory (inference + evaluation)")
    group.add_argument("--predictions", type=str,
                       help="Path to predictions.npz (generic evaluation)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to h5ad data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for evaluation results")
    parser.add_argument("--method", type=str, default=None,
                        help="Method name (written to eval.json)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--mode", type=str, default="reverse",
                        choices=["reverse", "forward", "both"],
                        help="RIVER model evaluation mode")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Also save predictions.npz")
    parser.add_argument("--score_mode", type=str, default=None,
                        help="Override GCPS score_mode (raw|pairwise|signature)")
    parser.add_argument("--finetune", action="store_true",
                        help="Per-perturbation fine-tuning (CausalDN protocol)")
    parser.add_argument("--ft_steps", type=int, default=10,
                        help="Fine-tuning steps per perturbation (default: 10)")
    parser.add_argument("--ft_lr", type=float, default=1e-3,
                        help="Fine-tuning learning rate (default: 1e-3)")
    parser.add_argument("--ft_wd", type=float, default=1e-6,
                        help="Fine-tuning weight decay (default: 1e-6)")
    parser.add_argument("--ft_grad_clip", type=float, default=1.0,
                        help="Gradient clipping max_norm (default: 1.0, CausalDN)")
    parser.add_argument("--diff_threshold", type=float, default=None,
                        help="Override GCPS diff_threshold")
    parser.add_argument("--max_interventions", type=int, default=None,
                        help="Override GCPS max_interventions")
    parser.add_argument("--mask_self_effect", action="store_true",
                        help="Mask self-effect: replace target gene expression with control values")
    parser.add_argument("--pairwise_k", type=int, default=None,
                        help="TSPS Stage 1 top-K candidates (for score_mode=pairwise, default: 50)")
    parser.add_argument("--filter_perts", type=str, default=None,
                        help="JSON file with perturbation names to evaluate (subset filter)")
    parser.add_argument("--cell_subsample_ratio", type=float, default=1.0,
                        help="Cell sub-sampling ratio for uncertainty (CausalDN protocol, default: 1.0)")
    parser.add_argument("--cell_subsample_seed", type=int, default=0,
                        help="Cell sub-sampling random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"[1] Loading data: {args.data}")
    ss_kwargs = {}
    if args.cell_subsample_ratio < 1.0:
        ss_kwargs = dict(cell_subsample_ratio=args.cell_subsample_ratio,
                         cell_subsample_seed=args.cell_subsample_seed)
        print(f"  [SubSample] ratio={args.cell_subsample_ratio}, seed={args.cell_subsample_seed}")
    dataset = PerturbationDataset(Path(args.data), split=args.split, **ss_kwargs)
    print(f"  {dataset.n_genes} genes, {len(dataset.samples)} {args.split} perts")

    if args.filter_perts:
        import json as _json
        with open(args.filter_perts) as _f:
            keep_perts = set(_json.load(_f))
        before = len(dataset.samples)
        dataset.samples = [s for s in dataset.samples if s["pert_name"] in keep_perts]
        print(f"  [Filter] {before} -> {len(dataset.samples)} perts (filter: {args.filter_perts})")

    print("[2] Computing per-perturbation DEGs (top-1000) ...")
    dataset.compute_per_pert_degs(n_top=1000, n_min=100)

    if args.predictions:
        print(f"[3] Loading predictions: {args.predictions}")
        pred_dict = load_predictions_npz(args.predictions)
        method_name = args.method or Path(args.predictions).parent.name
        print(f"  {len(pred_dict)} predictions loaded")

        print("[4] Evaluating ...")
        evaluate_predictions(dataset, pred_dict, output_dir, method_name)

    else:
        if args.mode in ("forward", "both"):
            _evaluate_forward(args.model_dir, dataset, args.device)

        if args.mode in ("reverse", "both"):
            print(f"[3] RIVER inference: {args.model_dir}")

            # Build training signatures if score_mode requires them
            train_sigs = None
            if args.score_mode == "signature":
                print("  [Signature] Building training perturbation signatures...")
                train_sigs = build_train_signatures(args.data)

            if args.finetune:
                print(f"  [FT] ft_steps={args.ft_steps}, ft_lr={args.ft_lr}")
                sample_interv = [
                    np.where(s["pert_label"])[0].tolist()
                    for s in dataset.samples
                ]
                pred_dict, fc_map = crisp_finetune(
                    args.model_dir, dataset, sample_interv,
                    device=args.device,
                    score_mode=args.score_mode,
                    ft_steps=args.ft_steps, ft_lr=args.ft_lr,
                    ft_wd=args.ft_wd,
                    ft_grad_clip=args.ft_grad_clip,
                    mask_self_effect=args.mask_self_effect,
                    train_signatures=train_sigs)
            else:
                pred_dict, fc_map = crisp_inference(
                    args.model_dir, dataset, args.device,
                    score_mode=args.score_mode,
                    diff_threshold=args.diff_threshold,
                    max_interventions=args.max_interventions,
                    mask_self_effect=args.mask_self_effect,
                    pairwise_k=args.pairwise_k,
                    train_signatures=train_sigs)

            method_name = args.method or "RIVER"

            if args.save_predictions:
                output_dir.mkdir(parents=True, exist_ok=True)
                pert_names = list(pred_dict.keys())
                pred_scores = np.stack([pred_dict[n] for n in pert_names])
                npz_path = output_dir / "predictions.npz"
                np.savez(npz_path,
                         pert_names=np.array(pert_names),
                         pred_scores=pred_scores)
                print(f"  Predictions saved to: {npz_path}")

            print("[4] Evaluating ...")
            evaluate_predictions(
                dataset, pred_dict, output_dir, method_name,
                forward_consistency_map=fc_map)


if __name__ == "__main__":
    main()
