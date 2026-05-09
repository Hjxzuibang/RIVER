"""RIVER Training Entry Point."""

import argparse
from pathlib import Path

from config import CPDConfig
from training.trainer import CPDTrainer


def main():
    parser = argparse.ArgumentParser(description="RIVER: Reverse Inference Via Effect Reconstruction")
    parser.add_argument("--config", type=str, default="configs/norman.yaml",
                        help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (overrides config)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (overrides config)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()
    
    config = CPDConfig.from_yaml(Path(args.config))
    
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.exp_name is not None:
        config.exp_name = args.exp_name
    
    data_path = config.get_data_path()
    output_dir = config.get_output_dir()
    
    print(f"Config:  {args.config}")
    print(f"Data:    {data_path}")
    print(f"Output:  {output_dir}")
    print(f"Device:  {config.device}")
    print(f"Seed:    {config.seed}")
    
    trainer = CPDTrainer(
        data_path=data_path,
        output_dir=output_dir,
        device=config.device,
        seed=config.seed,
        resume=args.resume,
        cg_n_epochs=config.stage1.n_epochs,
        cg_lr=config.stage1.lr,
        cg_hidden_dim=config.stage1.hidden_dim,
        cg_lambda_l1=config.stage1.lambda_l1,
        cg_lambda_l1_no_prior=config.stage1.lambda_l1_no_prior,
        cg_prior_mask_mode=config.stage1.prior_mask_mode,
        scm_n_epochs=config.stage2.n_epochs,
        scm_lr=config.stage2.lr,
        scm_hidden_dim=config.stage2.hidden_dim,
        scm_batch_size=config.stage2.batch_size,
        diff_threshold=config.stage3.diff_threshold,
        max_interventions=config.stage3.max_interventions,
        score_mode=config.stage3.score_mode,
        adj_threshold=config.stage1.adj_threshold,
        lambda_pert=config.stage1.lambda_pert,
        lambda_pert_scm=config.stage2.lambda_pert_scm,
        scheduled_sampling_start=config.stage2.scheduled_sampling_start,
        prior_graph_path=config.get_prior_graph_path(),
    )
    
    trainer.run()


if __name__ == "__main__":
    main()
