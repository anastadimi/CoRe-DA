import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for CoRe-DA experiment")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seed", type=int, default=2, help="Random seed")

    parser.add_argument(
        "--base_path_wts",
        type=str,
        default="/media",
        help="Path to save weights.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/anonymous_user",
        help="Base dir path (where the data is stored)",
    )

    parser.add_argument("--project", type=str, default="CoRe-DA", help="Project name")
    parser.add_argument("--exp", type=str, default="test", help="Experiment name")
    parser.add_argument(
        "--source_dataset", type=str, default="aixsuture", help="Source dataset name"
    )
    parser.add_argument(
        "--target_dataset", type=str, default="jigsaws", help="Target dataset name"
    )
    parser.add_argument("--anno", type=str, default="osats", help="Annotation type")

    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--norm_stats",
        type=dict,
        default={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        help="Normalization statistics",
    )
    parser.add_argument(
        "--norm_labels", action="store_true", help="Whether to normalize labels"
    )
    parser.add_argument(
        "--no_norm_labels",
        action="store_false",
        dest="norm_labels",
        help="Disable label normalization",
    )
    parser.set_defaults(norm_labels=True)
    parser.add_argument(
        "--num_segments", type=int, default=12, help="Number of segments"
    )
    parser.add_argument("--snippet_size", type=int, default=12, help="Snippet size")
    
    parser.add_argument(
        "--ordered_sampling", action="store_true", help="Whether to sample ordered snippets during training"
    )
    parser.add_argument(
        "--no_ordered_sampling",
        action="store_false",
        dest="ordered_sampling",
        help="Disable ordered_sampling",
    )
    parser.set_defaults(ordered_sampling=True)

    
    # ---------- Testing ----------
    parser.add_argument(
        "--n_exemplars",
        type=int,
        default=10,
        help="Number of source examplar videos for testing",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="uniform",
        choices=["uniform", "distribution", "random"],
        help="Sampling method for exemplars",
    )

    # ---------- Model ----------
    parser.add_argument(
        "--feature_dim", type=int, default=256, help="Feature dimension size"
    )
    parser.add_argument(
        "--dropout_keep_prob",
        type=float,
        default=0.5,
        help="I3D base model dropout probability",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained model"
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_false",
        dest="pretrained",
        help="Do not use pretrained model",
    )
    parser.set_defaults(pretrained=True)
    parser.add_argument(
        "--freeze_until",
        type=str,
        default=None,
        help="Must be one of InceptionI3d.VALID_ENDPOINTS",
    )
    parser.add_argument(
        "--freeze_bn_stats",
        action="store_true",
        help="If set, BatchNorm3d layers in frozen part will be set to eval()",
    )
    parser.add_argument(
        "--no_freeze_bn_stats",
        action="store_false",
        dest="freeze_bn_stats",
        help="If set, BatchNorm3d layers in frozen part will continue to update running stats.",
    )
    parser.set_defaults(freeze_bn_stats=True)
    parser.add_argument(
        "--weight_path",
        type=str,
        default="kinetics.pt",
        help="Path to Kinetics weights for I3D.",
    )
    
    # CoRe-DA
    parser.add_argument(
        "--abs_regressor_hidden_dim",
        type=int,
        default=256,
        help="Hidden dim of absolute regressor MLP.",
    )
    parser.add_argument(
        "--rel_regressor_hidden_dim",
        type=int,
        default=256,
        help="Hidden dim of relative regressor MLP",
    )


    # ---------- Optimizer ----------
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr_i3d", type=float, default=1e-05, help="Learning rate for feature extractor")
    parser.add_argument("--lr", type=float, default=5e-05, help="Learning rate")
    parser.add_argument(
        "--scheduler_epoch", type=int, default=None, help="Linear scheduler decay epoch"
    )
    parser.add_argument(
        "--stop_grad_y_T",
        action="store_true",
        help="Whether to stop gradient flow from y_T",
    )
    parser.add_argument("--S_only_warmup", type=int, default=0, help="Source only training epochs")

    
    # ---------- Losses ----------
    parser.add_argument("--coef_alpha", type=float, default=1.0, help="Coefficient for supervised losses")    
    parser.add_argument("--coef_beta", type=float, default=1.0, help="Coefficient for consistency loss (source)")    
    parser.add_argument("--coef_gamma", type=float, default=1.0, help="Coefficient for consistency loss (target)")    
    
    # --------- Background Mixing --------- 
    parser.add_argument("--mix_test",action="store_true",help="Whether to mix T at testing")
    
    args = parser.parse_args()
    return args
