import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for CoRe-DA experiment")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seed", type=int, default=2, help="Random seed")

    # ---------- Experiment namings and setup ----------
    parser.add_argument(
        "--path_to_datasets",
        type=str,
        default="path/to/datasets/dir",
        help="Base dir path",
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
        "--label_column", type=str, default="grs", help="Label column name"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers for dataloader"
    )

    # ---------- Data pipeline ----------
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
        "--ordered_sampling",
        action="store_true",
        help="Whether to sample ordered snippets during training",
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
        choices=["uniform", "distribution", "random", "closest_label"],
        help="Sampling method for exemplars",
    )

    # ---------- Model ----------
    # Feature Extractor
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
        default="Mixed_4c",
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
        default="rgb_imagenet.pt",
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
        "--epochs", type=int, default=150, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr_i3d",
        type=float,
        default=1e-05,
        help="Learning rate for feature extractor",
    )
    parser.add_argument("--lr", type=float, default=5e-05, help="Learning rate")
    parser.add_argument(
        "--scheduler_epoch", type=int, default=None, help="Linear scheduler decay epoch"
    )
    parser.add_argument(
        "--stop_grad_y_T",
        action="store_true",
        help="Whether to stop gradient flow from y_T",
    )

    # ---------- Losses ----------
    parser.add_argument(
        "--coef_sup_loss_rel",
        type=float,
        default=1.0,
        help="Coefficient for relative supervised loss (source)",
    )
    parser.add_argument(
        "--coef_sup_loss_abs",
        type=float,
        default=1.0,
        help="Coefficient for absolute supervised loss (source exemplar)",
    )
    parser.add_argument(
        "--coef_cons_loss_S",
        type=float,
        default=1.0,
        help="Coefficient for consistency loss (source)",
    )
    parser.add_argument(
        "--coef_cons_loss_T",
        type=float,
        default=1.0,
        help="Coefficient for consistency loss (target)",
    )
    parser.add_argument(
        "--use_cons_S", action="store_true", help="Whether to use consistency loss in S"
    )

    # --------- Background Mixing ---------
    parser.add_argument(
        "--use_video_BG_in_S",
        action="store_true",
        help="Whether to mix video-based BG to S branch",
    )
    parser.add_argument(
        "--use_clip_BG_in_S",
        action="store_true",
        help="Whether to mix clip-based BG to S branch",
    )
    parser.add_argument(
        "--use_video_BG_in_ex",
        action="store_true",
        help="Whether to mix video-based BG to ex branch",
    )
    parser.add_argument(
        "--use_clip_BG_in_ex",
        action="store_true",
        help="Whether to mix clip-based BG to ex branch",
    )
    parser.add_argument(
        "--use_video_BG_in_T",
        action="store_true",
        help="Whether to mix video-based BG to T branch",
    )
    parser.add_argument(
        "--use_clip_BG_in_T",
        action="store_true",
        help="Whether to mix clip-based BG to T branch",
    )
    parser.add_argument(
        "--gamma_alpha",
        type=float,
        default=0.5,
        help="Gamma value for uniform sampling used for BG mixing",
    )
    parser.add_argument(
        "--mix_test", action="store_true", help="Whether to mix T at testing"
    )

    args = parser.parse_args()
    return args
