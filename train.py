import json
import os

import torch
from torch.utils.data import DataLoader

from dataset import MixingDataset, TargetPlusExemplars
from models import CoRe_DA
from opts import get_args
from trainer import Trainer
from utils import seed_everything

args = get_args()
print("Configurations:")
print(f"Use stop grad = {args.stop_grad_y_T}")
print(f"Freeze I3D until: {args.freeze_until}")
print(f"Use use_cons_S: {args.use_cons_S}")
print(f"Exemplar samling method: {args.sampling_method}")


seed_everything(args.seed)

# --------- GPU ---------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("CUDA is not available.")


# --------- Read datasets from json file ---------
data_root_dir = args.path_to_datasets
source_db_json_path = os.path.join(
    os.getcwd(), f"db_json/{args.source_dataset}_{args.anno}.db.json"
)
with open(source_db_json_path, "r") as f:
    source_db = json.load(f)
target_db_json_path = os.path.join(
    os.getcwd(), f"db_json/{args.target_dataset}_{args.anno}.db.json"
)
with open(target_db_json_path, "r") as f:
    target_db = json.load(f)


# # --------- Datasets ---------
train_ds = MixingDataset(
    source_db_json_file=source_db,
    target_db_json_file=target_db,
    label_column="grs",
    data_root_dir=data_root_dir,
    num_segments=args.num_segments,
    snippet_size=args.snippet_size,
    norm_stats=args.norm_stats,
    ordered_sampling=args.ordered_sampling,
    norm_labels=True,
    use_video_BG_in_S=args.use_video_BG_in_S,
    use_clip_BG_in_S=args.use_clip_BG_in_S,
    use_video_BG_in_ex=args.use_video_BG_in_ex,
    use_clip_BG_in_ex=args.use_clip_BG_in_ex,
    use_video_BG_in_T=args.use_video_BG_in_T,
    use_clip_BG_in_T=args.use_clip_BG_in_T,
    # Mixing
    gamma_alpha=args.gamma_alpha,
    # Debug saving
    save_mixed_debug=False,
    debug_output_root="debug_mixed",
)

val_ds = TargetPlusExemplars(
    target_db_json_file=target_db,
    source_db_json_file=source_db,
    label_column="grs",
    data_root_dir=data_root_dir,
    num_segments=args.num_segments,
    snippet_size=args.snippet_size,
    norm_stats=args.norm_stats,
    n_exemplars=args.n_exemplars,
    sampling_method=args.sampling_method,
    gamma_alpha=0.25,  # fixed alpha
    mix_test=args.mix_test,
    norm_labels=True,
    save_mixed_debug=False,
    debug_output_root="debug_mixed",
)
# --------- Dataloaders ---------
train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    drop_last=True,
)

val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=args.workers
)  # bs should always be 1

# --------- Model ---------
model = CoRe_DA(
    # Feature extractor
    feat_args=dict(
        feature_dim=args.feature_dim,
        snippet_size=args.snippet_size,
        pretrained=args.pretrained,
        weight_path=args.weight_path,
        freeze_until=args.freeze_until,
        freeze_bn_stats=args.freeze_bn_stats,
        dropout_keep_prob=args.dropout_keep_prob,
    ),
    # Relative regressor
    rel_reg_args=dict(
        feature_dim=args.feature_dim,
        hidden_dim=args.rel_regressor_hidden_dim,
    ),
    # Absolute regressor
    abs_reg_args=dict(
        feature_dim=args.feature_dim,
        hidden_dim=args.abs_regressor_hidden_dim,
    ),
)


# Optimizer
param_groups = [
    {"params": model.feature_extractor.parameters(), "lr": args.lr_i3d},
    {"params": model.rel_regressor.parameters(), "lr": args.lr},
    {"params": model.abs_regressor.parameters(), "lr": args.lr},
]

optimizer = torch.optim.AdamW(param_groups, lr=args.lr)


trainer = Trainer(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    optimizer=optimizer,
    epochs=args.epochs,
    device=device,
    args=args,
)
trainer.run()
