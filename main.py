import torch
from torch.utils.data import DataLoader
from trainer import Trainer
from models import CoRe_DA
from utils import seed_everything
from dataset import CrossDomainDataset, TargetPlusExemplars
import os
import json
from opts import get_args


args = get_args()
print("Configurations:")
print(f'Use stop grad = {args.stop_grad_y_T}')
print(f"Exemplar samling method: {args.sampling_method}")


seed_everything(args.seed)

# --------- GPU ---------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("CUDA is not available.")

# --------- Path to save weights ---------
model_wts_path = os.path.join(args.base_path_wts, args.exp)
# Check and create directory if it doesn't exist
if not os.path.exists(model_wts_path):
    os.makedirs(model_wts_path)
    os.makedirs(os.path.join(model_wts_path,"wts"))
    print(f"Created directory: {model_wts_path}")
else:
    print(f"Directory already exists: {model_wts_path}")
    
# --------- Read datasets from json file ---------
data_root_dir = os.path.join(args.base_dir,'datasets')
source_db_json_path = os.path.join(args.base_dir,f'db_json/{args.source_dataset}_{args.anno}.db.json')
with open(source_db_json_path, "r") as f:
    source_db = json.load(f)
target_db_json_path = os.path.join(args.base_dir,f'db_json/{args.target_dataset}_{args.anno}.db.json')
with open(target_db_json_path, "r") as f:
    target_db = json.load(f)


# # --------- Datasets ---------
train_ds = CrossDomainDataset(
        source_db_json_file=source_db,
        target_db_json_file=target_db,
        label_column='grs',
        data_root_dir=data_root_dir,
        num_segments=args.num_segments,
        snippet_size=args.snippet_size,
        norm_stats=args.norm_stats,
        ordered_sampling=args.ordered_sampling,
        norm_labels=True, 
)

test_ds = TargetPlusExemplars(
    target_db_json_file=target_db,
    source_db_json_file=source_db,
    label_column='grs',
    data_root_dir=data_root_dir,
    num_segments=args.num_segments,
    snippet_size=args.snippet_size,
    norm_stats=args.norm_stats,
    n_exemplars=args.n_exemplars,
    sampling_method=args.sampling_method,
    gamma_alpha=0.25,     # fixed alpha
    mix_test=args.mix_test,
    norm_labels=True,
)
# --------- Dataloaders ---------
train_loader = DataLoader(train_ds, 
                        batch_size=args.batch_size, 
                        shuffle=True, 
                        num_workers=args.workers, 
                        drop_last=True)

val_loader = DataLoader(test_ds, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=args.workers)

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
    {"params": model.abs_regressor.parameters(), "lr": args.lr}
    ]

optimizer = torch.optim.AdamW(param_groups, lr=args.lr)


trainer = Trainer(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    optimizer=optimizer,
    epochs=args.epochs,
    device=device,
    model_wts_path=model_wts_path,
    args=args
)
trainer.run()