import os

import yaml

import wandb


def init_wandb(args_param, proj_name, name):
    if args_param.debug:
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(
        # Set the project where this run will be logged
        project=proj_name,
        # We pass a run name (otherwise it will be randomly assigned, like sunshine-lollypop-10)
        name=name,
        # Track hyperparameters and run metadata
        settings=wandb.Settings(start_method="fork"),
        config=args_param,
    )


def connect_wandb(args_param, proj_name, name):

    with open("../wandb_token/token.yaml", "r") as f:
        token = yaml.safe_load(f)["token"]

        wandb.login(key=token)
        init_wandb(args_param, proj_name, name)
