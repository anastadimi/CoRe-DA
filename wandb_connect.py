import wandb
import yaml
import os

def init_wandb(args_param, proj_name, name):
    if args_param.debug:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(
        project=proj_name,
        name=name,
        settings=wandb.Settings(start_method="fork"),
        config=args_param)
    
    
def connect_wandb(args_param, proj_name, name):
    
    with open('../wandb_token/token.yaml', 'r') as f:
        token = yaml.safe_load(f)['token']
        
        wandb.login(key=token)
        init_wandb(args_param, proj_name, name)
        