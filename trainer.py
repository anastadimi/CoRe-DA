import torch
import os
from wandb_connect import *
import copy
import torch.nn.functional as F
from torchmetrics.regression import SpearmanCorrCoef
from torch.nn import L1Loss
from utils import normalize_label, AverageMeter, denormalize_relative
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:  
    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device,
        model_wts_path: str,
        args,
    ):
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.model_wts_path = model_wts_path
        self.args = args
        
        self.model.to(self.device)
        
        # Metrics
        self.metrics = {
            'scc': SpearmanCorrCoef(),
            'mae': L1Loss()
        }
        
    def run(self):

        # Wandb initialization
        init_wandb(self.args, proj_name=self.args.project, name=self.args.exp)

        # Training Loop
        for epoch in tqdm(range(self.epochs)):
            print('Training...')
            train_losses, train_dy_S_ex_metrics, train_y_S_metrics, train_y_ex_metrics = self._train_one_epoch()
            print('Validating...')
            val_y_T_rec_metrics, val_plots = self._validate()
            # prints
            print(
                f"Train - Total Loss: {train_losses['loss']:.4f}, SCC: {train_dy_S_ex_metrics['scc']:.4f} || "
                f"Val - MAE: {val_y_T_rec_metrics['mae']:.4f}, SCC: {val_y_T_rec_metrics['scc']:.4f} || "
                f"Epoch: {epoch}"
            )
            # Log metrics
            metrics_wandb = {

                f'train_losses/sup_loss_rel' : train_losses['sup_loss_rel'],
                f'train_losses/sup_loss_abs' : train_losses['sup_loss_abs'],
                f'train_losses/cons_loss_S' : train_losses['cons_loss_S'],
                f'train_losses/cons_loss_T' : train_losses['cons_loss_T'],
                f'train_losses/loss' : train_losses['loss'],
                
                f'train_metrics/scc_dy_S_ex' : train_dy_S_ex_metrics['scc'],
                f'train_metrics/mae_dy_S_ex' : train_dy_S_ex_metrics['mae'],
                f'train_metrics/scc_y_S' : train_y_S_metrics['scc'],
                f'train_metrics/mae_y_S' : train_y_S_metrics['mae'],               
                f'train_metrics/scc_y_ex' : train_y_ex_metrics['scc'],
                f'train_metrics/mae_y_ex' : train_y_ex_metrics['mae'],
                                                   
                f'val/scc_y_T_rec': val_y_T_rec_metrics['scc'],
                f'val/mae_y_T_rec': val_y_T_rec_metrics['mae'],
                
                f'val_plots/y_T_rec_scatter_plot': wandb.Image(val_plots['y_T_rec_scatter_plot']),
            }
            
            if not self.args.debug:
                model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(model_wts,
                        os.path.join(self.model_wts_path,"wts",f"epoch_{epoch}.pth")
                        )

            wandb.log(metrics_wandb)
            
    def _train_one_epoch(self):
        """
        Performs one epoch of training.
        """
        
        self.model.train()
                
        sup_loss_rel_epoch = AverageMeter('sup_loss_rel')
        sup_loss_abs_epoch = AverageMeter('sup_loss_abs')
        cons_loss_S_epoch = AverageMeter('cons_loss_S')
        cons_loss_T_epoch = AverageMeter('cons_loss_T')
        loss_epoch = AverageMeter('loss')
        
        dy_S_ex_stack = []
        dy_S_ex_hat_stack = []
        y_S_stack = []
        y_S_hat_stack = []
        y_ex_stack = []
        y_ex_hat_stack = []
        
        for train_batch in tqdm(self.train_loader):
            x_S, y_S = train_batch['S']
            x_ex, y_ex = train_batch['ex']
            x_T = train_batch['T']

            x_S, y_S, x_ex, y_ex = x_S.to(self.device), y_S.to(self.device), x_ex.to(self.device), y_ex.to(self.device)
            dy_S_ex = y_S - y_ex
            x_T = x_T.to(self.device)
            
            self.optimizer.zero_grad()
            
            dy_S_ex_hat, dy_T_ex_hat, y_S_hat, y_T_hat, y_ex_hat = self.model(x_S=x_S,x_T=x_T,x_ex=x_ex,mode='train')
                
            sup_loss_rel = F.mse_loss(dy_S_ex_hat, dy_S_ex.unsqueeze(dim=1))
            sup_loss_abs = F.mse_loss(y_ex_hat,y_ex.unsqueeze(dim=1))
            cons_loss_S = F.mse_loss(dy_S_ex_hat+y_ex_hat,y_S_hat)
            cons_loss_T = F.mse_loss(dy_T_ex_hat+y_ex_hat,y_T_hat.detach() if self.args.stop_grad_y_T else y_T_hat)
            
            loss = (self.args.coef_alpha*(sup_loss_rel+sup_loss_abs)
                    + self.args.coef_beta*cons_loss_S
                    + self.args.coef_gamma*cons_loss_T
            )

            sup_loss_rel_epoch.update(sup_loss_rel.item(), x_S.size(0))
            sup_loss_abs_epoch.update(sup_loss_abs.item(), x_S.size(0))
            cons_loss_S_epoch.update(cons_loss_S.item(), x_S.size(0))
            cons_loss_T_epoch.update(cons_loss_T.item(), x_S.size(0))
            loss_epoch.update(loss.item(), x_S.size(0))
                        
            loss.backward()
            self.optimizer.step()
            
            # Track dy_S_ex_hat, y_s_hat, y_ex_hat
            dy_S_ex_stack.append(dy_S_ex) # labels
            dy_S_ex_hat_stack.append(dy_S_ex_hat) # predictions
            y_S_stack.append(y_S) # labels
            y_S_hat_stack.append(y_S_hat) # predictions
            y_ex_stack.append(y_ex) # labels
            y_ex_hat_stack.append(y_ex_hat) # predictions

                                
        dy_S_ex_stack = torch.cat(dy_S_ex_stack)
        dy_S_ex_hat_stack = torch.cat(dy_S_ex_hat_stack)
        y_S_stack = torch.cat(y_S_stack)
        y_S_hat_stack = torch.cat(y_S_hat_stack)            
        y_ex_stack = torch.cat(y_ex_stack)
        y_ex_hat_stack = torch.cat(y_ex_hat_stack)   
                
        # Compute metrics
        dy_S_ex_metrics = self._compute_metrics(dy_S_ex_hat_stack.squeeze(dim=1),dy_S_ex_stack,mode='rel')
        y_S_metrics = self._compute_metrics(y_S_hat_stack.squeeze(dim=1),y_S_stack)
        y_ex_metrics = self._compute_metrics(y_ex_hat_stack.squeeze(dim=1),y_ex_stack)
        epoch_losses = {
                        "sup_loss_rel": sup_loss_rel_epoch.avg,
                        "sup_loss_abs": sup_loss_abs_epoch.avg,
                        "cons_loss_S": cons_loss_S_epoch.avg,
                        "cons_loss_T": cons_loss_T_epoch.avg,
                        "loss": loss_epoch.avg
                        }
        
        return epoch_losses, dy_S_ex_metrics, y_S_metrics, y_ex_metrics

    
    def _validate(self):

        self.model.eval()
        y_T_hat_stack = []
        y_T_hat_rec_mean_stack = []
        y_T_stack_all = []
        y_T_hat_rec_stack_for_analysis = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                
                (x_T_stack, y_T_stack), (x_ex_stack, y_ex_stack) = batch # videos and labels for target and exemplars
                
                y_T_hat_rec_stack = []
                y_T_hat_abs_stack = []
                for k in range(self.args.n_exemplars):
                    x_T = x_T_stack[k].to(self.device) # one target video
                    y_T = y_T_stack[k].to(self.device) # one target label
                    
                    x_ex = x_ex_stack[k].to(self.device) # one examplar video
                    y_ex = y_ex_stack[k].to(self.device) # one examplar label
                    dy_T_ex_hat, y_T_hat = self.model(x_T=x_T.permute(0,2,1,3,4),x_ex=x_ex.permute(0,2,1,3,4),mode='test')
                    y_T_hat_rec = dy_T_ex_hat + y_ex.unsqueeze(dim=1) # y_T_hat - y_ex_hat + y_ex = y_T_hat                    
                    y_T_hat_rec_stack.append(y_T_hat_rec)
                    y_T_hat_abs_stack.append(y_T_hat)
                    
                y_T_hat_rec_stack = torch.cat(y_T_hat_rec_stack,dim=1) # (B,n_exemplars)
                y_T_hat_abs_stack = torch.cat(y_T_hat_abs_stack,dim=1) # (B,n_exemplars)
                y_T_hat_rec_mean = y_T_hat_rec_stack.mean(dim=1)
                y_T_hat_abs_mean = y_T_hat_abs_stack.mean(dim=1)
                
                y_T_stack_all.append(y_T)
    
                y_T_hat_stack.append(y_T_hat_abs_mean) # aggregate target predictions from abs regressor
                y_T_hat_rec_mean_stack.append(y_T_hat_rec_mean) # aggregate target predictions from rel regressor from all exemplars
                
                y_T_hat_rec_stack_for_analysis.append(y_T_hat_rec_stack)

        y_T_hat_stack = torch.cat(y_T_hat_stack)
        y_T_hat_rec_mean_stack = torch.cat(y_T_hat_rec_mean_stack)
        y_T_stack_all = torch.cat(y_T_stack_all)
	
        y_T_hat_rec_stack_for_analysis = torch.cat(y_T_hat_rec_stack_for_analysis,dim=0) # (N,n_exemplars)

        # Compute metrics
        y_T_rec_metrics = self._compute_metrics(y_T_hat_rec_mean_stack,y_T_stack_all)
        
        
        # Plots
        y_T_rec_scatter_plot = self._create_scatter_plot(y_T_hat_rec_mean_stack,y_T_stack_all,name='y_T_rec')
        val_plots = {
                    "y_T_rec_scatter_plot": y_T_rec_scatter_plot,
                     }       
        return y_T_rec_metrics, val_plots

    
    def _compute_metrics(self,preds,targets,mode='abs'):

        if self.args.norm_labels: # if model output and targets are normalized, denormalize them
            if mode == 'abs':
                preds = normalize_label(preds, label_column=self.args.label_column, denormalize=True)
                targets = normalize_label(targets, label_column=self.args.label_column, denormalize=True)
            elif mode == 'rel':
                preds = denormalize_relative(preds)
                targets = denormalize_relative(targets)
            else:
                raise ValueError(f'Not implemented mode {mode}')
        else:
            pass
        scc = self.metrics['scc'](preds,targets)
        mae = self.metrics['mae'](preds,targets)
        return {
            'scc': scc,
            'mae': mae
        }
        
    def _create_scatter_plot(self,preds,targets,name=''):
        preds = normalize_label(preds, label_column=self.args.label_column, denormalize=True)
        targets = normalize_label(targets, label_column=self.args.label_column, denormalize=True)
    
        y_true = targets.cpu().numpy()
        y_pred = preds.cpu().numpy()
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Predicted vs True Scatter for {name}")
        min_val = 0
            
        max_val = 30
        plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
        
        
        return fig