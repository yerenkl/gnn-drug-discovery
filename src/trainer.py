from functools import partial

import numpy as np
import torch
from tqdm import tqdm

class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        normalize=False,
    ):
        self.device = device

        self.supervised_criterion = supervised_criterion
        self.model = models[0]
        self.normalize = normalize
        all_params = [p for p in self.model.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # std and mean for normalization    
        all_targets = []
        with torch.no_grad():
            for _, targets in tqdm(self.train_dataloader):
                all_targets.append(targets.float().cpu())
            targets_tensor = torch.cat(all_targets, dim=0)
            mean_val = targets_tensor.mean().item()
            std_val = targets_tensor.std().item()
        self.target_mean = torch.tensor(mean_val, dtype=torch.float32, device=self.device)
        self.target_std = torch.tensor(std_val, dtype=torch.float32, device=self.device)

        # Logging
        self.logger = logger

    def validate(self):
        self.model.eval()

        val_losses = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction
                preds = [(self.model(x) * self.target_std) + self.target_mean] if self.normalize else [self.model(x)]
                avg_preds = torch.stack(preds).mean(0)
                
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs, validation_interval):
        #self.logger.log_dict()
        results = []
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            self.model.train()
            supervised_losses_logged = []
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                # Supervised loss
                predictions = self.model(x)
                if self.normalize:
                    targets = (targets - self.target_mean) / self.target_std
                supervised_losses = [self.supervised_criterion(predictions, targets)]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item())  
                loss = supervised_loss
                loss.backward() 
                self.optimizer.step()
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
                results.append(val_metrics["val_MSE"])
            self.logger.log_dict(summary_dict, step=epoch)
        return results
