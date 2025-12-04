from functools import partial
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

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
        
    def evaluate(self, dataloader):
            self.model.eval()

            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for x, targets in dataloader:
                    x, targets = x.to(self.device), targets.to(self.device)
                    
                    preds = [(self.model(x) * self.target_std) + self.target_mean] if self.normalize else [self.model(x)]
                    avg_preds = torch.stack(preds).mean(0)

                    all_preds.append(avg_preds.cpu())
                    all_targets.append(targets.cpu())

            preds_tensor = torch.cat(all_preds, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)
            
            mse = torch.nn.functional.mse_loss(preds_tensor, targets_tensor).item()
            return {"val_MSE": mse}

    def validate(self):
        return self.evaluate(self.val_dataloader)

    def train(self, total_epochs, validation_interval):
        results = []
        best_val_loss = float('inf')
        best_model_state = deepcopy(self.model.state_dict())  

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            self.model.train()
            supervised_losses_logged = []

            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model(x)
                if self.normalize:
                    targets_norm = (targets - self.target_mean) / self.target_std
                else:
                    targets_norm = targets

                supervised_loss = self.supervised_criterion(predictions, targets_norm)
                supervised_losses_logged.append(supervised_loss.detach().item())

                supervised_loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            avg_supervised_loss = np.mean(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": avg_supervised_loss,
            }

            # Validation
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
                results.append(val_metrics["val_MSE"])

                # Save best model
                if val_metrics["val_MSE"] < best_val_loss:
                    best_val_loss = val_metrics["val_MSE"]
                    best_model_state = deepcopy(self.model.state_dict())

            self.logger.log_dict(summary_dict, step=epoch)

        # After training, load the best validation model
        self.model.load_state_dict(best_model_state)
        print(f"Best validation MSE: {best_val_loss:.6f}")

        # Evaluate on test set with the best model
        test_metrics = self.evaluate(self.test_dataloader)
        print(f"Test MSE (best val model): {test_metrics['val_MSE']:.6f}")

        return results, test_metrics
