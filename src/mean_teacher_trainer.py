from functools import partial
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy, copy
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        gamma_end=0.1,
        alpha=0.99,
        noise_std=0.02,
        ramp_type='sigmoid',
        normalize=False,
    ):
        self.device = device
        self.student_model = models[0]
        self.teacher_model = deepcopy(self.student_model).to(self.device)
        self.gamma_end = gamma_end
        self.alpha = alpha  
        self.supervised_criterion = supervised_criterion
        self.noise_std = noise_std
        self.ramp_type = ramp_type
        self.normalize = normalize

        for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            teacher_param.data.copy_(student_param.data)

        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self.teacher_model.eval() 

        self.optimizer = optimizer(params=self.student_model.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        self.un_train_dataloader = datamodule.unsupervised_train_dataloader()

        # Calculate target mean and std for normalization
        all_targets = []
        with torch.no_grad():
            for _, targets in tqdm(self.train_dataloader, desc="Calculating stats"):
                all_targets.append(targets.float().cpu())
            targets_tensor = torch.cat(all_targets, dim=0)
            mean_val = targets_tensor.mean().item()
            std_val = targets_tensor.std().item()

        self.target_mean = torch.tensor(mean_val, dtype=torch.float32, device=self.device)
        self.target_std = torch.tensor(max(std_val, 1e-6), dtype=torch.float32, device=self.device)
        print(f"Target Mean: {self.target_mean.item():.4f}, Target Std: {self.target_std.item():.4f}")

        # Logging
        self.logger = logger
        
    def validate(self):
        self.teacher_model.eval()
        self.student_model.eval()

        teacher_losses = []
        student_losses = []

        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x = x.to(self.device)
                targets = targets.to(self.device).float()

                
                teacher_pred = (self.teacher_model(x) * self.target_std) + self.target_mean if self.normalize else self.teacher_model(x)
                student_pred = (self.student_model(x) * self.target_std) + self.target_mean if self.normalize else self.student_model(x)

                teacher_loss = torch.nn.functional.mse_loss(teacher_pred, targets)
                teacher_losses.append(teacher_loss.item())

                student_loss = torch.nn.functional.mse_loss(student_pred, targets)
                student_losses.append(student_loss.item())

        return {
            "val_MSE_teacher": float(np.mean(teacher_losses)),
            "val_MSE_student": float(np.mean(student_losses)),
        }

    def train(self, total_epochs, validation_interval):
        results = []
        self.consistency_loss_fn = torch.nn.MSELoss()
        
        ramp_epochs = int(total_epochs * 0.3)

        # Initialize iterator safely
        if self.un_train_dataloader:
            unsupervised_loader_iterator = iter(self.un_train_dataloader)
        else:
            unsupervised_loader_iterator = None

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            total_supervised = 0
            total_consistency = 0

            self.student_model.train()
            self.teacher_model.eval()
            
            if self.ramp_type == 'sigmoid':
                rampup_val = sigmoid_rampup(epoch, ramp_epochs)
            elif self.ramp_type == 'linear':
                rampup_val = np.clip(epoch / ramp_epochs, 0.0, 1.0)
            elif self.ramp_type == 'cosine':
                rampup_val = cosine_rampdown(epoch, ramp_epochs)
            else:
                rampup_val = 1.0

            for (x_s, targets) in self.train_dataloader:
                self.optimizer.zero_grad()
                
                try:
                    x_u, _ = next(unsupervised_loader_iterator)
                except StopIteration:
                    unsupervised_loader_iterator = iter(self.un_train_dataloader)
                    x_u, _ = next(unsupervised_loader_iterator)
                    
                x_s, x_u, targets = x_s.to(self.device), x_u.to(self.device), targets.to(self.device)
                if self.normalize==True:
                    targets = (targets - self.target_mean) / self.target_std
                x_u_noisy = add_noise_(x_u, sigma=self.noise_std)

                pred_s = self.student_model(x_s)
                pred_u = self.student_model(x_u_noisy)

                with torch.no_grad():
                    teacher_pred = self.teacher_model(x_u)

                class_loss = self.supervised_criterion(pred_s, targets)
                consistency_loss = self.consistency_loss_fn(pred_u, teacher_pred)

                total_loss = class_loss + self.gamma_end * rampup_val * consistency_loss

                total_loss.backward()
                self.optimizer.step()

                # EMA Update
                with torch.no_grad():
                    for t_param, s_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
                        t_param.data.mul_(self.alpha).add_(s_param.data, alpha=(1 - self.alpha))
                        
                    for t_buffer, s_buffer in zip(self.teacher_model.buffers(), self.student_model.buffers()):
                        t_buffer.data.mul_(self.alpha).add_(s_buffer.data, alpha=1.0 - self.alpha)

                total_supervised += class_loss.item()
                total_consistency += consistency_loss.item()

            avg_sup_loss = total_supervised / len(self.train_dataloader)
            avg_cons_loss = total_consistency / len(self.train_dataloader)

            print(f"Epoch {epoch}: Sup Loss {avg_sup_loss:.4f}, Cons Loss {avg_cons_loss:.4f}")

            summary_dict = {
                "supervised_loss": avg_sup_loss,
                "consistency_loss": avg_cons_loss,
            }

            val_metrics = {}
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
                results.append(val_metrics["val_MSE_teacher"])
                results.append(val_metrics["val_MSE_student"])

            self.logger.log_dict(summary_dict, step=epoch)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                if 'val_MSE_teacher' in val_metrics:
                    self.scheduler.step(val_metrics['val_MSE_teacher'])
            else:
                self.scheduler.step()

        return results
    
def add_noise_(data, sigma):
    noisy_data = copy(data)
    noisy_data.pos = noisy_data.pos + sigma * torch.randn_like(noisy_data.pos)
    return noisy_data

def sigmoid_rampup(current, rampup_length): 
    if rampup_length == 0: return 1.0 
    current = np.clip(current, 0.0, rampup_length) 
    phase = 1.0 - current / rampup_length 
    return float(np.exp(-5.0 * phase ** 2))

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))