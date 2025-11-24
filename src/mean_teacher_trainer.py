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
    ):
        self.device = device
        self.teacher_model = models[0]
        self.student_model = models[1]
        self.ema_decay = 0.99

        # Optim related things
        self.supervised_criterion = supervised_criterion
        # all_params = [p for m in self.models for p in m.parameters()]

        for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            teacher_param.data.copy_(student_param.data)

        self.optimizer = optimizer(params=self.student_model.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        self.un_train_dataloader = datamodule.unsupervised_train_dataloader()

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
                targets = targets.to(self.device)

                # Teacher predictions
                teacher_pred = self.teacher_model(x)
                teacher_loss = torch.nn.functional.mse_loss(teacher_pred, targets)
                teacher_losses.append(teacher_loss.item())

                # Student predictions
                student_pred = self.student_model(x)
                student_loss = torch.nn.functional.mse_loss(student_pred, targets)
                student_losses.append(student_loss.item())

        return {
            "val_MSE_teacher": float(np.mean(teacher_losses)),
            "val_MSE_student": float(np.mean(student_losses)),
        }


    def train(self, total_epochs, validation_interval):
        #self.logger.log_dict()
        results = []
        gamma_end = 0.3
        self.consistency_loss_fn = torch.nn.MSELoss()
        self.optimizer.zero_grad()
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            total_supervised = 0
            total_consistency = 0

            unsupervised_loader_iterator = (
                iter(self.un_train_dataloader) if self.un_train_dataloader else None
            )

            self.student_model.train()
            r = sigmoid_rampup(epoch, total_epochs * 0.3)
            gamma = gamma_end * r
            for (x_s, targets) in self.train_dataloader:
                unsupervised_loader_iterator, x_u = self._next_unsupervised_batch(unsupervised_loader_iterator)
                x_s = x_s.to(self.device)
                targets = targets.to(self.device)
                x_u = x_u[0].to(self.device)
                
                # change the noise function and its done
                x_s_noisy = add_noise(x_s)
                x_u_noisy = add_noise(x_u)

                pred_s = self.student_model(x_s_noisy)
                pred_u = self.student_model(x_u_noisy)

                with torch.no_grad():
                    teacher_pred = self.teacher_model(x_u)
                # Supervised loss
                class_loss = self.supervised_criterion(pred_s, targets)

                consistency_loss = self.consistency_loss_fn(pred_u, teacher_pred)

                total_loss = (1 - gamma) * class_loss + gamma * consistency_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    for t_param, s_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
                        t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=(1 - self.ema_decay))


                total_supervised += class_loss.item()
                total_consistency += consistency_loss.item()

            self.scheduler.step()

            summary_dict = {
                    "supervised_loss": total_supervised / len(self.train_dataloader),
                    "consistency_loss": total_consistency / len(self.train_dataloader),
                }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
                results.append(val_metrics["val_MSE_teacher"])
                results.append(val_metrics["val_MSE_student"])

            self.logger.log_dict(summary_dict, step=epoch)

        return results
    
    def _next_unsupervised_batch(self, unsupervised_loader_iterator):
        """
        Advance the unsupervised loader iterator and return the (iterator, batch).

        Returns (None, None) if the provided iterator is None. If the iterator is
        exhausted, it is restarted from self.unsupervised_train_dataloader.
        """
        if unsupervised_loader_iterator is None:
            return None, None
        try:
            x_unlabeled = next(unsupervised_loader_iterator)
        except StopIteration:  # Restart
            unsupervised_loader_iterator = iter(self.un_train_dataloader)
            x_unlabeled = next(unsupervised_loader_iterator)
        return unsupervised_loader_iterator, x_unlabeled
    
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def add_noise(data, sigma=0.1):
    noisy = data.clone()
    noisy.pos = data.pos + sigma * torch.randn_like(data.pos)
    return noisy