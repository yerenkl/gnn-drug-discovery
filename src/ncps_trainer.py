# src/trainer.py

from typing import List, Optional, Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class SemiSupervisedEnsemble:
    """
    Semi-supervised ensemble trainer with n-CPS style consistency.

    - Config (Hydra) provides: supervised_criterion, optimizer (partial),
      scheduler (partial), unsupervised_weight.
    - run.py (or Hydra instantiate) provides: models, dataloaders, logger, device.
    - If num_models == 1 or unsupervised_weight == 0.0 -> pure supervised.
    - If num_models >= 2 and unsupervised_weight > 0.0 -> semi-supervised n-CPS.
    - Targets are normalized (z-score) using the labeled train set.
      Training uses normalized targets; val_MSE is in original units.
    """

    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        datamodule,
        device,
        models,
        logger,
        unsupervised_weight: float = 1.0,
        **kwargs,
    ):
        # From config
        self.supervised_criterion = supervised_criterion
        # Optimizer and scheduler are usually hydra partials; we call them later
        self._optimizer_ctor = optimizer
        self._scheduler_ctor = scheduler
        self.unsupervised_weight = float(unsupervised_weight)

        # From run.py or Hydra kwargs
        self.models: List[torch.nn.Module] = models if models is not None else []
        self.train_dataloader = datamodule.train_dataloader()
        self.unsup_train_dataloader = datamodule.unsupervised_train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.logger = logger
        self.device = device if device is not None else torch.device("cpu")

        # Will be created lazily
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None

        # Target normalization (mean/std in original units)
        self.y_mean: Optional[torch.Tensor] = None
        self.y_std: Optional[torch.Tensor] = None

        # If everything is already provided at init-time, we can set up now.
        if self.models and self.train_dataloader is not None:
            self._post_init_setup()

    # ----------------- Internal helpers -----------------

    def _post_init_setup(self):
        """Set up optimizer, scheduler, device, and target normalization."""
        # Move models to device
        for m in self.models:
            m.to(self.device)

        # Build optimizer if not already built
        if self.optimizer is None and self._optimizer_ctor is not None:
            params = [p for m in self.models for p in m.parameters()]
            self.optimizer = self._optimizer_ctor(params=params)

        # Build scheduler if not already built
        if self.scheduler is None and self._scheduler_ctor is not None:
            self.scheduler = self._scheduler_ctor(self.optimizer)

        # Compute normalization stats if not already set
        if self.y_mean is None or self.y_std is None:
            if self.train_dataloader is None:
                raise ValueError(
                    "train_dataloader is required to compute target normalization stats."
                )
            mean, std = self._compute_target_stats(self.train_dataloader)
            if std <= 0.0:  # safety
                std = 1.0
            self.y_mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
            self.y_std = torch.tensor(std, dtype=torch.float32, device=self.device)
            print(
                f"[SemiSupervisedEnsemble] Target normalization: "
                f"mean={mean:.6f}, std={std:.6f}"
            )

    @staticmethod
    @torch.no_grad()
    def _compute_target_stats(dataloader: DataLoader) -> (float, float):
        """
        Compute mean and std of the training targets.

        Assumes dataloader yields (x, targets) where targets is the scalar
        QM9 property you care about (target=2) already selected by the datamodule.
        """
        all_targets = []
        for _, targets in dataloader:
            all_targets.append(targets.view(-1).float())
        y = torch.cat(all_targets, dim=0)
        mean = y.mean().item()
        std = y.std(unbiased=False).item() 
        return mean, std

    def train(self, total_epochs: int, validation_interval: int) -> None:
        """
        Main training loop.

        - For labeled data: supervised MSE in normalized target space.
        - For unlabeled data (if enabled): n-CPS consistency in normalized space.
        - Logs supervised_loss, unsupervised_loss, and val_MSE.
        """
        if not self.models:
            raise ValueError("No models provided to SemiSupervisedEnsemble.")
        if self.train_dataloader is None or self.val_dataloader is None:
            raise ValueError("train_dataloader and val_dataloader must be set.")

        # Ensure optimizer, scheduler, normalization, and devices are ready
        self._post_init_setup()

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()

            supervised_losses_logged: List[float] = []
            unsupervised_losses_logged: List[float] = []

            # Use unsupervised loss only if we have >=2 models, weight > 0,
            # and an unlabeled dataloader is available.
            use_unsup = (
                self.unsup_train_dataloader is not None
                and len(self.models) > 1
                and self.unsupervised_weight > 0.0
            )

            if use_unsup:
                unsup_iter = iter(self.unsup_train_dataloader)
            else:
                unsup_iter = None

            # ---------- Labeled loop ----------
            for x_labeled, targets in self.train_dataloader:
                x_labeled = x_labeled.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()

                # Normalize targets: (y - mean) / std
                targets_norm = (targets - self.y_mean) / self.y_std

                # ----- 1) Supervised loss on labeled data -----
                sup_preds_norm = [model(x_labeled) for model in self.models]
                sup_losses = [
                    self.supervised_criterion(pred, targets_norm)
                    for pred in sup_preds_norm
                ]
                supervised_loss = sum(sup_losses) / len(self.models)
                supervised_losses_logged.append(supervised_loss.detach().item())

                # ----- 2) Optional n-CPS on unlabeled data -----
                if use_unsup:
                    try:
                        x_unlabeled, _ = next(unsup_iter)
                    except StopIteration:
                        unsup_iter = iter(self.unsup_train_dataloader)
                        x_unlabeled, _ = next(unsup_iter)
                    x_unlabeled = x_unlabeled.to(self.device)

                    # predictions in normalized space
                    unsup_preds_norm = [model(x_unlabeled) for model in self.models]

                    # Build teacher targets as mean of other models (normalized space)
                    teacher_targets_norm = []
                    with torch.no_grad():
                        for i in range(len(self.models)):
                            others = [
                                p.detach()
                                for j, p in enumerate(unsup_preds_norm)
                                if j != i
                            ]
                            # len(others) >= 1 here because len(models) > 1
                            teacher_targets_norm.append(
                                torch.stack(others).mean(0)
                            )

                    unsup_losses = [
                        torch.nn.functional.mse_loss(
                            unsup_preds_norm[i], teacher_targets_norm[i]
                        )
                        for i in range(len(self.models))
                    ]
                    unsupervised_loss = sum(unsup_losses) / len(self.models)
                    unsupervised_losses_logged.append(
                        unsupervised_loss.detach().item()
                    )
                else:
                    # No unsupervised loss when only 1 model or weight == 0
                    unsupervised_loss = torch.tensor(0.0, device=self.device)

                # ----- 3) Total loss -----
                loss = supervised_loss + self.unsupervised_weight * unsupervised_loss
                loss.backward()
                self.optimizer.step()

            # Step LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Epoch-level logging
            supervised_losses_logged = float(np.mean(supervised_losses_logged))
            unsupervised_losses_logged = (
                float(np.mean(unsupervised_losses_logged))
                if len(unsupervised_losses_logged) > 0
                else 0.0
            )

            summary_dict: Dict[str, float] = {
                "supervised_loss": supervised_losses_logged,
                "unsupervised_loss": unsupervised_losses_logged,
            }

            # Validation
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)

            # Log through the provided logger (e.g., WandBLogger or dummy)
            if self.logger is not None:
                self.logger.log_dict(summary_dict, step=epoch)

    def validate(self) -> Dict[str, float]:
        """
        Validation loop.

        - Uses ensemble-mean prediction in normalized space.
        - Unnormalizes predictions back to original units.
        - Computes MSE in original units (what you see as val_MSE).
        """
        if not self.models:
            raise ValueError("No models provided to SemiSupervisedEnsemble.")
        if self.val_dataloader is None:
            raise ValueError("val_dataloader must be set.")

        # Make sure normalization is ready (in case validate() is called alone)
        if self.y_mean is None or self.y_std is None:
            if self.train_dataloader is None:
                raise ValueError(
                    "train_dataloader is required to compute normalization stats."
                )
            self._post_init_setup()

        for model in self.models:
            model.eval()

        val_losses: List[float] = []

        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x = x.to(self.device)
                targets = targets.to(self.device)

                # predictions in normalized space
                preds_norm = [model(x) for model in self.models]
                avg_preds_norm = torch.stack(preds_norm).mean(0)

                # unnormalize back to original target units
                avg_preds = avg_preds_norm * self.y_std + self.y_mean

                # MSE in original units
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())

        val_loss = float(np.mean(val_losses))
        return {"val_MSE": val_loss}