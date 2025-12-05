# GNN Introduction

This project provides an introduction to Graph Neural Networks (GNNs) using PyTorch and PyTorch Geometric on the dataset QM9.

## Installation

To run this project, you need to install the required Python packages. You can install them using pip:

```bash
# It is recommended to install PyTorch first, following the official instructions
# for your specific hardware (CPU or GPU with a specific CUDA version).
# See: https://pytorch.org/get-started/locally/

# For example, for a recent CUDA version:
# pip install torch torchvision torchaudio

# Or for CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# After installing PyTorch, install PyTorch Geometric.
# The exact command depends on your PyTorch and CUDA versions.
# Please refer to the PyTorch Geometric installation guide:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# Example for PyTorch 2.7 and CUDA 11.8
# pip install torch_geometric

# Then, install the other required packages:
pip install hydra-core omegaconf wandb pytorch-lightning numpy tqdm
```

## How to Run

The main entry point for this project is `src/run.py`. It uses `hydra` for configuration management.

To run the code, execute the following command from the root of the project:

```bash
python src/run.py
```

You can override the default configuration by passing arguments from the command line. For example, to use a different model configuration:

```bash
python src/run.py model=gcn
```

The configuration files are located in the `configs/` directory.


## Mean Teacher (Semi-Supervised)

Run the **Mean Teacher** semi-supervised training pipeline:

```bash
python ./src/run.py trainer=mean-teacher-train
```

This loads:

```
configs/trainer/mean-teacher-train.yaml
```
| Key                                  | Value                                         | Description                                                                   |
| ------------------------------------ | --------------------------------------------- | ----------------------------------------------------------------------------- |
| `train.total_epochs`                 | `100`                                         | Total number of training epochs.                                              |
| `train.validation_interval`          | `10`                                          | Run validation every N epochs.                                                |
| `init.noise_std`                     | `0.02`                                        | Standard deviation of Gaussian noise added to student inputs.                 |
| `init.gamma_end`                     | `0.1`                                         | Final value of gamma for ramp scheduling (e.g., consistency weight).          |
| `init.normalize`                     | `true`                                        | If true, normalizes model outputs/targets.                                    |
| `init.alpha`                         | `0.99`                                        | EMA decay for the teacher model update.                                       |
| `init.ramp_type`                     | `sigmoid`                                     | Ramp-up function type for consistency loss weighting.                         |
| `init.supervised_criterion._target_` | `torch.nn.MSELoss`                            | Criterion used for the supervised component of the loss.                      |
| `init.optimizer._target_`            | `torch.optim.AdamW`                           | Optimizer class used for training.                                            |
| `init.optimizer._partial_`           | `true`                                        | Indicates Hydra should pass parameters later when constructing the optimizer. |
| `init.optimizer.lr`                  | `0.001`                                       | Learning rate for the optimizer.                                              |
| `init.optimizer.weight_decay`        | `0.005`                                       | Weight decay (L2 regularization).                                             |
| `init.scheduler._target_`            | `torch.optim.lr_scheduler.StepLR`             | Scheduler class for adjusting learning rate.                                  |
| `init.scheduler._partial_`           | `true`                                        | Hydra will pass parameters later when constructing the scheduler.             |
| `init.scheduler.step_size`           | `1`                                           | Step interval for reducing the learning rate.                                 |
| `init.scheduler.gamma`               | `0.975`                                       | Multiplicative LR decay factor applied at each step.                          |
---

## N-CPS Training (Semi-Supervised) 

Run the **NCPS** training pipeline:

```bash
python ./src/run.py trainer=ncps-train
```

This loads:

```
configs/trainer/ncps-train.yaml
```

---

## Fully Supervised Baseline

Run the **fully supervised** baseline training mode:

```bash
python ./src/run.py trainer=fully-supervised-train
```

This loads:

```
configs/trainer/fully-supervised-train.yaml
```

