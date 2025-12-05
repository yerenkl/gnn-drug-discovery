# src/run.py

import hydra
import torch
from pathlib import Path

from omegaconf import OmegaConf

from utils import seed_everything
import os

@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    os.environ["HYDRA_FULL_ERROR"]='1'
    print(OmegaConf.to_yaml(cfg))

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    seed_everything(cfg.seed, cfg.force_deterministic)

    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)

    dm = hydra.utils.instantiate(cfg.dataset.init)

    num_models = 4

    models = []
    for _ in range(num_models):
        model = hydra.utils.instantiate(cfg.model.init).to(device)
        if cfg.compile_model:
            model = torch.compile(model)
        models.append(model)

    trainer = hydra.utils.instantiate(
        cfg.trainer.init,
        models=models,
        datamodule=dm,
        logger=logger,
        device=device,
    )

    trainer.train(**cfg.trainer.train)

    project_root = Path(__file__).resolve().parents[1]  
    save_dir = project_root / cfg.result_dir / cfg.model.name / f"seed={cfg.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    num_models = len(trainer.models)
    for i, model in enumerate(trainer.models):
        fname = "model.pt" if num_models == 1 else f"model_{i}.pt"
        save_path = save_dir / fname
        torch.save(model.state_dict(), save_path)
        print(f"Saved model {i} to {save_path}")


if __name__ == "__main__":
    main()