
from itertools import chain
import hydra
import torch
from omegaconf import OmegaConf
import os
from utils import seed_everything
import models


@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    os.environ["HYDRA_FULL_ERROR"]='1'
    # print out the full config
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
    models_list = []
    if cfg.trainer.method == "ncps-train":
        # Use multiple models (4 unless overridden)
        num_models = cfg.trainer.num_models

        for _ in range(num_models):
            model = hydra.utils.instantiate(cfg.model.init).to(device)
            if cfg.compile_model:
                model = torch.compile(model)
            models_list.append(model)

    else:
        # Normal single model
        model = hydra.utils.instantiate(cfg.model.init).to(device)
        if cfg.compile_model:
            model = torch.compile(model)
        models_list = [model]
        
    if cfg.compile_model:
        model = torch.compile(model)
    models = models_list
    trainer = hydra.utils.instantiate(cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device)

    results = trainer.train(**cfg.trainer.train)
    results = torch.Tensor(results)



if __name__ == "__main__":
    main()
