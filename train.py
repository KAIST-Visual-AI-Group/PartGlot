from pytorch_lightning import seed_everything
import hydra
import os.path as osp
from partglot.utils.simple_utils import print_config


@hydra.main(config_path="configs", config_name="train.yaml")
def main(config):
    print_config(config)

    datamodule = hydra.utils.instantiate(config.datamodule)

    if config.get("seed"):
        seed_everything(config.seed)

    model = hydra.utils.instantiate(
        config.model,
        word2int=datamodule.word2int,
        total_steps=config.epochs * len(datamodule.train_dataloader()),
    )

    callbacks = []
    if config.get("callbacks"):
        for _, cb_conf in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))

    logger = []
    if config.get("logger"):
        for _, lg_conf in config.logger.items():
            logger.append(hydra.utils.instantiate(lg_conf))

    trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
        deterministic=True,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
