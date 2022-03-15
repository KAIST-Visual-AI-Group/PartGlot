import hydra
import torch
import os
import os.path as osp

@hydra.main(config_path="configs", config_name="test.yaml")
def main(config):
    datamodule = hydra.utils.instantiate(config.datamodule)
    
    model = hydra.utils.instantiate(
        config.model,
        word2int=datamodule.word2int,
        total_steps=1
    )
    ckpt = torch.load(osp.join(config.work_dir, config.ckpt_path))
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    model.load_state_dict(ckpt)

    trainer = hydra.utils.instantiate(
        config.trainer, _convert_="partial"
    )

    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()




