from typing import List, Union, Type
from numpy import random
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from partglot.datamodules.datasets.partglot_dataset import PartglotDataset
from partglot.utils.simple_utils import unpickle_data
from partglot.datamodules.data_utils import split_indices_with_unseen_target_geo_in_test
from partglot.datamodules.data_utils import part_names
import os.path as osp
import os
import numpy as np
import h5py


class PartglotDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        only_correct: bool,
        only_easy_context: bool,
        max_seq_len: int,
        only_one_part_name: bool,
        seed: int,
        split_sizes: List[float],
        balance: bool,
        data_dir: str,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train = self.data_val = self.data_test = None
        self.prepare_data()

    def prepare_data(self):
        (
            self.game_data,
            self.word2int,
            self.int2word,
            self.int2sn,
            self.sn2int,
            self.sorted_sn,
        ) = unpickle_data(osp.join(self.hparams.data_dir, "game_data.pkl"))
        self.h5_data = h5py.File(
            osp.join(self.hparams.data_dir, "cic_bsp.h5"), "r"
        )

    def setup(self, stage=None):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = PartglotDataset(
                self.hparams, self.game_data, self.h5_data, self.word2int
            )
            split_ids = split_indices_with_unseen_target_geo_in_test(
                self.hparams.split_sizes, dataset.geo_ids, dataset.labels, seed = 2004
            )

            for i, s in enumerate(["train", "val", "test"]):
                unq = dataset.geo_ids[split_ids[s]]
                unq = np.unique(unq.flatten())
                print(f"{s} unique shape:", len(unq))

            self.data_train = Subset(dataset, split_ids["train"])
            self.data_val = Subset(dataset, split_ids["val"])
            self.data_test = Subset(dataset, split_ids["test"])

    def _build_dataloader(self, mode: str):
        self.setup()
        ds = getattr(self, "data_" + mode) 

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        if self.hparams.balance and mode == "train":
            sampler = self.get_balance_sampler(ds)

            return DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=False,
                drop_last=True,
                sampler=sampler,
                worker_init_fn=seed_worker,
                generator=g,
            )
        else:
            return DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                shuffle=mode == "train",
                num_workers=4,
                pin_memory=False,
                drop_last=mode == "train",
                worker_init_fn=seed_worker,
                generator=g,
            )
    def train_dataloader(self):
        return self._build_dataloader("train")

    def val_dataloader(self):
        return self._build_dataloader("val")

    def test_dataloader(self):
        return self._build_dataloader("test")

    def get_balance_sampler(self, ds: Union[Type[Dataset], Subset]):
        """
        Weights of Weighted Sampler are inversely proportional to the number of each part utterances.
        """
        # TODO: check Type[Dataset]

        total_ds_len = len(ds)
        weights = np.zeros(total_ds_len)

        if isinstance(ds, Subset):
            part_ind = ds.dataset.part_indicator[ds.indices]
        else:
            part_ind = ds.part_indicator

        num_example_per_part = np.sum(part_ind, axis=0)
        # fmt = f"total: {num_example_per_part.sum()}"
        # for i in range(4):
            # fmt += f"{part_names[i]}: {num_example_per_part[i]} "
        # print(fmt)
        part_ind = np.argmax(part_ind, axis=1)

        assert len(part_ind) == total_ds_len and len(num_example_per_part) == len(
            part_names
        )

        for i in range(len(part_names)):
            indices_per_part = (part_ind == i).nonzero()
            weights[indices_per_part] = 1 / num_example_per_part[i]

        g = torch.Generator()
        g.manual_seed(0)

        return WeightedRandomSampler(list(weights), len(ds), replacement=True, generator=g)
