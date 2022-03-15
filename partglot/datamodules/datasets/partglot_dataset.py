import numpy as np
import h5py
import os.path as osp
import torch
from torch.utils.data import Dataset
from partglot.utils.simple_utils import unpickle_data
from partglot.datamodules.data_utils import (
    part_names,
    convert_labels_to_one_hot,
    shuffle_game_geometries,
    pad_text_symbols_with_zeros,
    get_mask_of_game_data,
)


class PartglotDataset(Dataset):
    def __init__(
        self,
        hparams: dict,
        game_data,
        h5_data,
        word2int,
    ):
        super().__init__()
        self.game_data = game_data
        self.segs_data = h5_data["data"][:].astype(np.float32)
        self.segs_mask = h5_data["mask"][:].astype(np.float32)

        labels = convert_labels_to_one_hot(self.game_data["target_chair"])
        geo_ids = np.array(
            self.game_data[["chair_a", "chair_b", "chair_c"]], dtype=np.int32
        )

        geo_ids, labels = shuffle_game_geometries(geo_ids, labels)

        padded_text, seq_len = pad_text_symbols_with_zeros(
            self.game_data["text"], hparams["max_seq_len"], force_zero_end=True
        )

        mask, part_indicator = get_mask_of_game_data(
            game_data,
            word2int,
            hparams["only_correct"],
            hparams["only_easy_context"],
            hparams["max_seq_len"],
            hparams["only_one_part_name"],
        )
        self.mask = mask

        self.geo_ids = geo_ids[mask]
        self.labels = labels[mask]
        self.padded_text = padded_text[mask]
        self.part_indicator = part_indicator[mask]

        part_count = np.sum(self.part_indicator, axis=0).astype(np.int)
        
        print_format = "# of utterances"
        for i in range(len(part_names)):
            print_format += f" | {part_names[i]}: {part_count[i]}"
        
        print(print_format)

    def __getitem__(self, idx):
        geo_ids = torch.from_numpy(self.geo_ids[idx])
        target = torch.tensor(np.argmax(self.labels[idx]))
        text = torch.from_numpy(self.padded_text[idx])
        part_ind = torch.from_numpy(self.part_indicator[idx])

        geos = torch.from_numpy(self.segs_data[geo_ids])
        geos_mask = torch.from_numpy(self.segs_mask[geo_ids])

        return geos, geos_mask, text, part_ind, target

    def __len__(self):
        return len(self.geo_ids)


class PartglotTestDataset(Dataset):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        h5_data = h5py.File(
            osp.join(hparams["data_dir"], "shapenet_partseg_chair_bsp.h5")
        )
        self.segs_data = h5_data["data"][:].astype(np.float32)
        self.segs_mask = h5_data["mask"][:].astype(np.float32)

        self.groundtruths, self.signed_distances = unpickle_data(
            osp.join(hparams["data_dir"], "shapenet_partseg_chair_label_and_sd.pkl")
        )

    def __getitem__(self, idx):
        geos = torch.from_numpy(self.segs_data[idx])
        geos_mask = torch.from_numpy(self.segs_mask[idx])

        return geos, geos_mask

    def __len__(self):
        return len(self.segs_data)
    
    def get_groundtruth_and_signed_distance(self, idx):
        return self.groundtruths[idx], self.signed_distances[idx]

