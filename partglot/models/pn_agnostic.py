import os
from partglot.utils.simple_utils import get_iou_per_instance_per_part
import numpy as np
import torch
import pytorch_lightning as pl
import os.path as osp

from torch.utils.data.dataloader import DataLoader
from partglot.modules.encoders import LSTM, SupSegsEncoder, CrossAttention, MLP
from partglot.datamodules.datasets.partglot_dataset import PartglotTestDataset
from partglot.utils.losses import smoothed_cross_entropy
from partglot.utils.neural_utils import PolyDecayScheduler, tokenizing
from partglot.datamodules.data_utils import part_names


class PNAgnostic(pl.LightningModule):
    def __init__(
        self,
        text_dim: int,
        embedding_dim: int,
        sup_segs_dim: int,
        lr: float,
        data_dir: str,
        word2int: dict,
        total_steps: int,
        measure_iou_every_epoch: bool,
        save_pred_label_every_epoch: bool,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self._build_model()

    def _build_model(self):
        self.clsf_encoder = LSTM(
            self.hparams.text_dim,
            self.hparams.embedding_dim,
            vocab_size=len(self.hparams.word2int),
        )
        self.attn_encoder = LSTM(
            self.hparams.text_dim,
            self.hparams.embedding_dim,
            vocab_size=len(self.hparams.word2int),
        )
        self.sup_segs_encoder = SupSegsEncoder(self.hparams.sup_segs_dim)
        self.cross_attention = CrossAttention(
            src_dim=self.hparams.text_dim, tgt_dim=self.hparams.sup_segs_dim
        )
        self.mlp = MLP(input_dim=self.hparams.text_dim * 2)

    def _forward_sup_segs(self, sup_segs):
        """
        Input: [B,K,n_segs,n_points,3]
        Output: [B*K,n_segs,dim_out]
        """
        B, K, n_segs, n_points, _ = sup_segs.size()

        out = self.sup_segs_encoder(sup_segs.reshape(B * K, n_segs * n_points, -1))
        out = out.reshape(B * K, n_segs, n_points, -1)

        out = torch.max(out, 2)[0]
        return out

    def forward(self, segs, segs_mask, text, return_only_attn=False):
        """
        Input:
            segs: [B,K,n_segs,n_points,3]
            segs_mask: [B,K,n_segs]
            text: [B, len_seq]
            return_only_attn (if True): return only attention maps.
        Output: dict
            logits: [B, K]
            cross_attn_weights: [B, K, 1, n_segs]
        """
        outputs = dict()

        B, K, n_segs, n_points, in_dim = segs.size()
        segs_f = self._forward_sup_segs(segs)
        segs_mask = segs_mask.reshape(B * K, n_segs)

        attn_enc_f, attn_enc_attn = self.attn_encoder(text)
        attn_enc_f = torch.repeat_interleave(attn_enc_f, K, dim=0).unsqueeze(
            1
        )  # [B*K, 1, dim]

        clsf_enc_f, clsf_enc_attn = self.clsf_encoder(text)
        clsf_enc_f = torch.repeat_interleave(clsf_enc_f, K, dim=0).unsqueeze(
            1
        )  # [B*K, 1, dim]

        cross_attn_f, cross_attn_weights, _ = self.cross_attention(
            attn_enc_f, segs_f, segs_mask, return_only_attn=return_only_attn
        )
        cross_attn_weights = cross_attn_weights.reshape(B, K, 1, n_segs)

        if return_only_attn:
            return cross_attn_weights

        logits = self.mlp(torch.cat([clsf_enc_f, cross_attn_f], -1).squeeze(1))
        logits = logits.reshape(B, K)

        outputs["logits"] = logits
        outputs["cross_attn_weights"] = cross_attn_weights

        return outputs

    def step(self, batch):
        segs, segs_mask, text, part_indicator, targets = batch
        outputs = self(segs, segs_mask, text)

        logits = outputs["logits"]
        loss = smoothed_cross_entropy(logits, targets)

        preds = torch.max(logits, 1)[1]

        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        acc = (targets == preds).float().mean() * 100
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        acc = (targets == preds).float().mean() * 100
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs):
        if self.hparams.measure_iou_every_epoch:
            self.test_segmentation(stage="val")

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        correct = (targets == preds).sum()
        total = targets.numel()

        return dict(loss=loss, correct=correct, total=total)

    def test_epoch_end(self, outputs):
        loss_sum = total_correct = total_seen = 0
        for o in outputs:
            loss_sum += o["loss"]
            total_correct += o["correct"]
            total_seen += o["total"]

        self.log("test/loss", loss_sum / total_seen, on_epoch=True)
        self.log("test/acc", total_correct / total_seen * 100, on_epoch=True)

        self.test_segmentation(stage="test")

    def test_segmentation(self, stage: str):
        if stage == "val":
            save_dir = f"pred_label/epoch={self.current_epoch}"
        elif stage == "test":
            save_dir = f"pred_label/final"
        else:
            raise ValueError("stage must be val or test")

        os.makedirs(save_dir, exist_ok=True)

        test_seg_ds = PartglotTestDataset(self.hparams)
        test_seg_dl = DataLoader(test_seg_ds, batch_size=32, num_workers=4)

        total_attn_maps, total_geos_masks = self.process_test_dataloader(test_seg_dl)

        total_iou = []
        total_seen = 0
        for i in range(total_attn_maps.shape[0]):
            (
                groundtruth,
                signed_distance,
            ) = test_seg_ds.get_groundtruth_and_signed_distance(i)

            attn, mask = total_attn_maps[i], total_geos_masks[i]
            attn = attn[:, mask == 1]

            sup_segs2label = attn.max(0)[1].cpu().numpy()
            pc2sup_segs = np.argmax(signed_distance, 1)

            assign_ft = lambda x: sup_segs2label[x]

            pc2label = assign_ft(pc2sup_segs)

            if (stage == "val" and self.hparams.save_pred_label_every_epoch) or (
                stage == "test"
            ):
                np.save(osp.join(save_dir, f"{i}_mesh_label.npy"), sup_segs2label)
                np.save(osp.join(save_dir, f"{i}_pc_label.npy"), pc2label)

            total_iou.append(
                get_iou_per_instance_per_part(pc2label, groundtruth, len(part_names))
            )
            total_seen += 1

        total_iou = np.stack(total_iou, 0)

        miou_per_part = total_iou.mean(0)
        miou_avg = total_iou.mean(1).mean(0)

        for i, pn in enumerate(part_names):
            self.log(f"{stage}/{pn}_iou", miou_per_part[i])
        self.log(f"{stage}/miou_avg", miou_avg)

        if stage == "val":
            resultfilename = osp.join(save_dir, "miou_results.txt")
        else:
            resultfilename = "final_miou_results.txt"

        with open(resultfilename, "w") as f:
            for iou, pn in zip(miou_per_part, part_names):
                f.write(f"{pn}_iou: {iou*100:.1f}\n")
            f.write(f"miou_avg: {miou_avg*100:.1f}\n")

        print_fmt = f"avg mIoU: {miou_avg*100:.1f}%"
        for iou, pn in zip(miou_per_part, part_names):
            print_fmt += f" | {pn}: {iou*100:.1f}%"

        print(print_fmt)

    def process_test_dataloader(self, test_seg_dl):
        total_attn_maps, total_geos_masks = [], []

        prompts = []
        for pn in part_names:
            prompts.append(tokenizing(self.hparams.word2int, f"a chair with {pn}"))

        for geos, geos_mask in test_seg_dl:
            geos = geos.to(self.device).unsqueeze(1)
            geos_mask = geos_mask.to(self.device).unsqueeze(1)

            attn_maps = []
            for text_prompt in prompts:
                text_prompt = text_prompt.to(self.device)[None].expand(
                    geos.shape[0], -1
                )
                cross_attn_weights = self(
                    geos, geos_mask, text_prompt, return_only_attn=True
                )
                attn_maps.append(cross_attn_weights.squeeze())
            attn_maps = torch.stack(attn_maps, 1)

            total_attn_maps.append(attn_maps)
            total_geos_masks.append(geos_mask)

        total_attn_maps = torch.cat(total_attn_maps, 0)
        total_geos_masks = torch.cat(total_geos_masks, 0).squeeze(1)

        return total_attn_maps, total_geos_masks

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = {
            "scheduler": PolyDecayScheduler(optimizer, self.hparams.total_steps),
            "interval": "step",
        }

        return [optimizer], [scheduler]
