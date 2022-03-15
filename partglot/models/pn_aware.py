from partglot.modules.encoders import LSTM
from partglot.utils.losses import smoothed_cross_entropy, xnt_reg_loss
from partglot.datamodules.data_utils import part_names
from partglot.modules.encoders import CrossAttention, SupSegsEncoder, MLP
import torch
import torch.nn as nn
from partglot.models.pn_agnostic import PNAgnostic


class PNAware(PNAgnostic):
    def _build_model(self):
        pn_tensor = torch.tensor([self.hparams.word2int[x] for x in part_names])[None]
        self.register_buffer("pn_tensor", pn_tensor)

        self.clsf_encoder = LSTM(
            self.hparams.text_dim,
            self.hparams.embedding_dim,
            vocab_size=len(self.hparams.word2int),
        )
        self.attn_encoder = nn.Sequential(
            self.clsf_encoder.word_embedding,
            nn.Linear(self.hparams["embedding_dim"], self.hparams["text_dim"]),
        )
        self.sup_segs_encoder = SupSegsEncoder(self.hparams.sup_segs_dim)
        self.cross_attention = CrossAttention(
            src_dim=self.hparams.text_dim, tgt_dim=self.hparams.sup_segs_dim
        )
        self.mlp = MLP(input_dim=self.hparams.text_dim * 2)

    def forward(self, segs, segs_mask, text, part_indicator, return_only_attn=False):
        """
        Input:
            segs: [B,K,n_segs,n_points,3]
            segs_mask: [B,K,n_segs]
            text: [B,len_seq]
            part_indicator: [B,len(part_names)]
        Output:
            logits: [B, K]
            cross_attn_weights: [B,K,len(part_names),n_segs]
        """
        outputs = dict()

        B, K, n_segs, n_points, _ = segs.size()
        segs_f = self._forward_sup_segs(segs)
        segs_mask = segs_mask.reshape(B * K, n_segs)

        attn_enc_f = self.attn_encoder(self.pn_tensor).repeat(B * K, 1, 1)
        if part_indicator is not None:
            part_indicator = part_indicator.repeat_interleave(K, dim=0)

        cross_attn_f, cross_attn_weights, attn_along_pn = self.cross_attention(
            attn_enc_f, segs_f, segs_mask, part_indicator, double_softmax=True,
            return_only_attn=return_only_attn
        )
        cross_attn_weights = cross_attn_weights.reshape(B, K, attn_enc_f.shape[1], n_segs)
        
        if return_only_attn:
            return cross_attn_weights

        clsf_enc_f, clsf_enc_attn = self.clsf_encoder(text)
        clsf_enc_f = torch.repeat_interleave(clsf_enc_f, K, dim=0).unsqueeze(
            1
        )  # [B*K, 1, dim]

        logits = self.mlp(torch.cat([clsf_enc_f, cross_attn_f], -1).squeeze(1))
        logits = logits.reshape(B, K)

        outputs["logits"] = logits
        outputs["cross_attn_weights"] = cross_attn_weights
        outputs["attn_along_pn"] = attn_along_pn

        return outputs

    def step(self, batch):
        segs, segs_mask, text, part_indicator, targets = batch

        outputs = self(segs, segs_mask, text, part_indicator)

        logits = outputs["logits"]
        loss = smoothed_cross_entropy(logits, targets)

        if self.hparams["xnt_reg"]:
            xnt_loss = xnt_reg_loss(outputs["attn_along_pn"], segs_mask)
            loss = loss + self.hparams.xnt_reg_weights * xnt_loss

        preds = torch.max(logits, 1)[1]

        return loss, preds, targets
        
    def process_test_dataloader(self, test_seg_dl):
        total_attn_maps, total_geos_masks = [], []
        for geos, geos_mask in test_seg_dl:
            geos = geos.to(self.device).unsqueeze(1)
            geos_mask = geos_mask.to(self.device).unsqueeze(1)

            cross_attn_weights = self(geos, geos_mask, text=None, part_indicator=None, return_only_attn=True)

            total_attn_maps.append(cross_attn_weights.squeeze(1))
            total_geos_masks.append(geos_mask)

        total_attn_maps = torch.cat(total_attn_maps, 0)
        total_geos_masks = torch.cat(total_geos_masks, 0).squeeze(1)

        return total_attn_maps, total_geos_masks
