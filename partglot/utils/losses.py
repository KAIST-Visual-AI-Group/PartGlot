import torch
import torch.nn as nn
import torch.nn.functional as F

def smoothed_cross_entropy(pred, target, alpha=0.1):
    n_class = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter_(1, target.view(-1, 1), 1)
    one_hot = (
        one_hot * ((1.0 - alpha) + alpha / n_class) + (1.0 - one_hot) * alpha / n_class
    )  # smoothed

    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    return torch.mean(loss)


def get_point_xnt_loss(model, attn_along_label, alpha):
    B, K, M, N = attn_along_label.shape
    target = torch.max(attn_along_label, 2)[1].reshape(-1)
    pred = attn_along_label.reshape(B * K, M, N).transpose(1, 2).reshape(-1, M)

    n_class = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter_(1, target.view(-1, 1), 1)
    one_hot = (
        one_hot * ((1.0 - alpha) + alpha / n_class) + (1.0 - one_hot) * alpha / n_class
    )
    log_prb = torch.log(pred)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.mean()

    return loss

def xnt_reg_loss(attn_along_pn, geos_mask, alpha=0.1):
    """
    attn_along_pn: [B,len(part_names),n_segs]
    geos_mask: [B,n_segs]
    """
    B, n_parts, n_segs = attn_along_pn.shape
    
    targets = torch.max(attn_along_pn, 1)[1].reshape(B * n_segs)
    preds = attn_along_pn.transpose(1,2).reshape(-1, n_parts)
    geos_mask = geos_mask.reshape(B * n_segs)
    
    one_hot = torch.zeros_like(preds).scatter_(1, targets.view(-1, 1), 1)
    one_hot = one_hot * ((1.0 - alpha) + alpha / n_parts) + (1.0 - one_hot) * alpha / n_parts

    log_prb = torch.log(preds)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = (loss * geos_mask).mean()

    return loss



def get_point_xnt_loss_bsp(model, attn_along_label, mask, alpha):
    """
    score: [B,K,M,N]
    mask: [B,K,N]
    """

    B, K, M, N = attn_along_label.shape
    target = torch.max(attn_along_label, 2)[1].reshape(-1)  # [B*K*N]

    pred = (
        attn_along_label.reshape(B * K, M, N).transpose(1, 2).reshape(-1, M)
    )  # [B*K*N, M]
    mask = mask.reshape(-1)  # [B*K*N]

    n_class = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter_(1, target.view(-1, 1), 1)
    one_hot = (
        one_hot * ((1.0 - alpha) + alpha / n_class) + (1.0 - one_hot) * alpha / n_class
    )

    log_prb = torch.log(pred)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = (loss * mask).mean()

    return loss
