import torch
import math
from torch.optim.lr_scheduler import LambdaLR


class PolyDecayScheduler(LambdaLR):
    def __init__(self, optimizer, total_steps, power=0.9, lr_end=1e-7, last_epoch=-1):
        def lr_lambda(step):
            lr = (math.pow(1 - float(step) / float(total_steps), power))
            return lr if lr > lr_end else lr_end

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def tokenizing(word2int: dict, text: str):
    """
    Input:
        word2int: dict
        text: str
    Output:
        tensor of token: [len_seq]
    """
    token = list(map(lambda x: word2int[x], text.split(" ")))
    token = torch.tensor(token)

    return token
    
