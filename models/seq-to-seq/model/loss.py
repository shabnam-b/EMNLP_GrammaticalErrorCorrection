"""
Different loss functions.
"""

import torch
import torch.nn as nn

from utils import constant

class SequenceLoss(nn.Module):
    def __init__(self, vocab_size, alpha):
        super().__init__()
        weight = torch.ones(vocab_size)
        weight[constant.PAD_ID] = 0
        self.nll = nn.NLLLoss(weight)
        self.alpha = alpha

    def forward(self, inputs, targets, attn, cov):
        assert inputs.size(0) == targets.size(0)
        nll_loss = self.nll(inputs, targets)
        if self.alpha == 0:
            return nll_loss
        # add coverage loss
        cov_loss = torch.sum(torch.min(attn, cov), dim=2).view(-1)
        pad_mask = targets.eq(constant.PAD_ID)
        unpad_mask = targets.ne(constant.PAD_ID)
        cov_loss.masked_fill_(pad_mask, 0)
        denom = torch.sum(unpad_mask).float()
        cov_loss = torch.sum(cov_loss) / (denom + constant.SMALL_NUMBER)
        return nll_loss + self.alpha * cov_loss

    def update_alpha(self, alpha):
        print("[Update coverage loss weight to be {}]".format(alpha))
        self.alpha = alpha