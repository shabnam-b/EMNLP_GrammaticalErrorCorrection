import torch
from torch.optim import Optimizer


def get_optimizer(parameters, lr):
    return torch.optim.Adam(parameters, lr=lr)


def update_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var


def keep_partial_grad(grad, topk):
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad
