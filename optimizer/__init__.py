import torch.nn as nn


def adjust_learning_rate(optimizer, epoch, iter_size, iter_num, args):
    current_iter = epoch * iter_size + iter_num

    if args.warm_up is not None and current_iter < args.warm_up:
        from .optim import lr_warmup
        current_lr = lr_warmup(args.lr, current_iter, args.warm_up)
    else:
        from .optim import lr_cosine
        current_lr = lr_cosine(args.lr, epoch, args.epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return current_lr


def get_criterion(criterion_type='cross_entropy'):
    if criterion_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif criterion_type == 'ctc':
        from warpctc_pytorch import CTCLoss
        # from torch.nn import CTCLoss
        return CTCLoss()
    else:
        raise ValueError('unknown criterion type', criterion_type)
