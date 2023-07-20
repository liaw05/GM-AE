import math


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


def adjust_lr_cos_multi(optimizer, epoch, max_epochs=[160,180,200,220,240], lr_init=3e-4):
    """Decay the learning rate based on schedule"""
    # cosine lr schedule
    for max_epoch in max_epochs:
        if epoch < max_epoch:
            break
    lr = lr_init * 0.5 * (1. + math.cos(math.pi * (20-max_epoch+epoch) / 20))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr_cos(optimizer, epoch, max_epochs=300, warm_epoch=10, lr_init=1e-3, fix_lr=1e-4):
    """Decay the learning rate based on schedule"""
    # cosine lr schedule
    if epoch < warm_epoch:
        cur_lr = lr_init * (epoch*(1.0-0.1)/warm_epoch + 0.1)
    else:
        cur_lr = lr_init * 0.5 * (1. + math.cos(math.pi * (epoch-warm_epoch)/ (max_epochs-warm_epoch)))

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = fix_lr
        else:
            param_group['lr'] = cur_lr