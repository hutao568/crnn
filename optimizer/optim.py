import math


def lr_step(lr, current_epoch, step_size=30, gamma=0.1):
    return lr * (gamma ** (current_epoch // step_size))


def lr_cosine(lr, current_epoch, epochs):
    return lr * 0.5 * (1 + math.cos(current_epoch * math.pi / epochs))


def lr_warmup(lr, current_iter, warm_up, gamma=3):
    return lr * math.pow(current_iter / warm_up, gamma)
