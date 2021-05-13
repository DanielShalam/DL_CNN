import torch
import math


def num_params(layer):
    return sum([p.numel() for p in layer.parameters()])


def count_acc(y_hat, label):
    pred = torch.argmax(y_hat, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor)


def compute_out_size(num_layers, k_size, p, s, p_k=2, p_s=2):
    """ function to compute next output size. subtract by 2 because of pooling """
    out_h = 30
    out_w = 30
    for i in range(num_layers-1):
        # for convolution
        out_h = math.floor(((out_h + 2 * p - k_size) / s) + 1)
        out_w = math.floor(((out_w + 2 * p - k_size) / s) + 1)
        # for pooling (kernel size 2, stride 2)
        out_h = math.floor(((out_h + - p_k) / p_s) + 1)
        out_w = math.floor(((out_w + - p_k) / p_s) + 1)

    return int(out_h), int(out_w)
