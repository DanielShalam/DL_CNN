import torch
from torch import nn


def calc_accuracy(test_data, model):
    # calculate model accuracy
    accuracy = 0
    counter = 0
    for j, (data, label) in enumerate(test_data):
        data = data.view(-1, 30 * 30)
        y_hat = model(data)
        pred_y = y_hat.argmax(dim=1)
        # calculate the number of equal predicted labels and ground truth labels
        accuracy += int(torch.sum(pred_y == label))
        counter += 1

    accuracy /= counter
    return accuracy


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
