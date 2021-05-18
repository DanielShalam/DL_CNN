import torch
import math
import matplotlib.pyplot as plt
from torchvision.transforms import transforms


def count_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def count_acc(y_hat, label):
    pred = torch.argmax(y_hat, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor)


def compute_out_size(num_layers, k_size, p, s, p_k=2, p_s=2):
    """ function to compute next output size. subtract by 2 because of pooling """
    out_h = 30
    out_w = 30
    for i in range(num_layers - 1):
        # for convolution
        out_h = math.floor(((out_h + 2 * p - k_size) / s) + 1)
        out_w = math.floor(((out_w + 2 * p - k_size) / s) + 1)
        # for pooling (kernel size 2, stride 2)
        out_h = math.floor(((out_h + - p_k) / p_s) + 1)
        out_w = math.floor(((out_w + - p_k) / p_s) + 1)

    return int(out_h), int(out_w)


def plot_data(train_acc, val_acc, train_loss, val_loss, best_epoch):
    # accuracy plot
    x_axis = list(range(1, len(train_acc) + 1))
    plt.plot(x_axis, train_acc, color='g', label='Train accuracy')
    plt.plot(x_axis, val_acc, color='r', label='Validation accuracy')
    plt.scatter(best_epoch, val_acc[best_epoch - 1], label='Epoch taken')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Accuracy as a function of time")
    plt.show()

    # losses plot
    plt.plot(x_axis, train_loss, color='black', label='Train loss')
    plt.plot(x_axis, val_loss, color='brown', label='Validation loss')
    plt.scatter(best_epoch, val_loss[best_epoch - 1], label='Epoch taken')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Loss as a function of time")
    plt.show()


# Resize, normalize and shear image
data_shear = transforms.Compose([
    transforms.Resize((30, 30)),
    transforms.RandomAffine(degrees=15, shear=2),
    transforms.ColorJitter(brightness=0.2, hue=0.05, contrast=0.2),
    transforms.RandomRotation([-20, 20]),
    transforms.RandomPerspective(distortion_scale=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Resize, normalize and translate image
data_translate = transforms.Compose([
    transforms.Resize((30, 30)),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

data_transform = transforms.Compose([
        transforms.Resize((30, 30)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])