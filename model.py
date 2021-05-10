import torch
from torch import nn
from utils import compute_out_size, count_acc

# prints colors
CEND = '\33[0m'
CRED = '\33[31m'
CGREEN = '\33[32m'


def conv_block(in_channels, out_channels, k_size, padding=1, batch_norm=True):
    """ convolutional block. perform according to batch_norm flag."""
    if batch_norm:
        bn = nn.BatchNorm2d(out_channels)
        nn.init.uniform_(bn.weight)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=(1, 1), padding=(padding, padding)),
            bn,
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.5)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=(1, 1), padding=(padding, padding)),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


class ConvNet(nn.Module):
    """ CNN model """

    def __init__(self, fe_dims, k_size=3, padding=1, dropout=True):
        super().__init__()

        # build feature extractor layers
        fe_layers = []
        for dim in zip(fe_dims[:-1], fe_dims[1:]):
            in_dim, out_dim = dim
            fe_layers += [
                conv_block(in_dim, out_dim, (k_size, k_size), padding=padding, batch_norm=dropout)
            ]

        # compute size for fully connected classifier
        self.h, self.w = compute_out_size(num_layers=len(fe_dims), k_size=k_size, p=padding, s=1)

        self.feature_extractor = nn.Sequential(*fe_layers)

        # build classifier layers
        self.output_size = 43
        self.fc_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fe_dims[-1] * self.h * self.w, 120),
            nn.ReLU(),
            nn.Linear(120, 80),
            nn.ReLU(),
            nn.Linear(80, self.output_size)
        )

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # feature extractor
        features = self.feature_extractor(x)
        # features = features.view(features.size(0), -1)
        # classifier
        class_scores = self.fc_classifier(features)
        # softmax
        class_scores = self.log_softmax(class_scores)
        return class_scores


def eval_test(model, data, device):
    """ test set evaluation """
    n_correct = 0
    n_total = 0
    # evaluation loop
    model.eval()
    with torch.no_grad():
        for i, (batch, labels) in enumerate(data):
            labels = labels.to(device)
            batch = batch.to(device)
            # calc net output
            y_hat = model(batch)
            predictions = torch.argmax(y_hat, dim=1)
            n_correct += torch.sum(predictions == labels).type(torch.float32)
            n_total += batch.shape[0]

    acc = (n_correct / n_total).item()
    return acc


def train_loop(model, optimizer, loss_fn, data, device, epoch, train=True):
    """ main loop, for train set or test set """

    # decide train of validate
    if train:
        model.train()
    else:
        model.eval()

    n_total = 0
    epoch_acc = 0
    losses = torch.zeros(len(data))
    # train/validation loop
    for i, (batch, labels) in enumerate(data):
        optimizer.zero_grad()
        # convert to cuda
        labels = labels.to(device)
        batch = batch.to(device)
        # calc net output
        y_hat = model.forward(batch)
        # calc loss
        loss = loss_fn(y_hat, labels)
        # calc acc
        acc = count_acc(y_hat, labels)
        epoch_acc += torch.sum(acc).type(torch.float32)
        batch_acc = acc.mean().item()
        losses[i] = loss.item()
        n_total += batch.shape[0]

        if train:
            loss.backward()
            optimizer.step()
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(data), loss.item(), batch_acc))

        else:
            print('epoch {}, validation {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(data), loss.item(),
                                                                              batch_acc))
    epoch_acc = (epoch_acc / n_total).item()
    epoch_loss = torch.mean(losses)
    return epoch_acc, epoch_loss


def train_model(data, device, dropout=True):
    """ main function, train, validate, test """

    train_set, val_set, test_set = data
    # define model params
    fe_dims = [3, 64, 64, 64]
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    num_epochs = 30
    learning_rate = 0.001
    model = ConvNet(fe_dims=fe_dims, dropout=dropout).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train loop
    for epoch in range(num_epochs):
        # train
        train_acc, train_loss = train_loop(model=model, optimizer=optimizer, loss_fn=cross_entropy_loss, data=train_set,
                                           device=device, epoch=epoch, train=True)

        # validation
        val_acc, val_loss = train_loop(model=model, optimizer=optimizer, loss_fn=cross_entropy_loss, data=val_set,
                                       device=device, epoch=epoch, train=False)

        print("{} epoch {} | train loss: {}| train accuracy: {} {}".format(CRED, epoch, train_loss, train_acc, CEND))
        print("{} epoch {} | validation loss: {}| validation accuracy: {} {}".format(CRED, epoch, val_loss, val_acc, CEND))

        test_acc = eval_test(model, test_set, device)
        print(f"{CGREEN} Test set accuracy: {test_acc} {CEND}")
