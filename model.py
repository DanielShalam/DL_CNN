import torch
from torch import nn
from utils import compute_out_size, count_acc, num_params, plot_data
import numpy as np

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

    def __init__(self, fe_dims, c_dims, k_size=3, padding=1, dropout=True, mode='f_conv'):
        super().__init__()

        self.mode = mode
        self.n_params = 0
        fe_layers = []
        c_layers = []
        # compute size for fully connected classifier
        self.h, self.w = compute_out_size(num_layers=len(fe_dims), k_size=k_size, p=padding, s=1)

        # build feature extractor layers
        for in_dim, out_dim in zip(fe_dims[:-1], fe_dims[1:]):
            fe_layers += [
                conv_block(in_dim, out_dim, (k_size, k_size), padding=padding, batch_norm=dropout)
            ]
            self.n_params += num_params(fe_layers[-1])

        # build classifier layers
        if self.mode == 'f_conv':  # if model fully-convolutional
            dims = [fe_dims[-1], *c_dims]
            for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
                kernel_size = (self.h, self.w) if idx == 0 else (1, 1)
                c_layers += [
                    nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size),
                    nn.ReLU()
                ]
                self.n_params += num_params(fe_layers[-1])
        else:  # if model fully-connected
            c_layers += [nn.Flatten()]
            self.n_params += num_params(c_layers[-1])
            dims = [fe_dims[-1] * self.h * self.w, *c_dims]
            for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
                c_layers += [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU()
                ]
                self.n_params += num_params(c_layers[-1])

        self.feature_extractor = nn.Sequential(*fe_layers)
        self.classifier = nn.Sequential(*c_layers[:-1])
        print(f"Number of params: {self.n_params}")

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # feature extractor
        features = self.feature_extractor(x)
        # classifier
        class_scores = self.classifier(features)
        class_scores = self.log_softmax(class_scores)
        return class_scores


def init_weights(m):
    if type(m) == nn.Linear:
        # find number of features
        n = m.in_features
        y = 2.0 / np.sqrt(n)
        m.weight.data.xavier_normal_()
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    if type(m) == nn.Conv2d:
        n = len(m.weight)
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def eval_test(model, data, device, mode):
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
            if mode == 'f_conv':
                y_hat = torch.squeeze(y_hat)
            predictions = torch.argmax(y_hat, dim=1)
            n_correct += torch.sum(predictions == labels).type(torch.float32)
            n_total += batch.shape[0]

    acc = (n_correct / n_total).item()
    return acc


def train_loop(model, optimizer, loss_fn, data, device, epoch, train=True, mode='f_conv'):
    """ main loop, for train or validation set """

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
        if mode == 'f_conv':
            y_hat = torch.squeeze(y_hat)
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


def train_model(data, device, dropout=True, mode='f_conv'):
    """ main function, train, validate, test """

    train_set, val_set, test_set = data
    all_train_acc = []
    all_val_acc = []
    all_train_losses = []
    all_val_losses = []

    # define model params
    fe_dims = [3, 32, 32, 64]
    c_dims = [120, 80, 43]
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    max_epochs = 3
    learning_rate = 0.001
    model = ConvNet(fe_dims=fe_dims, c_dims=c_dims, dropout=dropout).to(device)
    model.apply(init_weights)    # initialize model weights
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # early stopping params
    best_val_loss = np.inf
    j_patience = 0
    patience = 5
    early_stop = False

    # train loop
    for epoch in range(max_epochs):
        # train
        train_acc, train_loss = train_loop(model=model, optimizer=optimizer, loss_fn=cross_entropy_loss, data=train_set,
                                           device=device, epoch=epoch, train=True, mode=mode)
        all_train_acc.append(train_acc)
        all_train_losses.append(train_loss)

        # validation
        val_acc, val_loss = train_loop(model=model, optimizer=optimizer, loss_fn=cross_entropy_loss, data=val_set,
                                       device=device, epoch=epoch, train=False, mode=mode)

        all_val_acc.append(val_acc)
        all_val_losses.append(val_loss)

        print("{}   epoch {} | train loss: {}| train accuracy: {} {}".format(CRED, epoch, train_loss, train_acc, CEND))
        print("{}   epoch {} | validation loss: {}| validation accuracy: {} {}".format(CRED, epoch, val_loss, val_acc,
                                                                                       CEND))

        # early stopping
        # check for improvement in the validation loss
        if val_loss < best_val_loss:
            # update algorithm parameters
            print(f"{CGREEN}Reached best loss of: {val_loss}{CEND}")
            best_epoch = epoch
            j_patience = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"models/best_model_{mode}.pt")
        else:
            j_patience += 1

        # if the model not improve for 'patience' iterations
        if j_patience >= patience:
            model.load_state_dict(torch.load(f"models/best_model_{mode}.pt"))
            early_stop = True
            break

    if not early_stop: best_epoch = max_epochs
    test_acc = eval_test(model, test_set, device, mode)
    plot_data(all_train_acc, all_val_acc, all_train_losses, all_val_losses, best_epoch=best_epoch)
    print(f"{CGREEN} Test set accuracy: {test_acc} {CEND}")
