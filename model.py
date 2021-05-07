import torch
from torch import nn
from utils import compute_out_size, count_acc


def conv_block(in_channels, out_channels, k_size, padding=1, batch_norm=True):
    """ convolutional block. perform according to batch_norm flag."""
    if batch_norm:
        bn = nn.BatchNorm2d(out_channels)
        nn.init.uniform_(bn.weight)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=1, padding=padding),
            bn,
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


class ConvNet(nn.Module):

    def __init__(self, fe_dims, k_size=3, padding=1, dropout=True):
        super().__init__()

        # build feature extractor layers
        fe_layers = []
        for dim in zip(fe_dims[:-1], fe_dims[1:]):
            in_dim, out_dim = dim
            fe_layers += [
                conv_block(in_dim, out_dim, k_size, padding=padding, batch_norm=dropout)
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
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.output_size)
        )

        self.dropout = nn.Dropout(0.5) if dropout else None
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # feature extractor
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        # check if dropout needed
        if self.dropout:
            features = self.dropout(features)
        # classifier
        class_scores = self.fc_classifier(features)
        # softmax
        class_scores = self.log_softmax(class_scores)
        return class_scores


def train_model(data, device):
    train_set, val_set, test_set = data
    # define model params
    fe_dims = [3, 32, 32, 64]
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    num_epochs = 30
    learning_rate = 0.001
    model = ConvNet(fe_dims=fe_dims).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train loop
    model.train()
    for epoch in range(num_epochs):
        for i, (batch, labels) in enumerate(train_set):
            optimizer.zero_grad()
            labels = labels.to(device)
            batch = batch.to(device)
            # calc net output
            y_hat = model.forward(batch)
            # calc loss
            loss = cross_entropy_loss(y_hat, labels)
            # calc acc
            acc = count_acc(y_hat, labels)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_set), loss.item(), acc))

            loss.backward()
            optimizer.step()
