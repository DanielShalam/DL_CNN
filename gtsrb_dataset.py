import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch.utils.data as data

# for test set
class GTSRB(Dataset):
    def __init__(self, root_dir, csv_path, transform=None):
        # load csv files
        self.root_dir = root_dir
        self.csv_data = pd.read_csv(os.path.join(root_dir, csv_path))
        # load transform
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)
        label = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

# for train validation set
def load_train_val(train_path, val_pct, transform):
    """ load train validation images using ImageFolder """
    # load train test
    train_data = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    # train val split
    val_len = int(len(train_data) / val_pct)
    train_len = len(train_data) - val_len
    train_data, val_data = data.random_split(train_data, [train_len, val_len])
    return train_data, val_data
