import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch.utils.data as data


# for test set
class CostumeGTSRB(Dataset):
    def __init__(self, root_dir, csv_path, transform=False, train=True):
        # load csv files
        self.root_dir = root_dir
        self.csv_data = pd.read_csv(os.path.join(root_dir, csv_path))
        # load transform
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_data = self.csv_data.iloc[idx]
        img_path = os.path.join(self.root_dir, img_data['Path'])

        img = Image.open(img_path)
        label = img_data['ClassId']
        # assert img_data['Width'] == img.width and img_data['Height'] == img.height

        if self.transform:
            img = self.transform(img)

        return img, label
