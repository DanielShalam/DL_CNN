import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


# for test set
class CostumeGTSRB(Dataset):
    def __init__(self, root_dir, csv_path, transform=False, csv_file=None):
        # load csv files
        self.root_dir = root_dir

        if csv_file is None:
            self.csv_data = pd.read_csv(os.path.join(root_dir, csv_path))
        else:
            self.csv_data = csv_file

        # load transform
        self.transform = transform

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

        return img, label, img_path
