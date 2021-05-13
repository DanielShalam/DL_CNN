import gtsrb_dataset as dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import model

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validation_split = 0.1
    mode = ['f_connected', 'f_connected_b_d', 'f_conv']
    # create transformations, include resize 30x30x3
    transform = transforms.Compose([
        transforms.Resize((30, 30)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])

    # build iterator for train and test set
    train_set = dataset.CostumeGTSRB(root_dir='data', csv_path='Train.csv', transform=transform)
    test_set = dataset.CostumeGTSRB(root_dir='data', csv_path='Test.csv', transform=transform)

    # train validation split
    # Creating data indices for training and validation splits:
    dataset_size = len(train_set)
    indices = np.arange(dataset_size)
    split = int(np.floor(validation_split * dataset_size))
    # shuffle data
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=64, num_workers=4, sampler=train_sampler)
    val_loader = DataLoader(train_set, batch_size=64, num_workers=4, sampler=val_sampler)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

    # train model according to mode
    model.train_model([train_loader, val_loader, test_loader], device, mode='f_conv')






