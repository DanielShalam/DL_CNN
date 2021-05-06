import gtsrb_dataset as dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':

    # create transformations, include resize 30x30x3
    transform = transforms.Compose([
        transforms.Resize((30, 30)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])

    # build train val
    train_set, validation_set = dataset.load_train_val('data/Train', val_pct=10, transform=transform)
    print(f"Train size: {len(train_set)}, Validation size: {len(validation_set)}")

    # build iterator for test set
    test_set = dataset.GTSRB(root_dir='data', csv_path='Test.csv', transform=transform)

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(validation_set, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

    cross_entropy_loss = torch.nn.CrossEntropyLoss()



