import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms


class GoogleDataset(Dataset):
    """Google Image Dataset."""

    def __init__(self, csv_file: str, image_dir: str, batch_size: int):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.batch_size = batch_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.image_dir,
                                self.df.iloc[idx].label,
                                self.df.iloc[idx].image_name)

        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = self[idx + 1]

        image = image.copy()
        if self.transform is not None:
            image = self.transform(image)

        return image
