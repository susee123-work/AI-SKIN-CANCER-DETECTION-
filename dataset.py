# dataset.py
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImgFolderDataset(Dataset):
    """
    Custom Dataset to load images from folder structure:
    root_dir/
        class1/
            img1.jpg
            img2.png
        class2/
            img3.jpg
    """
    def __init__(self, root_dir, train=True, image_size=224):
        self.samples = []
        self.root = Path(root_dir)
        if not self.root.exists():
            raise ValueError(f"Directory {root_dir} not found")
        
        # Classes sorted alphabetically
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not self.classes:
            raise ValueError(f"No class folders found in {root_dir}")
        
        # Gather all image file paths
        for label_idx, label in enumerate(self.classes):
            for f in (self.root / label).iterdir():
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    self.samples.append((str(f), label_idx))

        if not self.samples:
            raise ValueError(f"No image files found in {root_dir}")
        
        self.train = train
        self.transform = self._get_transform(train, image_size)

    def _get_transform(self, train, image_size):
        if train:
            return T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(20),
                T.ColorJitter(0.2, 0.2, 0.15, 0.05),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
