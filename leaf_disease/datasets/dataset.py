from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch

from leaf_disease.datasets.augmentation import DataAugmentation
from lightning import LightningDataModule
from PIL import Image
from PIL.Image import Image as PILImage
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class LeafImageDataset(Dataset):
    def __init__(
        self,
        image_path: list[str | Path],
        targets: list[str | Path],
        transform: A.Compose = None,
    ):
        super().__init__()
        self.image_path = image_path
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_path)

    def __getitem__(self, index: int) -> tuple[PILImage, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image = Image.open(self.image_path[index])
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        # Original shape is (W, H, C) = (600, 800, 3)
        # Pytorch needs this format (C, W, H) = (3, 600, 800)
        # if self.channel_first is True and self.grayscale is False:
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # image_tensor = torch.tensor(image, dtype=torch.float32)

        if self.targets is None:
            return image, torch.zeros(1, dtype=torch.long)

        target = self.targets[index]

        # return image_tensor, torch.tensor(target, dtype=torch.long)
        return image, torch.tensor(target, dtype=torch.long)


class LeafImageDataModule(LightningDataModule):
    def __init__(
        self,
        image_path: str | Path = "input",
        transforms: DataAugmentation = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.image_path = image_path
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)

        if self.transforms is None:
            self.transforms = DataAugmentation()

    def prepare_data(self):
        # download data, tokenize, save, etc.
        pass

    def setup(self, stage: str = None):
        # Read train.csv
        dfx = pd.read_csv(self.image_path / "train.csv")

        # Split between train and validation
        df_train, df_valid = split_data(dfx)

        # Read traning and validation images
        train_image_path = self.image_path / "train_images"
        train_image_paths = [train_image_path / img_id for img_id in df_train.image_id]
        valid_image_paths = [train_image_path / img_id for img_id in df_valid.image_id]
        print(f"{len(train_image_paths)=}")
        print(f"{len(valid_image_paths)=}")

        # Create Datasets
        self.train = LeafImageDataset(
            image_path=train_image_paths,
            targets=df_train.label.values,
            transform=self.transforms.train_transforms,
        )
        self.valid = LeafImageDataset(
            image_path=valid_image_paths,
            targets=df_valid.label.values,
            transform=self.transforms.valid_transforms,
        )

        # Dataloader for predictions
        test_image_path = self.image_path / "test_images"
        test_image_paths = list(test_image_path.glob("*.jpg"))
        self.predict_dl = LeafImageDataset(
            image_path=test_image_paths,
            targets=None,
            transform=self.transforms.valid_transforms,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return valid_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            dataset=self.predict_dl,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return predict_loader


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    train_idx, valid_idx = train_test_split(
        df.index.values, test_size=test_size, random_state=42, stratify=df.label.values
    )
    df_train = df.iloc[train_idx, :].reset_index(drop=True)
    df_valid = df.iloc[valid_idx, :].reset_index(drop=True)
    return df_train, df_valid


if __name__ == "__main__":
    # image_paths = ["input/train_images/6103.jpg", "input/train_images/218377.jpg"]
    # targets = [0, 1]

    # train_dataset = LeafImageDataset(image_path=image_paths, targets=targets)
    # print(train_dataset[1])
    dm = LeafImageDataModule()
    dm.setup()
