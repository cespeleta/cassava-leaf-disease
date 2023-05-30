import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataAugmentation:
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, image_size: int = 244) -> None:
        self.image_size = image_size

    @property
    def train_transforms(self):
        return A.Compose(
            [
                # A.RandomResizedCrop(height=self.image_size, width=self.image_size),
                # # Divide pixel values of an image by 255, so each pixel's value will
                # # lie in a range [0.0, 1.0]
                # A.Normalize(max_pixel_value=255),
                # ToTensorV2(),  # Reshape to [C, W, H]
                A.RandomResizedCrop(self.image_size, self.image_size),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255,
                ),
                ToTensorV2(),  # Reshape to [C, W, H]
            ]
        )

    @property
    def valid_transforms(self):
        return A.Compose(
            [
                A.CenterCrop(height=self.image_size, width=self.image_size),
                A.Normalize(max_pixel_value=255),
                ToTensorV2(),  # Reshape to [C, W, H]
            ]
        )

    # @property
    # def train_transforms(self):
    #     return self._train_transforms

    # @train_transforms.setter
    # def train_transforms(self, value):
    #     self._train_transforms = value


if __name__ == "__main__":
    print(
        f"{DataAugmentation().train_transforms=}",
        f"{DataAugmentation(image_size=512).train_transforms=}",
    )
    print(
        f"{DataAugmentation(image_size=512).valid_transforms=}",
    )
