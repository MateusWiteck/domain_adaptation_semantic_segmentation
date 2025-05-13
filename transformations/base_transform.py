import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class BaseTransform:
    def __init__(self, height=512, width=1024):
        self.height = height
        self.width = width

    def _finalize(self, transforms):
        return A.Compose(
            transforms + [
                A.Resize(height=self.height, width=self.width),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ]
        )

    def __call__(self, image, label):
        image_np = np.array(image)
        label_np = np.array(label).astype(np.uint8)  # <- force correct type

        augmented = self.transform(image=image_np, mask=label_np)
        return augmented['image'], augmented['mask']
