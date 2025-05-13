from .base_transform import BaseTransform
import albumentations as A

class RandomFlipTransform(BaseTransform):
    def __init__(self, p=0.5, height=512, width=1024):
        super().__init__(height=height, width=width)
        self.transform = self._finalize([
            A.HorizontalFlip(p=p)
        ])
