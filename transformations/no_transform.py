from .base_transform import BaseTransform
import albumentations as A

class NoTransform(BaseTransform):
    def __init__(self, height=512, width=1024):
        super().__init__(height=height, width=width)
        self.transform = self._finalize([
            A.NoOp()
        ])
