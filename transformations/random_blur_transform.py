from .base_transform import BaseTransform
import albumentations as A

class RandomBlurTransform(BaseTransform):
    def __init__(self,
                 p=0.5,
                 height=512,
                 width=1024,
                 blur_limit=(3, 7)):
        super().__init__(height=height, width=width)
        self.transform = self._finalize([
            A.GaussianBlur(
                blur_limit=blur_limit,
                p=p
            )
        ])
