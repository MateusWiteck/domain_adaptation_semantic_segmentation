from .base_transform import BaseTransform
import albumentations as A

class RandomEraseTransform(BaseTransform):
    def __init__(self, p=0.5, height=512, width=1024,
                 max_holes=1, max_height=64, max_width=64,
                 erase_value=0, erase_mask=255):
        super().__init__(height=height, width=width)
        self.transform = self._finalize([
            A.CoarseDropout(
                max_holes=max_holes,
                max_height=max_height,
                max_width=max_width,
                fill=erase_value,       # imagem (0 = preto)
                fill_mask=erase_mask,   # máscara (255 = ignorar)
                p=p
            )
        ])
