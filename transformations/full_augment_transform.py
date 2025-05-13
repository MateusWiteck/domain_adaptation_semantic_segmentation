from .base_transform import BaseTransform
import albumentations as A
import cv2

class FullAugmentTransform(BaseTransform):
    def __init__(self, p=0.5, height=512, width=1024,
                 rotation_limit=15, blur_limit=(3, 7),
                 max_holes=1, max_height=64, max_width=64,
                 erase_value=0, erase_mask=255,
                 ignore_index=255):
        super().__init__(height=height, width=width)

        self.transform = self._finalize([
            A.Rotate(
                limit=rotation_limit,
                p=p,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,                  # imagem (preto)
                fill_mask=ignore_index  # máscara (valor ignorado)
            ),
            A.HorizontalFlip(p=p),
            A.GaussianBlur(blur_limit=blur_limit, p=p),
            A.CoarseDropout(
                max_holes=max_holes,
                max_height=max_height,
                max_width=max_width,
                fill=erase_value,         # imagem (preto)
                fill_mask=erase_mask,     # máscara (ignorado)
                p=p
            )
        ])
