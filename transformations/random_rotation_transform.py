from .base_transform import BaseTransform
import albumentations as A
import cv2                           # OpenCV enums

IGNORE_INDEX = 255                    # ← pretend the ignore class is 1

class RandomRotationTransform(BaseTransform):
    def __init__(self,
                 rotation_limit,
                 p=0.5,
                 height=512,
                 width=1024):
        super().__init__(height=height, width=width)

        self.transform = self._finalize([
            A.Rotate(
                limit=rotation_limit,
                p=p,
                interpolation=cv2.INTER_LINEAR,       # image interpolation
                mask_interpolation=cv2.INTER_NEAREST, # keep labels intact
                border_mode=cv2.BORDER_CONSTANT,      # constant padding
                fill=0,                               # image fill (black)
                fill_mask=IGNORE_INDEX                # mask fill (ignore)
            )
        ])
