import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from transformations import RandomFlipTransform, NoTransform, RandomRotationTransform, RandomEraseTransform, RandomBlurTransform
from datasets.gta5 import GTA5

# Fix seed for reproducibility
torch.manual_seed(0)

# === With augmentation ===
transform_aug = RandomRotationTransform(p=1.0, rotation_limit=70) # Change Here--------
dataset_aug = GTA5(root_dir='data/GTA5', transform=transform_aug)

# Load specific sample (e.g., index 0)
image_aug, label_aug = dataset_aug[1]

# === Without augmentation ===
transform_none = NoTransform()
dataset_plain = GTA5(root_dir='data/GTA5', transform=transform_none)

# Load the same sample (index 0)
image_plain, label_plain = dataset_plain[1]

# === Convert to numpy for plotting ===
image_aug_np = image_aug.permute(1, 2, 0).numpy()
label_aug_np = label_aug.numpy()

image_plain_np = image_plain.permute(1, 2, 0).numpy()
label_plain_np = label_plain.numpy()

# === Visualization ===
cmap = plt.cm.get_cmap('gray', 21)
cmap.set_over('red')

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(image_plain_np)
axes[0, 0].set_title("Image (no aug)")
axes[0, 0].axis("off")

axes[0, 1].imshow(label_plain_np, cmap=cmap, vmin=0, vmax=20)
axes[0, 1].set_title("Label (no aug)")
axes[0, 1].axis("off")

axes[1, 0].imshow(image_aug_np)
axes[1, 0].set_title("Image (augmented)")
axes[1, 0].axis("off")

axes[1, 1].imshow(label_aug_np, cmap=cmap, vmin=0, vmax=20)
axes[1, 1].set_title("Label (augmented)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.show()
