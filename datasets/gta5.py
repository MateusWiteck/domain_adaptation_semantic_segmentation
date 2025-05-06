from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch


class GTA5(Dataset):
    def __init__(self, root_dir, transform=None, label_transform=None):
        super(GTA5, self).__init__()
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.transform = transform
        self.label_transform = label_transform
        self.images = os.listdir(self.image_dir)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_name = img_name  # Assumes label file has the same name as the image
        label_path = os.path.join(self.label_dir, label_name)
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.label_transform is not None:
            label = self.label_transform(label)

        # Convert label to tensor
        label_array = np.array(label).astype(np.int32)
        label_tensor = torch.tensor(label_array)
        
        return image, label_tensor

    def __len__(self):
        return len(self.images)
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    from torchvision import transforms

    # Path to the dataset (adjust as needed)
    root_dir = 'domain_adaptation_semantic_segmentation/data/GTA5'
    transform = transforms.ToTensor()

    # Initialize dataset
    dataset = GTA5(root_dir=root_dir, transform=transform, label_transform=None)

    # Select a random sample
    random.seed(42)  # Ensures reproducibility
    idx = random.randint(0, len(dataset) - 1)
    image, label_tensor = dataset[idx]

    # Convert label to numpy for display
    label_np = label_tensor.numpy()

    # Plot image and label
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image.permute(1, 2, 0))  # [C, H, W] -> [H, W, C]
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(label_np, cmap='gray')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')

    plt.show()

    print("Output in numpy array format:")
    print(label_np)
    print("Output in tensor format:")   
    print(label_tensor)
    print("Shape of label tensor:")
    print(label_tensor.shape)
