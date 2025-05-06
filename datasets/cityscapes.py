from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
from view_label_human import visualize_label_with_colors


class CityScapes(Dataset):
    def __init__(self, root_dir, split='train', transform=None, label_transform=None):
        super(CityScapes, self).__init__()
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)
        self.transform = transform
        self.label_transform = label_transform

        # Recursive search to find all image files
        self.image_paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith('.png'): 
                    self.image_paths.append(os.path.join(root, file))
                else:
                    print(f"Skipping non-image file: {file}")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Create the corresponding label path
        label_path = img_path.replace('leftImg8bit', 'gtFine_labelTrainIds')
        label_path = label_path.replace('/images/', '/gtFine/')

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.label_transform is not None:
            label = self.label_transform(label)

        label_array = np.array(label).astype(np.int32)
        label_tensor = torch.tensor(label_array)

        return image, label_tensor

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    from torchvision import transforms

    # Path to your dataset (adjust here)
    root_dir = 'data/Cityscapes/Cityspaces'

    # Simple transforms (ToTensor only for the image)
    transform = transforms.ToTensor()

    # Instantiate the dataset
    dataset = CityScapes(root_dir=root_dir, split='train', transform=transform, label_transform=None)

    # Select a random index
    idx = random.randint(0, len(dataset) - 1)
    image, label_tensor = dataset[idx]

    # Convert the label to numpy for display
    label_np = label_tensor.numpy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image.permute(1, 2, 0))  # Convert from [C, H, W] to [H, W, C]
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(label_np, cmap='gray', vmin=0, vmax=20) 
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')

    plt.show()

    #print("Output in numpy array format:")
    #print(label_np)
    #print("Output in tensor format:")   
    #print(label_tensor)
    #print("Shape of label tensor:")
    #print(label_tensor.shape)

    # Plot the image in the color pattern given by the GTA dataset
    visualize_label_with_colors(label_np)

