from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from .GTA5Label import GTA5Labels_TaskCV2017
from .view_label_human import visualize_label_with_colors


class GTA5(Dataset):
    def __init__(self, root_dir, transform):
        super(GTA5, self).__init__()
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.transform = transform
        self.images = os.listdir(self.image_dir)

        # Precompute the color → ID mapping once
        self._color_to_id = {label.color: label.ID for label in GTA5Labels_TaskCV2017.list_}

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_name = img_name  # Assumes label file has the same name as the image
        label_path = os.path.join(self.label_dir, label_name)

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        # Convert the label if needed
        label = self._convert_palette_to_class_ids(label)

        image, label = self.transform(image, label) 

        #label_tensor = label.long()
        return image, label


    def _convert_palette_to_class_ids(self, label_img):
        """
        Convert a palette ('P') mode image to 'L' mode with class IDs.

        Args:
            label_img (PIL.Image): The input label image.

        Returns:
            PIL.Image: 'L' mode image with class IDs.
        """
        if label_img.mode != 'P':
            # If already not palette mode, return as is
            return label_img

        # Convert palette to RGB
        label_rgb = label_img.convert('RGB')
        label_rgb_array = np.array(label_rgb)

        # Prepare the output array
        label_id_array = np.full(label_rgb_array.shape[:2], fill_value=255, dtype=np.uint8)

        # Map each color to its class ID
        for color, class_id in self._color_to_id.items():
            mask = np.all(label_rgb_array == color, axis=-1)
            label_id_array[mask] = class_id

        return Image.fromarray(label_id_array, mode='L')
    
    def __len__(self):
        return len(self.images) 


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    from torchvision import transforms

    # Path to the dataset (adjust as needed)
    root_dir = 'data/GTA5'
    transform = transforms.ToTensor()

    # Initialize dataset
    dataset = GTA5(root_dir=root_dir, transform=transform)

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

    # Display grayscale label
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

    # Print dataset information
    print(f"Dataset length: {len(dataset)}")
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label_tensor.shape}")
    