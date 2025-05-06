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
        label_name = img_name 
        label_path = os.path.join(self.label_dir, label_name)
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.label_transform is not None:
            label = self.label_transform(label)

        label_array = np.array(label)
        label_array = label_array.astype(np.int32)
        label_tensor = torch.tensor(label_array)
        
        return image, label_tensor

    def __len__(self):
        return len(self.images)
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    from torchvision import transforms

    # Configurações iniciais
    root_dir = '/path/GTA5'  # Altere para o caminho real
    transform = transforms.ToTensor()
    label_transform = transforms.ToTensor()

    # Instancia o dataset
    dataset = GTA5(root_dir=root_dir, transform=transform, label_transform=None)

    # Seleciona um índice aleatório
    idx = random.randint(0, len(dataset) - 1)
    image, label_tensor = dataset[idx]

    # Converte a label para numpy para exibição (como máscara)
    label_np = label_tensor.numpy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image.permute(1, 2, 0))  # [C, H, W] -> [H, W, C]
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(label_np, cmap='gray')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')

    plt.show()
