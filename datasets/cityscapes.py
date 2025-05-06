from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch

class CityScapes(Dataset):
    def __init__(self, root_dir, split='train', transform=None, label_transform=None):
        super(CityScapes, self).__init__()
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'gtFine', split)
        self.transform = transform
        self.label_transform = label_transform

        # Busca recursiva para encontrar todos os arquivos de imagem
        self.image_paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith('.png'):  # ajuste se sua extensão for diferente
                    self.image_paths.append(os.path.join(root, file))

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Cria o caminho da label correspondente
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

    # Caminho para o seu dataset (ajuste aqui)
    root_dir = 'domain_adaptation_semantic_segmentation/data/Cityscapes/Cityspaces'

    # Transforms simples (ToTensor só para a imagem)
    transform = transforms.ToTensor()

    # Instancia o dataset
    dataset = CityScapes(root_dir=root_dir, split='train', transform=transform, label_transform=None)

    # Seleciona um índice aleatório
    idx = random.randint(0, len(dataset) - 1)
    image, label_tensor = dataset[idx]

    # Converte a label para numpy para exibição
    label_np = label_tensor.numpy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image.permute(1, 2, 0))  # Converte de [C, H, W] para [H, W, C]
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(label_np, cmap='gray')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')

    plt.show()
