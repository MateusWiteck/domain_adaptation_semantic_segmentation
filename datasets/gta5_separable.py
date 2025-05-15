from torch.utils.data import Dataset


class GTA5WithAug(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.base_dataset)
