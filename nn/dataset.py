import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision.io import read_image
from torchvision import transforms

from nn.image_state_model import ImageProperties

EVAL_TRANSFORMER = transforms.Resize((128, 128))
TRAIN_TRANSFORMER = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale()
])


def get_test_data_loaders(root: str):
    test_dataset = ColorizerImageDataset(root=root)
    return DataLoader(test_dataset, batch_size=4, num_workers=4)


def check_shapes(data_loader: DataLoader):
    for X, y in data_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape} {X.dtype}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    print('shape>', images.shape)

    dataiter = iter(data_loader)
    original, eval = next(dataiter)
    print('shape>', original.shape, eval.shape)


def get_data_loaders(img_root, batch_size, workers, max_image=None):
    dataset = ColorizerImageDataset(
        root=img_root,
        max_image=max_image
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        collate_fn=custom_collate_fn
    )


def get_test_image(path):
    image_gray = TRAIN_TRANSFORMER(Image.open(path))
    to_tensor = transforms.ToTensor()
    image_gray = to_tensor(image_gray)
    ground_truth = read_image(path).float() / 255.0

    return ImageProperties(image_gray, EVAL_TRANSFORMER(ground_truth))


def get_test_images(root: str, number_of_image: int):
    images = list(os.listdir(root))
    if number_of_image is not None:
        images = images[:number_of_image]
    return [get_test_image(os.path.join(root, path)) for path in images]


def custom_collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.empty(0)
    return default_collate(batch)


class ColorizerImageFilesDataset(Dataset):
    def __init__(self, root, images, train_transformer, eval_transformer):
        self.root = root
        self.images = images
        self.train_transformer = train_transformer
        self.eval_transformer = eval_transformer

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        img_original = read_image(img_path).float() / 255.0

        # filter out grayscale image
        if img_original.shape[0] == 1:
            return None

        img = self.train_transformer(img_original)
        target = self.eval_transformer(img_original)
        return img, target

    def __len__(self):
        return len(self.images)


class ColorizerImageDataset(ColorizerImageFilesDataset):
    def __init__(self, root, train_transformer=TRAIN_TRANSFORMER, eval_transformer=EVAL_TRANSFORMER, max_image=None):
        self.root = root
        self.train_transformer = train_transformer
        self.eval_transformer = eval_transformer
        images = list(sorted(os.listdir(root)))
        if max_image is not None:
            images = images[:max_image]

        super().__init__(root, images, train_transformer, eval_transformer)


class ImageDatasetSharder:
    def __init__(self, root, batch_size=4, workers=2, shards=2):
        self.current = 0
        self.root = root
        self.batch_size = batch_size
        self.shards = shards
        self.workers = workers
        self.images = list(sorted(os.listdir(root)))
        self.data_size = len(self.images)
        self.chunks = np.array_split(self.images, shards)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.shards:
            images_chunk = self.chunks[self.current]
            dataset = ColorizerImageFilesDataset(self.root, images_chunk, TRAIN_TRANSFORMER, EVAL_TRANSFORMER)
            test_dataloader = DataLoader(dataset, collate_fn=custom_collate_fn)
            self.current += 1
            return test_dataloader, self.current
        else:
            raise StopIteration
