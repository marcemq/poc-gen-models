import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from datasets import load_dataset
from torchvision import transforms
from datasets import features
import torch

class CheckerboardDataset(Dataset):
    def __init__(self, cfg_ds):
        """
        Args:
            N: Number of points to sample
            x_min and x_max: min and max values over x axis
            y_min and y_max: min and max values over y axis
            length: length of checkboard pattern
        """
        self.N = cfg_ds.N
        self.x_min = cfg_ds.X_MIN
        self.x_max = cfg_ds.X_MAX
        self.y_min = cfg_ds.Y_MIN
        self.y_max = cfg_ds.Y_MAX
        self.length = cfg_ds.LENGTH
        # Checkerboard pattern
        self.checkerboard_pattern = np.indices((self.length, self.length)).sum(axis=0) % 2
        self.sampled_points = self._sample_checkerboard_data()

    def _sample_checkerboard_data(self):
        """
        Return a ndarray of sampled points that follows a checkerboard pattern
        """
        sampled_points = []
        # Sample points in regions where checkerboard pattern is 1
        while len(sampled_points) < self.N:
            # Randomly sample a point within the x and y range
            x_sample = np.random.uniform(self.x_min, self.x_max)
            y_sample = np.random.uniform(self.y_min, self.y_max)

            # Determine the closest grid index
            i = int((x_sample - self.x_min) / (self.x_max - self.x_min) * self.length)
            j = int((y_sample - self.y_min) / (self.y_max - self.y_min) * self.length)

            # Check if the sampled point is in a region where checkerboard == 1
            if self.checkerboard_pattern[j, i] == 1:
                sampled_points.append((x_sample, y_sample))

        # Convert to NumPy array for easier plotting
        sampled_points = np.array(sampled_points)
        logging.info(f'Sampled points shape:{sampled_points.shape}')
        return sampled_points

    def __len__(self):
        return len(self.sampled_points)

    def __getitem__(self, idx):
        return torch.tensor(self.sampled_points[idx])
    
class CustomTransform:
    def __init__(self, image_size):
        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Normalize the color values in the range [-1,1]
            transforms.Normalize([0.5], [0.5]),
        ])

    def __call__(self, image):
        return self.preprocess(image.convert("RGB"))  # Ensure it's a PIL image

def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    tensors = tensors.cpu()  # Debugging: move to CPU
    assert torch.isfinite(tensors).all(), "NaN or Inf detected in tensors"
    return (((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0)

class ButterfliesDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        dataset_path = "huggan/smithsonian_butterflies_subset"
        self.raw_dataset = load_dataset(dataset_path, split="train")
        
        # Ensure images are properly loaded
        self.raw_dataset = self.raw_dataset.cast_column("image", features.Image())

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        image = self.raw_dataset[idx]["image"]  # Get PIL Image
        
        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image
