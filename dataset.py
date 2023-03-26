import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class SegmentationDataset(Dataset):
    """A custom dataset class"""

    def __init__(self, images: list, masks: list, transforms: Compose) -> None:
        # Get paths to images and masks
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        # Return the number of images
        return len(self.images)

    def __getitem__(self, index: int) -> (Tensor, Tensor):
        # Get image and mask paths
        image_path = self.images[index]
        mask_path = self.masks[index]

        # Open image and convert to RGB
        image = Image.open(image_path).convert('RGB')
        # Open mask and convert to grayscale
        mask = Image.open(mask_path).convert('L')

        # Apply transforms
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
            # Converting float values to integers
            mask = mask * 255
            mask = mask.squeeze().to(torch.int64)
            # Ground truth labels are 1, 2, 3. therefore subtract one to achieve 0, 1, 2:
            mask -= 1

        # Return the image and corresponding mask
        return image, mask
