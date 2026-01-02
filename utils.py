import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image, Mask
from typing import Any, Dict, List, Sequence
from torchvision import tv_tensors as tvt
from PIL import Image

T.Resize


class RandomGamma(T.Transform):
    _transformed_types = (tvt.Image, Image.Image, torch.Tensor)

    def __init__(self, gamma_range=(0.5, 4)):
        super().__init__()
        self.gamma_range = gamma_range

    def transform(self, img, *args, **kwargs):
        gamma = np.random.uniform(*self.gamma_range)
        return T.functional.adjust_gamma(img, gamma)
    

class RandomContrast(T.Transform):
    _transformed_types = (tvt.Image, Image.Image, torch.Tensor)

    def __init__(self, contrast_range=(0.6, 3)):
        super().__init__()
        self.contrast_range = contrast_range

    def transform(self, img, *args, **kwargs):
        contrast = np.random.uniform(*self.contrast_range)
        return T.functional.adjust_contrast(img, contrast)


class SaltAndPepperNoise(T.Transform):
    _transformed_types = (tvt.Image, Image.Image, torch.Tensor)

    def __init__(self, amount=0.1, scale=1):
        super().__init__()
        self.amount = amount
        self.scale = scale

    def transform(self, img, *args, **kwargs):
        mask = torch.rand(*img.shape) < self.amount
        img[mask] = torch.randint(0, 256, (mask.sum(),)) / 255 * self.scale
        return img


class RandomSpeckleNoise(T.Transform):
    _transformed_types = (tvt.Image, Image.Image, torch.Tensor)

    def __init__(self, amount=(0.1, 0.5)):
        super().__init__()
        self.amount = amount

    def transform(self, img, *args, **kwargs):
        if img.min() < 0 or img.max() > 1:
            raise ValueError("Image must be in the range [0, 1]")
        amount = np.random.uniform(*self.amount)
        img = img + amount * img.std() * torch.randn(*img.shape)
        img = torch.clip(img, 0, 1)
        return img


class PixelAugmentations(T.Transform):
    _transformed_types = (tvt.Image, Image.Image, torch.Tensor)

    def __init__(
        self,
        gamma_range=None,
        contrast_range=None,
        salt_and_pepper_amount=None,
        speckle_noise_amount=None,
    ):
        super().__init__()
        self.augmentations = []
        if gamma_range is not None:
            self.augmentations.append(RandomGamma(gamma_range))
        if contrast_range is not None:
            self.augmentations.append(RandomContrast(contrast_range))
        if salt_and_pepper_amount is not None:
            self.augmentations.append(SaltAndPepperNoise(salt_and_pepper_amount))
        if speckle_noise_amount is not None:
            self.augmentations.append(RandomSpeckleNoise(speckle_noise_amount))

    def transform(self, img, *args, **kwargs):
        for aug in self.augmentations:
            img = aug(img)
        return img


def simple_build_pixel_augmentations(level='medium'): 
    if level == 'none':
        return T.Identity()
    elif level == 'low':
        return PixelAugmentations(
            gamma_range=(0.8, 1.2),
            contrast_range=(0.8, 1.2),
            salt_and_pepper_amount=0.01,
            speckle_noise_amount=None,
        )
    elif level == 'medium':
        return PixelAugmentations(
            gamma_range=(0.5, 1.5),
            contrast_range=(0.6, 1.5),
            salt_and_pepper_amount=0.05,
            speckle_noise_amount=0.1,
        )
    elif level == 'high':
        return PixelAugmentations(
            gamma_range=(0.3, 2.0),
            contrast_range=(0.4, 2.0),
            salt_and_pepper_amount=0.1,
            speckle_noise_amount=0.2,
        )
    else:
        raise ValueError(f"Unknown augmentation level: {level}")


def get_crop_to_mask_params(img_size, mask, threshold=0.5, padding=0):
    """Return the parameters to crop an image to the bounding box of a mask.

    Args:
        mask (np.ndarray): The mask to crop to.
        threshold (float): The threshold to apply to the mask.
        padding (int): The padding to add around the mask.

    Returns:
        Tuple[int, int, int, int]: The parameters to crop the image to the mask. (Top, Left, Height, Width)
    """
    if not isinstance(padding, Sequence):
        padding = [padding, padding]

    mask = mask > threshold
    x, y = np.where(mask)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_min = max(0, x_min - padding[0])
    x_max = min(img_size[0], x_max + padding[0])
    y_min = max(0, y_min - padding[1])
    y_max = min(img_size[1], y_max + padding[1])

    return x_min, y_min, x_max - x_min, y_max - y_min


class CropToMask(T.Transform): 

    def __init__(self, reference_mask_key, padding=0): 
        super().__init__()
        self.reference_mask_key = reference_mask_key
        self.padding = padding

    def forward(self, *inputs: Any) -> Any:
        if not isinstance(inputs[0], dict): 
            raise ValueError("Expected dict for this transform")
        
        # Mark the reference mask
        inputs[0][self.reference_mask_key]._is_reference_mask = True
        # Get all the tensors from the dict
        flat_inputs = list(inputs[0].values())
        # Get transform parameters
        params = self._get_params(flat_inputs)
        # Apply transform to each tensor in the dict
        output = {}
        for key, value in inputs[0].items():
            output[key] = self._transform(value, params)
            
        return output

    def _is_reference_mask(self, inp): 
        return getattr(inp, "_is_reference_mask", False)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        reference_mask = [inp for inp in flat_inputs if self._is_reference_mask(inp)][0]
        try: 
            top, left, height, width = get_crop_to_mask_params(reference_mask[0].shape, reference_mask[0], padding=self.padding)
        except: 
            # it is possible for there to be no mask. In that case, we should be a no-op.
            return {}

        return dict(
            top=top, 
            left=left, 
            height=height, 
            width=width
        )

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params: 
            return inpt
        return T.functional.crop(inpt, **params)
    
import torch
import random
import torchvision.transforms.functional as TF

class RandomNeedlePatch:
    """
    Extract random patches from the needle region of an ultrasound image.

    Args:
        patch_size (int or tuple): Size of the square patch (or (H,W)).
        num_patches (int): Number of patches to sample per image.
        resize_to (int or tuple, optional): Resize patches to this size for encoder.
        threshold (float): Mask threshold to consider as positive region.
    """
    def __init__(self, patch_size=128, num_patches=5, resize_to=None):
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size
        self.num_patches = num_patches
        self.resize_to = resize_to

    def __call__(self, bmode, needle_mask):
        """
        Args:
            bmode (Tensor): [C,H,W] image tensor
            needle_mask (Tensor): [1,H,W] binary mask tensor
        Returns:
            List[Tensor]: List of patches [C, patch_H, patch_W]
        """
        H, W = needle_mask.shape[1], needle_mask.shape[2]
        positive_coords = torch.nonzero(needle_mask[0])

        if len(positive_coords) == 0:
            # fallback: center crop
            top = (H - self.patch_size[0]) // 2
            left = (W - self.patch_size[1]) // 2
            patches = [TF.crop(bmode, top, left, self.patch_size[0], self.patch_size[1])]
        else:
            patches = []
            for _ in range(self.num_patches):
                idx = random.randint(0, len(positive_coords)-1)
                center_y, center_x = positive_coords[idx]  # note: row=y, col=x

                top = max(0, center_y - self.patch_size[0] // 2)
                left = max(0, center_x - self.patch_size[1] // 2)

                # ensure patch is fully inside image
                if top + self.patch_size[0] > H:
                    top = H - self.patch_size[0]
                if left + self.patch_size[1] > W:
                    left = W - self.patch_size[1]

                patch = TF.crop(bmode, top, left, self.patch_size[0], self.patch_size[1])
                if self.resize_to is not None:
                    patch = TF.resize(patch, self.resize_to)
                patches.append(patch)

        return patches
