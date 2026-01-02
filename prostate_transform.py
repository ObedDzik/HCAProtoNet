import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode
from torchvision.tv_tensors import Image, Mask
from typing import Any, Dict, List, Sequence
from torchvision import tv_tensors as tvt
from utils import CropToMask, RandomGamma, RandomContrast, RandomNeedlePatch
from scipy.ndimage import gaussian_filter


psa_min = 0.2
psa_max = 32.95
psa_avg = 6.821426488456866
age_min = 0
age_max = 79
age_avg = 62.5816
approx_psa_density_min = 4.615739672282483e-06
approx_psa_density_max = 0.000837278201784
approx_psa_density_avg = 0.000175347951594383


CORE_LOCATION_TO_IDX = {
    "LML": 0,
    "RBL": 1,
    "LMM": 2,
    "RMM": 2,
    "LBL": 1,
    "LAM": 3,
    "RAM": 3,
    "RML": 0,
    "LBM": 4,
    "RAL": 5,
    "RBM": 4,
    "LAL": 5,
}


# class ProstateTransform:
#     def __init__(
#         self,
#         augment="none",
#         image_size=1024,
#         mask_size=256,
#         mean=(0, 0, 0),
#         std=(1, 1, 1),
#         normalize=True,
#         crop_to_prostate=True, 
#         crop_to_needle=False,
#         first_downsample_size=None,
#         return_raw_images=False,
#         grade_group_for_positive_label=1,
#         flip_ud=False, 
#     ):
#         self.augmentations = augment
#         self.image_size = image_size
#         self.mask_size = mask_size
#         self.mean = mean
#         self.std = std
#         self.normalize = normalize
#         self.crop_to_prostate = crop_to_prostate
#         self.crop_to_needle = crop_to_needle
#         self.first_downsample_size = first_downsample_size
#         self.return_raw_images = return_raw_images
#         self.grade_group_for_positive_label = grade_group_for_positive_label
#         self.flip_ud = flip_ud

#     def _coerce_input(self, item):
#         if 'image' in item:
#             # this is from the OPTMUM needle dataset format. We need to convert it to the standard format.
#             item = _ProstateDatasetAdapterOptimum()(item)
#         return item

#     def __call__(self, item):
#         item = self._coerce_input(item)
#         out = item.copy()

#         bmode = item["bmode"]
#         needle_mask = item.get("needle_mask", np.zeros((bmode.shape[:2]), np.uint8))
#         prostate_mask = item.get("prostate_mask", np.ones((bmode.shape[:2]), np.uint8))

#         if self.flip_ud:
#             bmode = np.flipud(bmode).copy()
#             needle_mask = np.flipud(needle_mask).copy()
#             prostate_mask = np.flipud(prostate_mask).copy()

#         if self.return_raw_images:
#             out['bmode_raw'] = bmode.copy()
#             out['needle_mask_raw'] = needle_mask.copy() 
#             out['prostate_mask_raw'] = prostate_mask.copy()

#         bmode_tensor = torch.from_numpy(bmode.copy()).float().unsqueeze(0).repeat(3,1,1)
#         bmode_tensor = Image((bmode_tensor - bmode_tensor.min()) / (bmode_tensor.max() - bmode_tensor.min()))
#         if self.normalize:
#             bmode_tensor = T.Normalize(self.mean, self.std)(bmode_tensor)

#         needle_mask_tensor = Mask(torch.from_numpy(needle_mask.copy()).float().unsqueeze(0))
#         prostate_mask_tensor = Mask(torch.from_numpy(prostate_mask.copy()).float().unsqueeze(0))

#         needle_patches = []
#         if getattr(self, "crop_to_needle", False):
#             needle_patches = RandomNeedlePatch(
#                 patch_size=128,       
#                 num_patches=5,
#                 resize_to=self.image_size 
#             )(bmode_tensor, needle_mask_tensor)

#         augmented_patches = []
#         for patch in needle_patches:
#             if "gamma" in self.augmentations:
#                 patch = T.RandomApply([RandomGamma((0.6,3))])(patch)
#             if "contrast" in self.augmentations:
#                 patch = T.RandomApply([RandomContrast((0.7,2))])(patch)
#             if "translate" in self.augmentations:
#                 patch = T.RandomAffine([0,0],[0.05,0.05])(patch)
#             augmented_patches.append(patch)

#         needle_patches = augmented_patches

#         bmode_tensor, needle_mask_tensor, prostate_mask_tensor = T.Resize(
#             (self.image_size, self.image_size), antialias=True
#         )(bmode_tensor, needle_mask_tensor, prostate_mask_tensor)

#         if self.crop_to_prostate:
#             bmode_tensor, needle_mask_tensor, prostate_mask_tensor = CropToMask('prostate_mask', 16)(
#                 dict(bmode=bmode_tensor, needle_mask=needle_mask_tensor, prostate_mask=prostate_mask_tensor)
#             ).values()

#         if "translate" in self.augmentations:
#             bmode_tensor, needle_mask_tensor, prostate_mask_tensor = T.RandomAffine([0,0],[0.2,0.2])(
#                 bmode_tensor, needle_mask_tensor, prostate_mask_tensor
#             )
#         if "random_crop" in self.augmentations and (torch.rand(1).item() > 0.5):
#             bmode_tensor, needle_mask_tensor, prostate_mask_tensor = T.RandomResizedCrop(
#                 (self.image_size, self.image_size),
#                 scale=(0.3,1)
#             )(bmode_tensor, needle_mask_tensor, prostate_mask_tensor)
#         else:
#             bmode_tensor = T.Resize((self.image_size, self.image_size))(bmode_tensor)
#         if "gamma" in self.augmentations:
#             bmode_tensor = T.RandomApply([RandomGamma((0.6,3))])(bmode_tensor)
#         if "contrast" in self.augmentations:
#             bmode_tensor = T.RandomApply([RandomContrast((0.7,2))])(bmode_tensor)

#         needle_mask_tensor = T.Resize(
#             (self.mask_size, self.mask_size),
#             antialias=False, interpolation=InterpolationMode.NEAREST
#         )(needle_mask_tensor)
#         prostate_mask_tensor = T.Resize(
#             (self.mask_size, self.mask_size),
#             antialias=False, interpolation=InterpolationMode.NEAREST
#         )(prostate_mask_tensor)

#         out["bmode"] = bmode_tensor
#         out["needle_mask"] = needle_mask_tensor
#         out["prostate_mask"] = prostate_mask_tensor
#         out["needle_patches"] = needle_patches

#         if "grade_group" in item:
#             label = item["grade_group"]
#             if torch.isnan(torch.tensor(label)):
#                 label = 0
#             out["label"] = torch.tensor(label).long()
#         if "primus" in item:
#             out["primus_label"] = torch.tensor(item["primus"]-1).long() 

#         if "psa" in item:
#             psa = item["psa"] if not np.isnan(item["psa"]) else psa_avg
#             out["psa"] = torch.tensor([(psa - psa_min) / (psa_max - psa_min)]).float()
#         if "age" in item:
#             age = item["age"] if not np.isnan(item["age"]) else age_avg
#             out["age"] = torch.tensor([(age - age_min) / (age_max - age_min)]).float()
#         if "approx_psa_density" in item:
#             approx_psa_density = item["approx_psa_density"] if not np.isnan(item["approx_psa_density"]) else approx_psa_density_avg
#             out["approx_psa_density"] = torch.tensor([(approx_psa_density - approx_psa_density_min) / (approx_psa_density_max - approx_psa_density_min)]).float()

#         return out




class ProstateTransform:
    def __init__(
        self,
        augment="all",
        image_size=224,
        mask_size=256,
        mean=(0, 0, 0),
        std=(1, 1, 1),
        normalize=True,
        crop_to_prostate=True, 
        crop_to_needle=False,
        first_downsample_size=None,
        return_raw_images=False,
        grade_group_for_positive_label=1,
        flip_ud=False,
        augment_strength='light',  # Changed default to 'light' for ProtoViT
    ):
        self.augmentations = augment
        self.image_size = image_size
        self.mask_size = mask_size
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.crop_to_prostate = crop_to_prostate
        self.crop_to_needle = crop_to_needle
        self.first_downsample_size = first_downsample_size
        self.return_raw_images = return_raw_images
        self.grade_group_for_positive_label = grade_group_for_positive_label
        self.flip_ud = flip_ud
        self.augment_strength = augment_strength
        self.aug_strength = 1.0
        
        # Define augmentation parameters based on strength
        self._set_augment_params()

    def _coerce_input(self, item):
        if 'image' in item:
            # this is from the OPTIMUM needle dataset format. We need to convert it to the standard format.
            item = _ProstateDatasetAdapterOptimum()(item)
        return item
    
    def set_aug_strength(self, strength):
        self.aug_strength = float(strength)

    def _set_augment_params(self):
        """Set augmentation parameters based on strength level - tuned for ProtoViT"""
        if self.augment_strength == 'light':
            self.aug_params = {
                'rotation_degrees': 10,
                'translate': (0.05, 0.05),
                'scale': (0.9, 1.1),
                'gamma_range': (0.85, 1.15),
                'contrast_range': (0.85, 1.15),
                'brightness_delta': 0.08,
                'noise_std': 0.015,
                'blur_sigma_range': (0.1, 0.5),
                'sharpness_factor': 1.5,
                'geometric_prob': 0.3,
                'intensity_prob': 0.25,
                'cutout_prob': 0.15,
                'cutout_size': (0.1, 0.2),  # fraction of image
            }
        elif self.augment_strength == 'medium':
            self.aug_params = {
                'rotation_degrees': 15,
                'translate': (0.1, 0.1),
                'scale': (0.85, 1.15),
                'gamma_range': (0.75, 1.3),
                'contrast_range': (0.8, 1.25),
                'brightness_delta': 0.12,
                'noise_std': 0.025,
                'blur_sigma_range': (0.1, 1.0),
                'sharpness_factor': 1.8,
                'geometric_prob': 0.4,
                'intensity_prob': 0.35,
                'cutout_prob': 0.25,
                'cutout_size': (0.15, 0.25),
            }
        elif self.augment_strength == 'strong':
            self.aug_params = {
                'rotation_degrees': 30, #20,
                'translate': (0.25, 0.25), #(0.15, 0.15),
                'scale': (0.7, 1.3), #(0.8, 1.2),
                'shear_range': (-8.0 * self.aug_strength, 8.0 * self.aug_strength),
                'gamma_range': (0.5, 1.6), #(0.7, 1.4),
                'contrast_range':  (0.6, 1.5), #(0.7, 1.3),
                'brightness_delta': 0.25, #0.15,
                'noise_std': 0.06, #0.035,
                'blur_sigma_range': (0.1, 2.2), #(0.1, 1.5),
                'sharpness_factor': 2.5, #2.0,
                'geometric_prob': 0.75, #0.6, #0.5,
                'intensity_prob': 0.75, #0.6, #0.4,
                'cutout_prob': 0.6, #0.4, #0.3,
                'cutout_size': (0.25, 0.45), #(0.3, 0.4) #(0.2, 0.3),
            }

        else:
            self.aug_params = None

    def _apply_cutout(self, image_tensor, cutout_size):
        """Apply random cutout/erasing to encourage diverse prototype learning"""
        c, h, w = image_tensor.shape
        
        # Random cutout size
        cutout_h = int(h * torch.FloatTensor(1).uniform_(*cutout_size).item())
        cutout_w = int(w * torch.FloatTensor(1).uniform_(*cutout_size).item())
        
        # Random position
        top = torch.randint(0, h - cutout_h + 1, (1,)).item()
        left = torch.randint(0, w - cutout_w + 1, (1,)).item()
        
        # Fill with random gray value (realistic for ultrasound)
        fill_value = torch.FloatTensor(1).uniform_(0.3, 0.7).item()
        image_tensor[:, top:top+cutout_h, left:left+cutout_w] = fill_value
        
        return image_tensor

    def _apply_augmentations(self, bmode_tensor, needle_mask_tensor, prostate_mask_tensor, is_training=True):
        """Apply augmentations with proper probability control and correct implementations"""
        
        if not is_training or self.augmentations == "none" or self.aug_params is None:
            return bmode_tensor, needle_mask_tensor, prostate_mask_tensor
        
        geo_p = self.aug_params['geometric_prob'] * self.aug_strength
        int_p = self.aug_params['intensity_prob'] * self.aug_strength
        cutout_p = self.aug_params['cutout_prob'] * self.aug_strength
        self.aug_params['shear_range'] = [deg * self.aug_strength for deg in self.aug_params['shear_range']]
       
        # 0. RANDOM RESIZED CROP (before affine)
        if torch.rand(1).item() < 0.6:
            i, j, h, w = T.RandomResizedCrop.get_params(
                bmode_tensor,
                scale=(0.6, 1.0),
                ratio=(0.85, 1.15)
            )

            bmode_tensor = T.functional.resized_crop(
                bmode_tensor, i, j, h, w,
                size=bmode_tensor.shape[-2:],
                interpolation=InterpolationMode.BILINEAR
            )
            needle_mask_tensor = T.functional.resized_crop(
                needle_mask_tensor, i, j, h, w,
                size=bmode_tensor.shape[-2:],
                interpolation=InterpolationMode.NEAREST
            )
            prostate_mask_tensor = T.functional.resized_crop(
                prostate_mask_tensor, i, j, h, w,
                size=bmode_tensor.shape[-2:],
                interpolation=InterpolationMode.NEAREST
            )

        # 1. GEOMETRIC AUGMENTATIONS (translation + scale + shear)
        if torch.rand(1).item() < geo_p:
            translate = self.aug_params['translate']
            scale = self.aug_params['scale']
            shear_range = self.aug_params.get('shear_range', (-5.0 , 5.0))

            scale_factor = torch.FloatTensor(1).uniform_(*scale).item()
            translate_x = torch.FloatTensor(1).uniform_(-translate[0], translate[0]).item()
            translate_y = torch.FloatTensor(1).uniform_(-translate[1], translate[1]).item()

            # Shear in degrees (x and y independently)
            shear_x = torch.FloatTensor(1).uniform_(*shear_range).item()
            shear_y = torch.FloatTensor(1).uniform_(*shear_range).item()

            affine_params = {
                'angle': 0.0,
                'translate': (
                    int(translate_x * bmode_tensor.shape[-1]),
                    int(translate_y * bmode_tensor.shape[-2])
                ),
                'scale': scale_factor,
                'shear': [shear_x, shear_y]
            }

            # Apply SAME affine to all tensors
            bmode_tensor = T.functional.affine(
                bmode_tensor,
                interpolation=InterpolationMode.BILINEAR,
                **affine_params
            )

            needle_mask_tensor = T.functional.affine(
                needle_mask_tensor,
                interpolation=InterpolationMode.NEAREST,
                **affine_params
            )

            prostate_mask_tensor = T.functional.affine(
                prostate_mask_tensor,
                interpolation=InterpolationMode.NEAREST,
                **affine_params
            )

        
        # 2. INTENSITY AUGMENTATIONS (apply to image only)
        # Reduced frequency compared to original to preserve diagnostic features
        
        # Random gamma correction (simulates different ultrasound gain settings)
        if torch.rand(1).item() < int_p:
            gamma = torch.FloatTensor(1).uniform_(*self.aug_params['gamma_range']).item()
            # Clamp to [0, 1] before applying gamma
            bmode_tensor = torch.clamp(bmode_tensor, 0, 1)
            bmode_tensor = torch.pow(bmode_tensor, gamma)
        
        # Random contrast (simulates different ultrasound TGC settings)
        if torch.rand(1).item() < int_p:
            contrast = torch.FloatTensor(1).uniform_(*self.aug_params['contrast_range']).item()
            mean = bmode_tensor.mean(dim=[1, 2], keepdim=True)
            bmode_tensor = (bmode_tensor - mean) * contrast + mean
            bmode_tensor = torch.clamp(bmode_tensor, 0, 1)
        
        # Random brightness
        if torch.rand(1).item() < int_p:
            brightness_factor = torch.FloatTensor(1).uniform_(
                1.0 - self.aug_params['brightness_delta'], 
                1.0 + self.aug_params['brightness_delta']
            ).item()
            bmode_tensor = bmode_tensor * brightness_factor
            bmode_tensor = torch.clamp(bmode_tensor, 0, 1)
        
        # Random Gaussian noise (simulates ultrasound speckle noise)
        if torch.rand(1).item() < int_p * 0.7:
            noise = torch.randn_like(bmode_tensor) * self.aug_params['noise_std']
            bmode_tensor = bmode_tensor + noise
            bmode_tensor = torch.clamp(bmode_tensor, 0, 1)
        
        # Random Gaussian blur (simulates different ultrasound frequencies)
        if torch.rand(1).item() < int_p * 0.5:
            sigma = torch.FloatTensor(1).uniform_(*self.aug_params['blur_sigma_range']).item()
            # Use odd kernel size based on sigma
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
            kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
            bmode_tensor = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(bmode_tensor)
        
        # Random sharpening (mutually exclusive with blur in same augmentation)
        elif torch.rand(1).item() < int_p * 0.3:
            sharpness = self.aug_params['sharpness_factor']
            bmode_tensor = T.functional.adjust_sharpness(bmode_tensor, sharpness)
        
        # Random cutout/erasing (helps ProtoViT learn diverse prototypes)
        if torch.rand(1).item() < cutout_p:
            bmode_tensor = self._apply_cutout(bmode_tensor, self.aug_params['cutout_size'])
        
        return bmode_tensor, needle_mask_tensor, prostate_mask_tensor

    def __call__(self, item):
        item = self._coerce_input(item)
        out = item.copy()

        bmode = item["bmode"]
        needle_mask = item.get("needle_mask", np.zeros((bmode.shape[:2]), np.uint8))
        prostate_mask = item.get("prostate_mask", np.ones((bmode.shape[:2]), np.uint8))

        if self.flip_ud:
            bmode = np.flipud(bmode).copy()
            needle_mask = np.flipud(needle_mask).copy()
            prostate_mask = np.flipud(prostate_mask).copy()

        if self.return_raw_images:
            out['bmode_raw'] = bmode.copy()
            out['needle_mask_raw'] = needle_mask.copy() 
            out['prostate_mask_raw'] = prostate_mask.copy()

        # Convert to tensor
        bmode_tensor = torch.from_numpy(bmode.copy()).float().unsqueeze(0).repeat(3, 1, 1)
        bmode_tensor = (bmode_tensor - bmode_tensor.min()) / (bmode_tensor.max() - bmode_tensor.min() + 1e-8)
        
        needle_mask_tensor = torch.from_numpy(needle_mask.copy()).float().unsqueeze(0)
        prostate_mask_tensor = torch.from_numpy(prostate_mask.copy()).float().unsqueeze(0)

        # IMPROVED ORDER: Crop to prostate first (on full resolution)
        if self.crop_to_prostate:
            bmode_tensor, needle_mask_tensor, prostate_mask_tensor = CropToMask('prostate_mask', 16)(
                dict(bmode=Image(bmode_tensor), 
                     needle_mask=Mask(needle_mask_tensor), 
                     prostate_mask=Mask(prostate_mask_tensor))
            ).values()

        # APPLY AUGMENTATIONS (on cropped, full-resolution images)
        is_training = self.augmentations != "none"
        bmode_tensor, needle_mask_tensor, prostate_mask_tensor = self._apply_augmentations(
            bmode_tensor, needle_mask_tensor, prostate_mask_tensor, is_training=is_training
        )

        # THEN resize to target size (after augmentation)
        bmode_tensor = T.Resize((self.image_size, self.image_size), 
                               interpolation=InterpolationMode.BILINEAR,
                               antialias=True)(bmode_tensor)
        needle_mask_tensor = T.Resize((self.image_size, self.image_size), 
                                     interpolation=InterpolationMode.NEAREST,
                                     antialias=False)(needle_mask_tensor)
        prostate_mask_tensor = T.Resize((self.image_size, self.image_size), 
                                       interpolation=InterpolationMode.NEAREST,
                                       antialias=False)(prostate_mask_tensor)

        # Handle needle patches (if needed)
        needle_patches = []
        if self.crop_to_needle:
            needle_patches = RandomNeedlePatch(
                patch_size=128,       
                num_patches=5,
                resize_to=self.image_size 
            )(Image(bmode_tensor), Mask(needle_mask_tensor))

        # Normalize AFTER augmentation
        if self.normalize:
            bmode_tensor = T.Normalize(self.mean, self.std)(bmode_tensor)

        # Resize masks to mask_size
        needle_mask_tensor = T.Resize(
            (self.mask_size, self.mask_size),
            interpolation=InterpolationMode.NEAREST,
            antialias=False
        )(needle_mask_tensor)
        prostate_mask_tensor = T.Resize(
            (self.mask_size, self.mask_size),
            interpolation=InterpolationMode.NEAREST,
            antialias=False
        )(prostate_mask_tensor)

        out["bmode"] = bmode_tensor
        out["needle_mask"] = needle_mask_tensor
        out["prostate_mask"] = prostate_mask_tensor
        out["needle_patches"] = needle_patches

        # Handle labels and clinical features
        if "grade_group" in item:
            label = item["grade_group"]
            if torch.isnan(torch.tensor(label)):
                label = 0
            out["label"] = torch.tensor(label).long()
        if "primus" in item:
            out["primus_label"] = torch.tensor(item["primus"] - 1).long() 

        if "psa" in item:
            psa = item["psa"] if not np.isnan(item["psa"]) else psa_avg
            out["psa"] = torch.tensor([(psa - psa_min) / (psa_max - psa_min)]).float()
        if "age" in item:
            age = item["age"] if not np.isnan(item["age"]) else age_avg
            out["age"] = torch.tensor([(age - age_min) / (age_max - age_min)]).float()
        if "approx_psa_density" in item:
            approx_psa_density = item["approx_psa_density"] if not np.isnan(item["approx_psa_density"]) else approx_psa_density_avg
            out["approx_psa_density"] = torch.tensor([(approx_psa_density - approx_psa_density_min) / (approx_psa_density_max - approx_psa_density_min)]).float()

        return out


class _ProstateDatasetAdapterOptimum:
    def __init__(self, image_orientation="probe_top"):
        self.image_orientation = image_orientation

    def __call__(self, item):
        """
        Adapter function to convert the Optimum dataset items to the format expected by prostnfound.
        """

        bmode = item["image"]
        bmode = np.array(bmode)[..., 0]
        needle_mask = np.array(item["needle_mask"])
        prostate_mask = np.ones_like(needle_mask)

        if self.image_orientation == "probe_top":
            bmode = np.flipud(bmode)  # Flip the image vertically
            needle_mask = np.flipud(needle_mask)  # Flip the mask vertically
            prostate_mask = np.flipud(prostate_mask)  # Flip the mask vertically

        info = item["info"]
        primus_val = info.get("PRI-MUS", None)
        try:
            primus_val = int(primus_val)
        except (ValueError, TypeError):

            primus_val = 1

        info["PRI-MUS"] = primus_val


        return {
            "bmode": bmode,
            "needle_mask": needle_mask,
            "prostate_mask": prostate_mask,
            "grade": info.get("Diagnosis", "Unknown"),
            "pct_cancer": info.get("% Cancer", 0.0),
            "psa": info.get("psa", 0.0),
            "age": info.get("age", 0.0),
            "approx_psa_density": info.get("approx_psa_density", 0.0),
            "family_history": info.get("family_history", np.nan),
            "center": info.get("center", "Unknown"),
            "all_cores_benign": info.get("all_cores_benign", False),
            "core_id": info.get("cine_id", "Unknown"),
            "patient_id": info["case"],
            "loc": info["Sample ID"],
            "grade_group": info["GG"],
            "clinically_significant": info.get("clinically_significant", False),
            "primus": int(info["PRI-MUS"]),
        }
    



