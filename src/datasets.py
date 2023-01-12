import glob
import os
import random
from typing import Dict, Optional

from monai.transforms import ScaleIntensity
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, CenterCrop, Compose

from src.utils import fft2c

class ACDCDataset(Dataset):
    def __init__(
        self, root: str, train: bool = True, max_patients: Optional[int] = None
    ) -> None:
        self.root = root
        data_dir = 'training' if train else 'testing'
        self.pat_dir = os.path.join(root, 'database', data_dir)

        pattern = os.path.join(self.pat_dir, 'patient???', 'patient???_frame01.nii.gz')
        frames = sorted(glob.glob(pattern))

        if max_patients is not None:
            random.seed(42)
            random.shuffle(frames)
            frames = frames[:max_patients]

        self.file_index = []
        for frame in frames:
            arr = nib.load(frame).get_fdata()
            seg = nib.load(frame.replace('.nii.gz', '_gt.nii.gz')).get_fdata()

            assert arr.shape == seg.shape
            num_slices = arr.shape[-1]

            for i in range(2, num_slices - 2):
                self.file_index.append({
                    'img': arr[:, :, i],
                    'seg': seg[:, :, i],
                })

        self.img_transforms = Compose([
            ScaleIntensity(),
            Resize(256),
            CenterCrop(256)
        ])

        self.seg_transforms = Compose([
            Resize(256),
            CenterCrop(256)
        ])

    def __len__(self) -> int:
        return len(self.file_index)

    def __getitem__(self, idx: int) -> Dict:
        img = self.file_index[idx]['img']
        img = torch.tensor(img).unsqueeze(0).float()
        img = self.img_transforms(img)

        seg = self.file_index[idx]['seg']
        seg = torch.tensor(seg).unsqueeze(0).float()
        seg = self.seg_transforms(seg)

        k_space = fft2c(img)

        return {'img': img, 'seg': seg, 'k_space': k_space}
