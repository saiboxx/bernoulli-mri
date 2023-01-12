import csv
import glob
import os
import random
from typing import Dict, Optional

from monai.transforms import ScaleIntensity
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, CenterCrop, Compose

from src.utils import fft2c, ifft2c, load_h5_slice


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


class KneeDataset(Dataset):
    def __init__(self, root: str, train: bool = True) -> None:
        self.root = root
        data_dir = 'singlecoil_train' if train else 'singlecoil_val'
        self.img_dir = os.path.join(root, data_dir)

        annotations_file = 'annotations.csv'
        self.annot_df = pd.read_csv(os.path.join(root, annotations_file))

        # Filter for existing files
        files = [f.replace('.h5', '') for f in os.listdir(self.img_dir)]
        self.annot_df = self.annot_df[self.annot_df['file'].isin(files)]
        self.annot_df = self.annot_df[self.annot_df['label'] != 'artifact']
        self.annot_df['x'] = self.annot_df['x'].astype(int)
        self.annot_df['y'] = self.annot_df['y'].astype(int)
        self.annot_df['width'] = self.annot_df['width'].astype(int)
        self.annot_df['height'] = self.annot_df['height'].astype(int)

        self.file_index = []
        for file, file_group in self.annot_df.groupby('file'):
            for sl, slice_group in file_group.groupby('slice'):

                annotations = []
                for _, entry in slice_group.iterrows():
                    a = (entry['x'], entry['y'], entry['width'], entry['height'])
                    annotations.append(a)

                self.file_index.append({
                    'file': str(file) + '.h5',
                    'slice': int(sl),
                    'annotations': tuple(annotations)
                })

        self.transforms = CenterCrop(320)

    def __len__(self) -> int:
        return len(self.file_index)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.file_index[idx]

        k_space = load_h5_slice(
            fp=os.path.join(self.img_dir, entry['file']),
            slice_idx=entry['slice']
        )
        img = ifft2c(k_space)

        # Synthesize "segmentation" from annotations bounding boxes
        seg = torch.zeros_like(img, dtype=torch.float)
        counter = 1.
        for x, y, width, height in entry['annotations']:
            seg[:, y:y+height, x:x+width] = counter
            counter += 1.

        img = self.transforms(img)
        img = torch.abs(img)
        seg = self.transforms(seg)
        k_space = fft2c(img)

        return {
            'img': img,
            'seg': seg,
            'k_space': k_space,
            'annotations': entry['annotations']
        }
