import csv
import glob
import json
import os
import random
from typing import Dict, Optional

import nibabel
from monai.transforms import ScaleIntensity
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, CenterCrop, Compose
from tqdm import tqdm

from src.utils import fft2c, ifft2c, load_h5_slice, min_max_normalize


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
            rng = random.Random(42)
            rng.shuffle(frames)
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
        self.preproc_dir = os.path.join(root, 'preprocessed')

        self.transforms = CenterCrop(320)

        if not os.path.exists(self.preproc_dir):
            self._preprocess()

        self.file_dir = os.path.join(
            self.root, 'preprocessed', 'train' if train else 'test'
        )

        self.files = os.listdir(self.file_dir)

    def _preprocess(self):
        print('Preprocessing dataset...')
        annotations_file = 'annotations.csv'
        annot_df = pd.read_csv(os.path.join(self.root, annotations_file))

        self._preprocess_dir(annot_df.copy(), train=True)
        self._preprocess_dir(annot_df.copy(), train=False)

    def _preprocess_dir(self, annot_df: pd.DataFrame, train: bool) -> None:
        data_dir = 'singlecoil_train' if train else 'singlecoil_val'
        img_dir = os.path.join(self.root, data_dir)

        target_dir = os.path.join(self.preproc_dir, 'train' if train else 'test')
        os.makedirs(target_dir, exist_ok=True)

        # Filter for existing files
        files = [f.replace('.h5', '') for f in os.listdir(img_dir)]
        annot_df = annot_df[annot_df['file'].isin(files)]
        annot_df = annot_df[annot_df['label'] != 'artifact']
        annot_df['x'] = annot_df['x'].astype(int)
        annot_df['y'] = annot_df['y'].astype(int)
        annot_df['width'] = annot_df['width'].astype(int)
        annot_df['height'] = annot_df['height'].astype(int)

        for file, file_group in tqdm(annot_df.groupby('file')):
            for sl, slice_group in file_group.groupby('slice'):

                annotations = []
                for _, entry in slice_group.iterrows():
                    a = (entry['x'], entry['y'], entry['width'], entry['height'])
                    annotations.append(a)

                k_space = load_h5_slice(
                    fp=os.path.join(img_dir, str(file) + '.h5'),
                    slice_idx=int(sl)
                )
                img = ifft2c(k_space)

                # Synthesize "segmentation" from annotations bounding boxes
                seg = torch.zeros_like(img, dtype=torch.float)
                counter = 1.
                for x, y, width, height in annotations:
                    seg[:, y:y + height, x:x + width] = counter
                    counter += 1.

                img = self.transforms(img)
                img = torch.abs(img)
                seg = self.transforms(seg)
                k_space = fft2c(img)

                f_name = str(file) + '_sl_' + str(sl).zfill(3) + '.npz'
                f_path = os.path.join(target_dir, f_name)
                np.savez_compressed(
                    f_path,
                    img=img.numpy(),
                    seg=seg.numpy(),
                    k_space=k_space.numpy(),
                    annot=np.asarray(annotations)
                )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        f_path = os.path.join(self.file_dir, self.files[idx])
        arr = np.load(f_path)

        img = torch.tensor(arr['img'])
        seg = torch.tensor(arr['seg'])
        k_space = torch.tensor(arr['k_space'])
        annot = torch.tensor(arr['annot'])

        return {
            'img': img,
            'seg': seg,
            'k_space': k_space,
            #'annotations': annot
        }


class BrainDataset(Dataset):
    def __init__(self, root: str, train: bool = True) -> None:
        self.root = root

        meta_file = os.path.join(self.root, 'dataset.json')
        with open(meta_file, 'r') as f:
            self.meta_data = json.load(f)

        if not os.path.exists(os.path.join(self.root, 'preprocessed')):
            self._preprocess()

        self.file_dir = os.path.join(
            self.root, 'preprocessed', 'train' if train else 'test'
        )

        self.files = os.listdir(self.file_dir)
        self.transforms = Compose([Resize(256), CenterCrop(256)])

    def _preprocess(self):
        print('Preprocessing dataset...')
        preprocess_dir = os.path.join(self.root, 'preprocessed')
        os.makedirs(preprocess_dir, exist_ok=True)

        # As testing labels are not directly available, we decide to statically split
        # 20% of subjects for validation.
        file_collection = self.meta_data['training'].copy()
        rng = random.Random(42)
        rng.shuffle(file_collection)
        cut_idx = int(len(file_collection) * 0.8)

        train_subjects = file_collection[:cut_idx]
        train_dir = os.path.join(preprocess_dir, 'train')
        self._exec_subject_preprocess(train_subjects, train_dir)
        test_subjects = file_collection[cut_idx:]
        test = os.path.join(preprocess_dir, 'test')
        self._exec_subject_preprocess(test_subjects, test)

    def _exec_subject_preprocess(self, subjects, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        slice_range = (60, 110)
        for entry in tqdm(subjects, desc='Subjects'):
            img_path = os.path.join(self.root, entry['image'][2:])
            seg_path = os.path.join(self.root, entry['label'][2:])

            img = nibabel.load(img_path).get_fdata()
            seg = nibabel.load(seg_path).get_fdata()

            for idx in range(slice_range[0], slice_range[1]):
                img_slice = img[:, :, idx, 2]
                seg_slice = seg[:, :, idx]

                f_name = entry['image'][17:20] + '_sl_' + str(idx).zfill(3) + '.npz'
                f_path = os.path.join(target_dir, f_name)
                np.savez_compressed(f_path, img=img_slice, seg=seg_slice)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        f_path = os.path.join(self.file_dir, self.files[idx])
        arr = np.load(f_path)

        img = torch.tensor(arr['img'])
        seg = torch.tensor(arr['seg'])

        img = min_max_normalize(img, min_quantile=0.001, max_quantile=0.999)
        img = img.unsqueeze(0).float()
        img = self.transforms(img)

        seg = seg.unsqueeze(0).float()
        seg = self.transforms(seg)

        k_space = fft2c(img)

        return {'img': img, 'seg': seg, 'k_space': k_space}
