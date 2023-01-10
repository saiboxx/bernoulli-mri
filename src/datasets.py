import glob
import os
from typing import Dict, Optional

import nibabel as nib
import torch
from torch.utils.data import Dataset


class ACDCDataset(Dataset):
    def __init__(
        self, root: str, train: bool = True, max_patients: Optional[int] = None
    ) -> None:
        self.root = root
        data_dir = 'training' if train else 'testing'
        self.pat_dir = os.path.join(root, 'database', data_dir)

        pattern = os.path.join(self.pat_dir, 'patient???', 'patient???_frame??.nii.gz')
        self.frames = glob.glob(pattern)
        self.frames_seg = [f.replace('.nii.gz', '_gt.nii.gz') for f in self.frames]
        self.pat_ids = [d for d in os.listdir(self.pat_dir) if 'patient' in d]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict:
        frame_path = self.frames[idx]
        seg_path = self.frames_seg[idx]

        frame = nib.load(frame_path)
        seg = nib.load(seg_path)

        return {'img': frame, 'seg': seg}
