from dataclasses import dataclass, fields
from typing import Dict, Optional, Union, Tuple

import torch


@dataclass
class Configuration:
    file_path: str
    slice_idx: int
    coil_idx: Optional[int]
    cropping: Optional[Tuple]
    steps: int
    learning_rate: float
    bern_samples: int
    mask_style: str
    dense_target: float
    dense_start: float
    dense_end: float
    device: Union[str, float, torch.device]
    log_dir: str
    log_imgs: int


def get_configuration(arg_dict: Dict) -> Configuration:
    field_set = {f.name for f in fields(Configuration) if f.init}
    filtered_arg_dict = {k: v for k, v in arg_dict.items() if k in field_set}
    return Configuration(**filtered_arg_dict)
