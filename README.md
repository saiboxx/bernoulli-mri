# Constrained Probabilistic Mask Learning for Task-specific Undersampled MRI Reconstruction

Code for the paper [Constrained Probabilistic Mask Learning for Task-specific Undersampled MRI Reconstruction](tbd)
at the *MICCAI 2023*.

> Undersampling is a common method in Magnetic Resonance Imaging (MRI) to subsample the 
> number of data points in k-space and thereby reduce acquisition times at the cost 
> of decreased image quality. In this work, we directly learn the undersampling masks 
> to derive task- and anatomic-specific patterns. To solve this discrete optimization
> challenge, we propose a general optimization routine called **ProM**: 
> A fully probabilistic, differentiable, versatile, and model-free framework for 
> mask optimization that enforces acceleration factors through a convex constraint.
> Our framework opens up various directions of interesting research in data-driven 
> undersampling mask learning. Analyzing knee, brain, and cardiac MRI datasets with
> our method, we discover that different modalities reveal different optimal
> undersampling masks. Furthermore, **ProM** can create undersampling masks that 
> maximize performance in downstream tasks like segmentation with networks trained 
> on fully sampled MRIs. Even with extreme acceleration factors, **ProM** yields 
> reasonable performance while being more versatile than existing methods, 
> paving the way for data-driven all-purpose mask generation.


<p align="center">
<img src=assets/prom_progress.png />
</p>

## Requirements:

Required packages are listed in the `requirements.txt` file, which can be install
e.g. over `pip`:

```shell
pip install -r requirements.txt
```

## Datasets

Our repository uses [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html),
[BraTS](https://www.med.upenn.edu/sbia/brats2018/data.html) and [fastMRI Knee](https://fastmri.med.nyu.edu).
Additionally, please download the [knee annotations](https://github.com/microsoft/fastmri-plus/blob/main/Annotations/knee.csv)
from fastMRI+ into the fastMRI Knee directory.
Per default, the data is supposed to be located at `data`.
