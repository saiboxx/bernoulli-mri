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

We use Python 3.9.

## Datasets

Our repository uses [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html),
[BraTS](https://www.med.upenn.edu/sbia/brats2018/data.html) and [fastMRI Knee](https://fastmri.med.nyu.edu).
Additionally, please download the [knee annotations](https://github.com/microsoft/fastmri-plus/blob/main/Annotations/knee.csv)
from fastMRI+ into the fastMRI Knee directory.
Per default, the data is supposed to be located at `data`.

The pre-processing of the datasets will be triggered on the first call
of the respective dataset objects: `ACDCDataset`, `BrainDataset`, `KneeDataset` in
`src/datasets.py`.


## Experiments

The sections below contain the experiments, which were conducted in the paper.
A single run for a dataset only takes minutes but be aware that in this reproducibility
setting, 10 runs with random initializations for each method are triggered.

Generally, optimization runs are in `scripts`. Each file contains a dictionary,
for configuring the optimization procedure. Evaluation is done in `notebooks`.

## Iterative Gradient Sampling (IGS)

The results for the IGS method can by created by calling `scripts/05_run_igs.py`
like:

```shell
for i in {1..10}; do python scripts/05_run_igs.py --idx $i; done
```

## Reconstruction Experiments with fastMRI Knee

The reconstruction experiments are executed by calling `scripts/04_run_optim_knee.py`:

```shell
python scripts/04_run_optim_knee.py
```
Afterwards, the notebook `notebooks/01_reconstruction_knee.ipynb` needs to be executed
to obtain the metrics. Our experiments resulted in following metrics:

### Acceleration Factor x4
|            | **PSNR** | **SSIM** | **NMSE** | 
|------------|----------|----------|----------|
| **Equi.**  | 24.481   | 0.601    | 0.066    | 
| **Gauss.** | 29.570   | 0.664    | 0.027    | 
| **IGS**    | 28.553   | 0.640    | 0.031    | 
| **ProM**   | 29.787   | 0.672    | 0.026    |

### Acceleration Factor x8
|            | **PSNR** | **SSIM** | **NMSE** | 
|------------|----------|----------|----------|
| **Equi.**  | 23.803   | 0.524    | 0.077    | 
| **Gauss.** | 28.118   | 0.560    | 0.035    | 
| **IGS**    | 26.428   | 0.532    | 0.047    | 
| **ProM**   | 28.472   | 0.570    | 0.034    |

### Acceleration Factor x16
|            | **PSNR** | **SSIM** | **NMSE** | 
|------------|----------|----------|----------|
| **Equi.**  | 23.365   | 0.474    | 0.085    | 
| **Gauss.** | 23.342   | 0.440    | 0.112    | 
| **IGS**    | 24.376   | 0.458    | 0.070    | 
| **ProM**   | 27.575   | 0.511    | 0.040    |

### Acceleration Factor x32
|            | **PSNR** | **SSIM** | **NMSE** | 
|------------|----------|----------|----------|
| **Equi.**  | 23.158   | 0.448    | 0.090    | 
| **Gauss.** | 17.373   | 0.299    | 0.396    | 
| **IGS**    | 22.313   | 0.409    | 0.108    | 
| **ProM**   | 26.739   | 0.473    | 0.046    |


## Segmentation Experiments with ACDC

The pre-trained segmentation model is contained in `models/acdc_unet.pt`.
It can be retrained by adapting `01_train_unet.py` with the `acdc` dataset flag.

To start computation for ProM with a 2D mask, call:

```shell
python scripts/04_run_optim_acdc.py -m f
```

For the 1D ProM call:

```shell
python scripts/04_run_optim_acdc.py -m h
```

After compute has finished, the below metrics are obtained by running 
`notebooks/02_segmentation_acdc.ipynb`

| **Acc. Fac.** | **Metric** | **Equi.** | **Gauss.** | **IGS** | **ProM (1D)** | **ProM (2D)** |
|---------------|------------|-----------|------------|---------|---------------|---------------|
| x8            | Dice       | 0.671     | 0.847      | 0.828   | 0.762         | 0.839         |
| x8            | IoU        | 0.546     | 0.752      | 0.726   | 0.650         | 0.742         |
| x16           | Dice       | 0.645     | 0.745      | 0.745   | 0.717         | 0.789         |
| x16           | IoU        | 0.517     | 0.534      | 0.627   | 0.599         | 0.679         |
| x32           | Dice       | 0.644     | 0.399      | 0.592   | 0.587         | 0.727         |
| x32           | IoU        | 0.517     | 0.301      | 0.466   | 0.460         | 0.606         |


## Segmentation Experiments with BraTS

The pre-trained segmentation model is contained in `models/brain_unet.pt`.
It can be retrained by adapting `01_train_unet.py` with the `brain` dataset flag.

To start computation for ProM with a 2D mask, call:

```shell
python scripts/04_run_optim_brats.py -m f
```

For the 1D ProM call:

```shell
python scripts/04_run_optim_brats.py -m h
```

After compute has finished, the below metrics are obtained by running 
`notebooks/03_segmentation_brats.ipynb`

| **Acc. Fac.** | **Metric** | **Equi.** | **Gauss.** | **IGS** | **ProM (1D)** | **ProM (2D)** |
|---------------|------------|-----------|------------|---------|---------------|---------------|
| x8            | Dice       | 0.596     | 0.733      | 0.716   | 0.646         | 0.739         |
| x8            | IoU        | 0.489     | 0.638      | 0.619   | 0.542         | 0.646         |
| x16           | Dice       | 0.589     | 0.597      | 0.651   | 0.537         | 0.735         |
| x16           | IoU        | 0.481     | 0.494      | 0.546   | 0.426         | 0.640         |
| x32           | Dice       | 0.580     | 0.315      | 0.537   | 0.483         | 0.706         |
| x32           | IoU        | 0.472     | 0.226      | 0.428   | 0.374         | 0.608         |