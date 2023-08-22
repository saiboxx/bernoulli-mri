# Constrained Probabilistic Mask Learning for Task-specific Undersampled MRI Reconstruction

Code for the paper [Constrained Probabilistic Mask Learning for Task-specific Undersampled MRI Reconstruction](https://arxiv.org/abs/2305.16376) @ WACV 2024.

> Undersampling is a common method in Magnetic Resonance Imaging (MRI) to subsample the number of data points in k-space, reducing acquisition times at the cost of decreased image quality.
> A popular approach is to employ undersampling patterns following various strategies, e.g., variable density sampling or radial trajectories.
> In this work, we propose a method that directly learns the undersampling masks from data points, thereby also providing task- and domain-specific patterns.
> To solve the resulting discrete optimization problem, we propose a general optimization routine called ProM: A fully probabilistic, differentiable, versatile, and model-free framework for mask optimization that enforces acceleration factors through a convex constraint.
> Analyzing knee, brain, and cardiac MRI datasets with our method, we discover that different anatomic regions reveal distinct optimal undersampling masks,
> demonstrating the benefits of using custom masks, tailored for a downstream task.
> Furthermore, ProM can create undersampling masks that maximize performance in downstream tasks like segmentation with networks trained on fully-sampled MRIs.
> Even with extreme acceleration factors, ProM yields reasonable performance while being more versatile than existing methods, paving the way for data-driven all-purpose mask generation.

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

## Use Own Data

In general, a target dataset needs to fulfill the requirement of implementing a subclass of `ProMDataset` in `src/datasets.py`.
This implies that the `__get_item__` method should return a dictionary containing tensors of the image in 
image space, k-space and a segmentation. Hereby, the segmentation maybe an empty dummy if the segmentation
downstream task is not applied.
Also an image size should be supplied.

The `run_dataset_optim`, which kicks off the training procedure, allows passing a custom dataset
directly like:

```python
run_dataset_optim(cfg=cfg, ds=MyCustomDataset())
```

## Train ProM

The subfolder `scripts` contains a few starter scripts on how to use apply ProM to a PyTorch dataset.
The first script `01_run_prom_reconstruction.py` shows how to configure and train ProM for
a classic reconstruct task. `02_train_unet.py` trains the U-nets used for the segmentation downstream tasks
in our paper. These are also available in the `models` directory.
Subsequently, `03_run_prom_segmentation.py` applies the trained networks in the ProM procedure.
Lastly, use `04_eval_mask.py` to obtain metrics.



## Citation

If you use our repository in your research, please cite our paper *Constrained Probabilistic Mask Learning for Task-specific Undersampled MRI Reconstruction*:

```
@inproceedings{weber2024constrained,
  title={Constrained Probabilistic Mask Learning for Task-specific Undersampled MRI Reconstruction},
  author={Weber, Tobias and Ingrisch, Michael and Bischl, Bernd and R{\"u}gamer, David},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year      = {2024},
}
```
