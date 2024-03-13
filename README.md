# DuoVAE

PyTorch implementation of DuoVAE (a VAE-framework for property-controlled data generation) proposed in\
[**Collagen Fiber Centerline Tracking in Fibrotic Tissue via Deep Neural Networks with Variational Autoencoder-based Synthetic Training Data Generation**](https://www.sciencedirect.com/science/article/pii/S1361841523002219),\
Hyojoon Park*, Bin Li*, Yuming Liu, Michael S. Nelson, Helen M. Wilson, Eftychios Sifakis, Kevin W. Eliceiri, \
Medical Image Analysis 2023.

DuoVAE generates data with desired properties controlled by continuous property values.\
This repository is designed specifically for training and testing on the VAE benchmark datasets:\
[dSprites](https://github.com/deepmind/dsprites-dataset) and [3dshapes](https://github.com/deepmind/3d-shapes).

![duovae](/etc/figures/duovae_all_loop.gif)


## Related repositories
 - [Collagen fiber extraction and analysis in cancer tissue microenvironment](https://github.com/uw-loci/collagen-fiber-metrics)

 <div align="left">
  <img src="https://github.com/uw-loci/collagen-fiber-metrics/raw/master/thumbnails/output.png" width="800px" />
</div>

 - [Training of the collagen fiber centerline extract network (Stage I, II, and III)](https://github.com/hjoonpark/collagen-fiber-centerline-extraction)

![figure](/etc/figures/pipeline.png)

## Prerequisites
- Linux, MacOS, Windows
- CPU, CUDA, MPS (arm64 Apple silicon)

  (*MPS is supported for MacOS 12.3+, but may be unstable currently - September 2022*)

## Installation

For [conda](https://docs.anaconda.com/anaconda/install/) users,

    conda env create -f environment.yml

For [pip](https://pip.pypa.io/en/stable/installation/) users,

    pip install -r requirements.txt

Then, install [PyTorch](https://pytorch.org/get-started/locally/) with either CPU, CUDA, or MPS supports.

Other configuration methods can be found [here](/etc/doc/installation.md).\
List of tested versions can be found [here](/etc/doc/tested_versions.md).

## Prepare datasets

Running `train.py` (see `Train` section below) will automatically download the needed datasets.
To download them directly, run

    bash download_datasets.sh

  This will download [dSprites](https://github.com/deepmind/dsprites-dataset) and [3dshapes](https://github.com/deepmind/3d-shapes) to: `./datasets/data/`.

## Train

Command format is `python train.py <dataset-type>`, for example

    python train.py duovae 2d

`<dataset-type>` 
- `2d` to use [dSprites](https://github.com/deepmind/dsprites-dataset)
- `3d` to use [3dshapes](https://github.com/deepmind/3d-shapes)

Additional parameters can be configured in `parameters.json`.\
More command examples can be found [here](run.sh).

## Test

(coming)

## Results

### 1. Property-controlled image generations

**dSprites** dataset

![figure](/etc/figures/y_traverse_dsprites_duovae.png)

The controlled properties (from left to right in each row) are 
- $y_1$: scale of a shape $\rightarrow$ from small to large,
- $y_2$: $x$ position of a shape $\rightarrow$ from left to right,
- $y_3$: $y$ position of a shape $\rightarrow$ from top to bottom.

**3dshapes** dataset

![figure](/etc/figures/y_traverse_3dshapes_duovae.png)

The controlled properties (from left to right in each row) are 
- $y_1$: scale of a shape $\rightarrow$ from small to large,
- $y_2$: wall color $\rightarrow$ from red to violet,
- $y_3$: floor color $\rightarrow$ from red to violet.

### 2. Normalized mutual information (MI)

The [mutual information](https://en.wikipedia.org/wiki/Mutual_information) of two random variables quantifies the amount of information (in units such as shannons (bits) or nats) obtained about one random variable by observing the other random variable.
In the ideal case, the heatmaps of normalized MI between each of the properties and latent variables should be 1 in the diagonal values and 0 in the off-diagonal values as well as for $\mathbf{z}_{avg}$ (average MI of all latent variables $z_i\in\mathbf{z}$), indicating perfect correlations where each property $y_i$ is completely inferred by one supervised latent variable $w_i$.

Below are the heatmaps of the normalized MI on *dSprites* (left) and *3dshapes* (right) dataset.

![figure](/etc/figures/MI_double.png)

We can see that the diagonal values of the heatmap are high and close to 1 for both datasets, implying high correlations between each of the respective property values $y_i$ and latent variables $w_i$.

### 3. Latent variable traverse

It is possible to have a very high MI score but with very poor reconstructions. Therefore, we visualize the reconstructions generated when traversing the latent variables to validate whether the latent representation spaces are smooth and disentangled.

**dSprites** dataset with supervised latent variables $(w_1, w_2, w_3)$ for (scale, $x$ position, $y$ position), respectively, and unsupervised latent variables $(z_1, z_2, z_3, z_4)$.

![figure](/etc/figures/zw_traverse_dsprites_duovae.png)

**3dshapes** dataset with supervised latent variables $(w_1, w_2, w_3)$ for (scale, wall color, floor color), respectively, and unsupervised latent variables $(z_1, z_2, z_3, z_4)$.

![figure](/etc/figures/zw_traverse_3dshapes_duovae.png)

Each of the supervised latent variables $\mathbf{w}$ (top 3 rows) captures the information of each property $(y_1, y_2, y_3)$, respectively, whereas the rest of the latent variables $\mathbf{z}$ (bottom 4 rows) appear to have captured the rest of the information in an entangled way.

## Citation

    @article{park2023collagen,
             title={Collagen fiber centerline tracking in fibrotic tissue via deep neural networks with variational autoencoder-based synthetic training data generation},
             author={Park, Hyojoon and Li, Bin and Liu, Yuming and Nelson, Michael S and Wilson, Helen M and Sifakis, Eftychios and Eliceiri, Kevin W},
             journal={Medical Image Analysis},
             volume={90},
             pages={102961},
             year={2023},
             publisher={Elsevier}
    }
