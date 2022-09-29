# DuoVAE

(brief introduction)

**DuoVAE**: [Paper]() | [Project]()

(Image)

--- 
## Prerequisites
- Linux, MacOS, Windows
- CPU, CUDA, MPS (arm64 Apple silicon)

--- 
## Installation

1. Clone this repo: `git clone https://github.com/hjoonpark/DuoVAE.git`

2. Configure a python environment
    - Option 1 (Recommended) 
        
        Manually create and configure a python (recommended version v3.7+) virtual environment and install the following packages `torch, matplotlib, seaborn, h5py`.

        An example using [conda](https://docs.anaconda.com/anaconda/install/):

          conda create --name duovae python=3.9 matplotlib seaborn h5py
          conda activate duovae
          # Then, install pytorch

    - Option 2
        - For [conda](https://docs.anaconda.com/anaconda/install/) users, create an environment using `conda env create -f environment.yml`
        - For [pip](https://pip.pypa.io/en/stable/installation/) users, create an environment using `pip install -r requirements.txt`
        
    - Common
        - In the configured environment, install [PyTorch](https://pytorch.org/get-started/locally/) (tested versions: 1.12.x, 1.
        13.x) with either CPU, CUDA, or MPS supports.

--- 
## Run

### Train

    python train.py duovae 2d

Command format is `python train.py <model-name> <dataset-type>`.
- `<model-name>`: `duovae` to use DuoVAE (ours), `pcvae` to use [PCVAE](https://github.com/xguo7/PCVAE) (for comparisons)
- `<dataset-type>`: `2d` to use [dSprites](https://github.com/deepmind/dsprites-dataset), `3d` to use [3dshapes](https://github.com/deepmind/3d-shapes)

Additional parameters can be configured in `parameters.json`.

--- 
## Test

(coming)

--- 
### Evaluation results

#### 1. Property-controlled image generations

`dSprites` dataset
![figure](/figures/y_traverse_dsprites_duovae.png)
The controlled properties (from left to right in each row) are 
- $y_1$: scale of a shape $\rightarrow$ from small to large,
- $y_2$: $x$ position of a shape $\rightarrow$ from left to right,
- $y_3$: $y$ position of a shape $\rightarrow$ from top to bottom.

`3dshapes` dataset
![figure](/figures/y_traverse_3dshapes_duovae.png)
The controlled properties (from left to right in each row) are 
- $y_1$: scale of a shape $\rightarrow$ from small to large,
- $y_2$: wall color $\rightarrow$ from red to violet,
- $y_3$: floor color $\rightarrow$ from red to violet.

#### 2. Normalized mutual information (MI)
In the ideal case, the heatmaps of MI between each property and latent variable should be 1 in the diagonal values and 0 in the off-diagonal values as well as for $\mathbf{z}_{avg}$ (indicating perfect correlations where each property $y_i$ is completely inferred by one supervised latent variable $w_i$).

`dSprites` dataset

![figure](/figures/MI_duovae_2d.png)

`3dshapes` dataset

![figure](/figures/MI_duovae_3d.png)

#### 3. Latent variable traverse

It is possible to have very high MI scores but poor reconstructions. Therefore, we traverse the latent variables to test whether the latent variables are smooth and disentangled.

`dSprites` dataset with property $(y_1, y_2, y_3)$=(scale, $x$ position, $y$ position).

![figure](/figures/zw_traverse_dsprites_duovae.png)

`3dshapes` dataset with property $(y_1, y_2, y_3)$=(scale, wall color, floor color).

![figure](/figures/zw_traverse_3dshapes_duovae.png)

Each of the supervised latent variables $\mathbb{w}$ (top 3 rows) captures the information of each property ($y_1$, $y_2$, $y_3$), respectively, whereas the rest of the latent variables $\mathbb{z}$ (bottom 4 rows) appear to have captured the rest of the information in an entangled way.

---
## Citation

    coming
---

## Tested versions
    
### Ubuntu
20.04.5 LTS

    python=3.9.12
    numpy=1.21.5
    torch=1.12.1
    h5py=3.6.0
    matplotlib=3.5.1
    seaborn=0.11.2

### MacOS 
Monterey 12.5 with Apple M1 chip

    python=3.9.13
    numpy=1.22.3
    torch=1.13.0.dev20220928
    h5py=3.6.0
    matplotlib=3.5.2
    seaborn=0.11.2

### Windows

    coming