## Installation

- Option 1
    - *(Recommended)* For [conda](https://docs.anaconda.com/anaconda/install/) users: `conda env create -f environment.yml`
    - For [pip](https://pip.pypa.io/en/stable/installation/) users: `pip install -r requirements.txt`

- Option 2

    Manually create and configure a python (recommended version 3.7+) virtual environment and install the following packages `matplotlib`, `seaborn`, `h5py`.
    An example using [conda](https://docs.anaconda.com/anaconda/install/):

        conda create --name duovae python=3.9 matplotlib seaborn h5py
        conda activate duovae
    
- Common

    In the configured environment, install [PyTorch](https://pytorch.org/get-started/locally/) (tested versions: 1.12.x, 1.
13.x) with either CPU, CUDA, or MPS supports.