# DuoVAE

(brief introduction)

## Supported devices
- CPU, CUDA, MPS (arm64 Apple silicon)

## Install

    conda env create -f environment.yml

## Run

### Train

    python train.py duovae 2d

Command format is `python train.py <model-name> <dataset-type>`.
- `<model-name>`: `duovae`: DuoVAE (ours), `pcvae`: [PCVAE](https://github.com/xguo7/PCVAE) (for comparisons)
- `<dataset-type>` `2d`: [dSprites](https://github.com/deepmind/dsprites-dataset), `3d`: [3dshapes](https://github.com/deepmind/3d-shapes)

Parameters can be configured in `parameters_default.json`.

### Test

(coming)

### Result

(coming)