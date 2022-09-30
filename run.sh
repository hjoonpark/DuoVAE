echo "Train DuoVAE and PCVAE on both dSprites and 3dshapes datasets."
python train.py duovae 2d
python train.py duovae 3d
python train.py pcvae 2d
python train.py pcvae 3d