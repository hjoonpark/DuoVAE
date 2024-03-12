clear;

# download dSprites
url="https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
file="dsprites.npz"

save_dir="./datasets/data"
mkdir ${save_dir}
save_path="${save_dir}/${file}"
curl -L ${url} --output ${save_path}

# download 3d shapes
url="https://storage.googleapis.com/3d-shapes/3dshapes.h5"
file="3dshapes.h5"

save_path="${save_dir}/${file}"
curl -L ${url} --output ${save_path}