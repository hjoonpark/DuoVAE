import os
import subprocess
import torch
import numpy as np
import h5py
import urllib.request


class Dataset3d(torch.utils.data.Dataset):
    """
    https://github.com/deepmind/3d-shapes
    """
    def __init__(self, logger, load_dir="./datasets/data", label_indices=[3, 1, 0]):
        """
        labels:
            0 floor hue: 10 values linearly spaced in [0, 1]
            1 wall hue: 10 values linearly spaced in [0, 1]
            2 object hue: 10 values linearly spaced in [0, 1]
            3 scale: 8 values linearly spaced in [0, 1]
            4 shape: 4 values in [0, 1, 2, 3]
            5 orientation: 15 values linearly spaced in [-30, 30]
        """
        os.makedirs(load_dir, exist_ok=True)
        self.logger = logger
        imgs, labels, labels_unnormalized = self.load_data(load_dir, label_indices)

        self.imgs = imgs
        self.labels = labels
        self.labels_unnormalized = labels_unnormalized
        self.img_channel = 3

        self.y_mins = torch.min(labels, dim=0).values
        self.y_maxs = torch.max(labels, dim=0).values
        # sanity check
        assert imgs.shape[0] == labels.shape[0]

    def __getitem__(self, index):
        output = {'x': self.imgs[index], 'y': self.labels[index], 'y_unnormalized': self.labels_unnormalized[index]}
        return output

    def __len__(self):
        return len(self.imgs)

    def load_data(self, load_dir, label_indices):
        url = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
        file = "3dshapes.h5"

        # download if not exists
        load_path = os.path.join(load_dir, file)
        self.logger.print("loading dataset from: {}".format(load_path))
        if not os.path.exists(load_path):
            self.logger.print("Not exists! Downloading dataset: {}".format(url))
            subprocess.check_call(["curl", "-L", url, "--output", load_path])

        dataset = h5py.File(load_path, "r")
        imgs = torch.Tensor((np.asarray(dataset["images"])/255.0).astype(np.float32))
        labels = torch.Tensor(np.asarray(dataset["labels"]))
        self.logger.print("Dataset loaded: images={}, labels={}".format(imgs.shape, labels.shape))

        label_shape = labels.shape[1:]  # [6]
        self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        self._NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}


        # normalize labels
        labels_unnormalized = labels.clone()
        mins, _ = torch.min(labels, dim=0, keepdim=True)
        maxs, _ = torch.max(labels, dim=0, keepdim=True)
        labels = (labels-mins) / (maxs-mins)

        # slice
        labels = labels[:, label_indices]

        subset = True
        if subset:
            n_samples = 100
            indices = torch.randperm(len(imgs))[:n_samples]
            imgs = imgs[indices]
            labels = labels[indices]
            labels_unnormalized = labels_unnormalized[indices]
        imgs = imgs.permute(0, 3, 1, 2)
        self.logger.print("[3dshapes stats]")
        self.logger.print("  - images: {} | min/max=({:.2f}, {:.2f})".format(imgs.shape, imgs.min(), imgs.max()))
        self.logger.print("  - labels: {}".format(labels.shape))
        for y_idx in range(labels.shape[-1]):
            self.logger.print("      labels[:, {}] min/max=({:.2f}, {:.2f}), min={:.2f}, std={:.2f}".format(y_idx, labels[:, y_idx].min(), labels[:, y_idx].max(), labels[:, y_idx].mean(), labels[:, y_idx].std()))
        return imgs, labels, labels_unnormalized