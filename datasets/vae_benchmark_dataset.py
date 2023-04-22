import os
import subprocess
import torch
import numpy as np
import h5py

class VaeBenchmarkDataset(torch.utils.data.Dataset):
    """
    2d: dSprites
    https://github.com/deepmind/dsprites-dataset

    3d: 3dshapes
    https://github.com/deepmind/3d-shapes
    """
    def __init__(self, dataset_name, subset, logger, load_dir="./datasets/data"):
        os.makedirs(load_dir, exist_ok=True)
        self.logger = logger

        if dataset_name == "2d":
            imgs, labels, labels_unnormalized = self.load_dSprites(load_dir, subset=subset)
            self.img_channel = 1 # grayscale
        elif dataset_name == "3d":
            imgs, labels, labels_unnormalized = self.load_3dshapes(load_dir, subset=subset)
            self.img_channel = 3 # rgb
        else:
            raise NotImplementedError("Only 2d and 3d benchmark datasets are supported.")

        self.imgs = imgs
        self.labels = labels
        self.labels_unnormalized = labels_unnormalized

        # traversing y values to generate property-controlled reconstructions
        self.y_mins = torch.min(labels, dim=0).values
        self.y_maxs = torch.max(labels, dim=0).values

        # sanity check
        assert imgs.shape[0] == labels.shape[0]

    def __getitem__(self, index):
        output = {'x': self.imgs[index], 'y': self.labels[index], 'y_unnormalized': self.labels_unnormalized[index]}
        return output

    def __len__(self):
        return len(self.imgs)

    def load_dSprites(self, load_dir, label_indices=[2, 4, 5], subset=False):
        """
        labels:
            0 Color: white
            1 Shape: ellipse, square, heart
            2 Scale: 6 values linearly spaced in [0.5, 1]
            3 Orientation: 40 values in [0, 2 pi]
            4 Position X: 32 values in [0, 1]
            5 Position Y: 32 values in [0, 1]
        """
        url = "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
        file = "dsprites.npz"

        # download if not exists
        load_path = os.path.join(load_dir, file)
        self.logger.print("loading dataset from: {}".format(load_path))
        if not os.path.exists(load_path):
            self.logger.print("dSprites dataset does not exist! -> downloading: {}".format(url))
            subprocess.check_call(["curl", "-L", url, "--output", load_path])

        dataset_zip = np.load(load_path, allow_pickle=True, encoding='latin1')
        imgs = dataset_zip['imgs']
        labels_values = dataset_zip['latents_values']
        metadata = dataset_zip['metadata'][()]

        # Define number of values per labels and functions to convert to indices
        labels_sizes = metadata['latents_sizes']
        labels_bases = np.concatenate((labels_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))
        def latent_to_index(labels):
            return np.dot(labels, labels_bases).astype(int)

        # [N, 64, 64] -> [N, 1, 64, 64]
        imgs = torch.FloatTensor(imgs[:, None, :, :])
        labels_unnormalized = torch.FloatTensor(labels_values)
        labels = labels_unnormalized.clone()[:, label_indices]

        # normalize labels
        mins, _ = torch.min(labels, dim=0)
        maxs, _ = torch.max(labels, dim=0)
        labels = (labels-mins)/(maxs-mins)

        if subset:
            # subset for quick debugging
            # n_samples = 100
            n_samples = 6*32*32*5
            def sample_latent(size=1):
                samples = np.zeros((size, labels_sizes.size))
                for lat_i, lat_size in enumerate(labels_sizes):
                    samples[:, lat_i] = np.random.randint(lat_size, size=size)
                return samples
            
            # MODIFY HERE
            latents_sampled = sample_latent(size=n_samples)

            """
            fix some of the latents
            """
            # use only 1 shape: 0=square, 1=eplisoid, 2=heart
            latents_sampled[:, 1] = 1
            # fix orientation
            # latents_sampled[:, 3] = 0

            indices_sampled = latent_to_index(latents_sampled)
            imgs = torch.FloatTensor(imgs[indices_sampled][:, :, :]) # [N, 64, 64] -> [N, 1, 64, 64]
            labels = labels[indices_sampled]

        self.logger.print("[dSprites stats]")
        self.logger.print("  - images: {} | min/max=({:.2f}, {:.2f})".format(imgs.shape, imgs.min(), imgs.max()))
        self.logger.print("  - labels: {}".format(labels.shape))
        for y_idx in range(labels.shape[-1]):
            self.logger.print("      labels[:, {}] min/max=({:.2f}, {:.2f}), min={:.2f}, std={:.2f}".format(y_idx, labels[:, y_idx].min(), labels[:, y_idx].max(), labels[:, y_idx].mean(), labels[:, y_idx].std()))
        return imgs, labels, labels_unnormalized

    def load_3dshapes(self, load_dir, label_indices=[3, 1, 0], subset=False):
        """
        labels:
            0 floor hue: 10 values linearly spaced in [0, 1]
            1 wall hue: 10 values linearly spaced in [0, 1]
            2 object hue: 10 values linearly spaced in [0, 1]
            3 scale: 8 values linearly spaced in [0, 1]
            4 shape: 4 values in [0, 1, 2, 3]
            5 orientation: 15 values linearly spaced in [-30, 30]
        """
        url = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
        file = "3dshapes.h5"

        # download if not exists
        load_path = os.path.join(load_dir, file)
        self.logger.print("loading dataset from: {}".format(load_path))
        if not os.path.exists(load_path):
            self.logger.print("3dshapes dataset does not exist! -> downloading: {}".format(url))
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

        if subset:
            # subset for quick debugging
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