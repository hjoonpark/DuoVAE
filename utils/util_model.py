import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from utils.util_io import as_np
from utils.logger import Logger, LogLevel

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

def get_available_devices(logger):
    """
    loads whichever available GPUs/CPU
    """
    gpu_ids = []
    device = "cpu"
    if torch.cuda.is_available():
        logger.print("CUDA is available")
        torch.cuda.empty_cache()
        n_gpu = torch.cuda.device_count()
        
        gpu_ids = [i for i in range(n_gpu)]
        device = "cuda:0"
    else:
        """
        support for arm64 MacOS
        https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/
        """
        # this ensures that the current MacOS version is at least 12.3+
        logger.print("CUDA is not available")
        if torch.backends.mps.is_available():
            logger.print("Apple silicon (arm64) is available")
            # this ensures that the current PyTorch installation was built with MPS activated.
            device = "mps:0"
            gpu_ids.append(0) # (2022) There is no multi-MPS Apple device, yet
            if not torch.backends.mps.is_built():
                logger.print("Current PyTorch installation was not built with MPS activated. Using cpu instead.")
                device = "cpu"

    return device, gpu_ids

def reparameterize(mean, logvar, sample):
    if sample:
        std = torch.sqrt(torch.exp(logvar))
        P = dist.Normal(mean, std)
        z = P.rsample()
        return z, P
    else:
        return mean, None

def kl_divergence(Q, P):
    batch_size, z_dim = Q.loc.shape
    return dist.kl_divergence(Q, P).sum()

def save_model(save_root_dir, epoch, model):
    save_dir = os.path.join(save_root_dir, "{:05d}".format(epoch))
    os.makedirs(save_dir, exist_ok=True)

    for model_name in model.model_names:
        save_path = os.path.join(save_dir, "{}.pt".format(model_name))

        net = getattr(model, model_name)
        if len(model.gpu_ids) == 1:
            torch.save(net.cpu().state_dict(), save_path)
            net.to(model.device)
        elif len(model.gpu_ids) > 1:
            torch.save(net.module.cpu().state_dict(), save_path)
            net.to(model.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)
    
    # only keep the recently saved models
    folders_names = sorted(next(os.walk(save_root_dir))[1])
    if len(folders_names) > 3:
        f_to_delete = os.path.join(save_root_dir, folders_names[0])
        shutil.rmtree(f_to_delete)
    return save_dir

def load_model(model, load_dir, logger):
    for model_name in model.model_names:
        load_path = os.path.join(load_dir, "{}.pt".format(model_name))
        print("load_path:", load_path)
        state_dict = torch.load(load_path, map_location=str(model.device))
        net = getattr(model, model_name)    
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        try:
            logger.print("loading model: {}".format(load_path))
            net.load_state_dict(state_dict)
        except:
            logger.print("failed to load - architecture mismatch! Initializing new instead: {}".format(model_name), LogLevel.WARNING.name)
    print("[INFO] models loaded succesfully!")

def save_mutual_information(dataloader, model):
    Z = None
    W = None
    labels = None
    with torch.no_grad():
        for data in dataloader:
            x = data["x"].to(model.device)
            y = data["y"].to(model.device)

            (z, w), _ = model.encode(x, sample=False)

            Z = as_np(z) if Z is None else np.concatenate((Z, as_np(z)), axis=0)
            W = as_np(w) if W is None else np.concatenate((W, as_np(w)), axis=0)
            labels = as_np(y) if labels is None else np.concatenate((labels, as_np(y)), axis=0)

    latents = np.concatenate((Z, W), axis=1)
    MI_score = _compute_MI_score(latents, labels)

    z_dim = z.shape[-1]
    MI_score_z = MI_score[0:z_dim, :].mean(axis=0)
    MI_score = np.vstack((MI_score_z, MI_score[z_dim:]))
    return MI_score

def _compute_MI_score(latents, labels):
    latent_dim = latents.shape[1]
    label_dim = labels.shape[1]
    
    all_score = np.zeros((latent_dim, label_dim))

    for latent_idx in range(latent_dim):
        for label_idx in range(label_dim):
            score = _calc_MI(labels[:, label_idx], latents[:, latent_idx])
            all_score[latent_idx, label_idx] = score

    return all_score

def _calc_MI(X, Y, bins=10):
    c_XY = np.histogram2d(X,Y,bins)[0]
    c_X = np.histogram(X,bins)[0]
    c_Y = np.histogram(Y,bins)[0]

    H_X = _shan_entropy(c_X)
    H_Y = _shan_entropy(c_Y)
    H_XY = _shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    normalized_MI = 2*MI/(H_X + H_Y)
    return normalized_MI

def _shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H