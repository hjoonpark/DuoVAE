import os, sys
import json
import argparse
import shutil

from models.duovae import DuoVAE

from datasets.vae_benchmark_dataset import VaeBenchmarkDataset
from utils.logger import Logger, LogLevel
from utils.util_io import make_directories
from utils.util_visualize import save_image, save_reconstructions, save_losses, save_MI_score
from utils.util_model import load_model, save_model, save_mutual_information, get_losses, traverse_y
import torch, glob
import numpy as np
from PIL import Image

def load_parameters(param_path):
    # load parameters from .json file
    params = json.load(open(param_path, "r"))
    return params

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_all_seeds(0)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model-dir", type=str, help="", default=None)
    parser.add_argument("--param-path", type=str, help="", default=None)
    parser.add_argument("--dataset", type=str, help="", default=None)
    args = parser.parse_args()
    model_dir = args.model_dir
    dataset = args.dataset

    save_dir = os.path.join(model_dir, "..", "..", "test")
    os.makedirs(save_dir, exist_ok=1)

    logger = Logger(save_path=os.path.join(save_dir, "log.txt"), muted=False)

    # load parameters
    params = load_parameters(param_path=args.param_path)

    # load dataset
    dataset = VaeBenchmarkDataset(dataset_name=dataset, subset=True, logger=logger)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1)
    params["common"]["img_channel"] = dataset.img_channel
    params["common"]["y_dim"] = dataset.labels.shape[-1]
    print(params)
    
    # init model
    model = DuoVAE(params=params, is_train=False, logger=logger)
    load_model(model, model_dir, logger)
    model.eval()

    model_name = "duovae"
    for batch_idx, data in enumerate(dataloader):
        model.set_input(data)

        # save y traverse
        traversed_y, _ = traverse_y(model_name, model, x=model.x, y=model.y, y_mins=dataset.y_mins, y_maxs=dataset.y_maxs, n_samples=7)
        save_path = os.path.join(save_dir, "{:03d}.png".format(batch_idx))
        save_image(traversed_y.squeeze(), save_path)
        logger.print("y-traverse saved: {}".format(save_path))

        # save input image
        save_path = os.path.join(save_dir, "{:03d}_gt.png".format(batch_idx))
        x = np.transpose(data["x"].squeeze().numpy(), (1,2,0))
        save_image(x, save_path)
    
    # save normalized mutual information as heatmap
    MI_score = save_mutual_information(dataloader, model)
    save_path = save_MI_score(save_dir, MI=MI_score, model_name=model_name)
    logger.print("MI score saved: {}".format(save_path))
    print("### DONE ###")