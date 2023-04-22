import os, sys
import json
import argparse
import shutil

from models.duovae import DuoVAE
from models.pcvae import PcVAE

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
    parser.add_argument("--model-num", type=int, help="", default=None)
    parser.add_argument("--save-dir", type=str, help="", default=None)
    args = parser.parse_args()
    model_num = args.model_num

    save_dir = args.save_dir
    args.model_dir = os.path.join(save_dir, "..", f"output_{model_num}")

    os.makedirs(save_dir, exist_ok=1)
    logger = Logger(save_path=os.path.join(save_dir, "log_{:02d}.txt".format(model_num)), muted=False)

    # load parameters
    param_path = glob.glob(os.path.join(args.model_dir, "log", "parameters*.json"))[0]
    params = load_parameters(param_path=param_path)

    # load dataset
    dataset = VaeBenchmarkDataset(dataset_name="2d", subset=True, logger=logger)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1)
    params["common"]["img_channel"] = dataset.img_channel
    params["common"]["y_dim"] = dataset.labels.shape[-1]
    print(params)
    
    # init model
    model = DuoVAE(params=params, is_train=False, logger=logger)
    load_model(model, os.path.join(args.model_dir, "model"), logger)
    model.eval()

    img_save_dir = os.path.join(save_dir)
    os.makedirs(img_save_dir, exist_ok=1)
    for batch_idx, data in enumerate(dataloader, 0):
        model.set_input(data)
        # save y traverse
        traversed_y, traversed_y_list = traverse_y("duovae", model, x=model.x, y=model.y, y_mins=dataset.y_mins, y_maxs=dataset.y_maxs, n_samples=7, logger=logger)
        save_path = os.path.join(img_save_dir, "model{:02d}_data{:02d}.png".format(model_num, batch_idx))
        save_image(traversed_y.squeeze(), save_path)
    
        save_path = os.path.join(img_save_dir, "model{:02d}_data{:02d}_gt.png".format(model_num, batch_idx))
        save_image(data["x"].squeeze().numpy(), save_path)
        # for r in range(len(traversed_y_list)):
        #     for c in range(len(traversed_y_list[r])):
        #         save_path = os.path.join(img_save_dir, "model{:01d}_{:01d}_{:01d}.png".format(model_num, r, c))
        #         I = traversed_y_list[r][c].squeeze()
        #         I = (I*255).astype(np.uint8)
        #         I = Image.fromarray(I)
        #         I.save(save_path)
        #         print(save_path)
        break
    print("### DONE ###")