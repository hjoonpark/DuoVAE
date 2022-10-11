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
import torch
import numpy as np

MODELS_SUPPORTED = set(["duovae", "pcvae"])
DATASETS_SUPPORTED = set(["2d", "3d"])

def load_parameters(model_name, save_dir):
    # load parameters from .json file
    params = json.load(open("parameters.json", "r"))

    # keep a record of the parameters for future reference
    save_path = os.path.join(save_dir, "parameters.json")
    json.dump(params, open(save_path, "w+"), indent=4)
    return params

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # for reproducibility
    set_all_seeds(0)

    # parse input arguments: two inputs are required <model_name> and <dataset_type>
    description = "PyTorch implementation of DuoVAE"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("model", type=str, help="Model: duovae, pcvae", default="duovae")
    parser.add_argument("dataset", type=str, help="Dataset: 2d, 3d", default="2d")
    args = parser.parse_args()

    # input args needed
    model_name = args.model
    dataset_name = args.dataset

    # safe-guard parsed input arguments
    assert model_name in MODELS_SUPPORTED, "[ERROR] model_name={} is not supported! Chooose from {}".format(model_name, MODELS_SUPPORTED)
    assert dataset_name in DATASETS_SUPPORTED, "[ERROR] dataset_name={} is not supported! Chooose from {}".format(dataset_name, DATASETS_SUPPORTED)

    # make output directories
    dirs = make_directories(root_dir=os.path.join("output", model_name, dataset_name), sub_dirs=["log", "model", "visualization"])

    # init helper class
    logger = Logger(muted=False)
    logger.print("=============== START ===============")
    logger.print("  model  : {}".format(model_name))
    logger.print("  dataset: {}".format(dataset_name))
    logger.print("=====================================")

    # load user-defined parameters
    params = load_parameters(model_name=model_name, save_dir=os.path.join(dirs["log"]))

    # load dataset
    dataset = VaeBenchmarkDataset(dataset_name=dataset_name, logger=logger)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=params["train"]["batch_size"])
    params["common"]["img_channel"] = dataset.img_channel
    params["common"]["y_dim"] = dataset.labels.shape[-1]

    # init models
    if model_name == "duovae":
        model = DuoVAE(params=params, is_train=True, logger=logger)
    elif model_name == "pcvae":
        model = PcVAE(params=params, is_train=True, logger=logger)
    else:
        raise NotImplementedError("Only duovae and pcvae models are supported.")

    # log model information
    logger.print(model)
    logger.print(params)
    logger.print("Device={}, GPU Ids={}".format(model.device, model.gpu_ids))
    logger.print("Training on {:,} number of data".format(len(dataset)))

    # make a copy of the model to future reference
    shutil.copyfile(os.path.join("models", "{}.py".format(model_name)), os.path.join(dirs["model"], "{}.py".format(model_name)))

    """
    # To continue training from a saved checkpoint, set load_dir to a directory containing *.pt files   
    # example: load_dir = "output/duovae/2d/model/"
    """
    load_dir = "output/duovae/2d/model/"
    # load_dir = None
    if load_dir is not None:
        load_model(model, load_dir, logger)
    model.train()

    # train
    losses_all = {}
    for epoch in range(1, params["train"]["n_epoch"]+1):
        losses_curr_epoch = {}
        batch_idx = 0
        for batch_idx, data in enumerate(dataloader, 0):
            # ===================================== #
            # main train step
            # ===================================== #
            # set input data
            model.set_input(data)

            # training happens here
            model.optimize_parameters()

            # ===================================== #
            # below are all for plots
            # ===================================== #
            # keep track of loss values
            losses = get_losses(model)
            for loss_name, loss_val in losses.items():
                if loss_name not in losses_curr_epoch:
                    losses_curr_epoch[loss_name] = 0
                losses_curr_epoch[loss_name] += loss_val.detach().cpu().item()

            # save reconstruct results
            if epoch % params["train"]["save_freq"] == 0 and batch_idx == 0:
                save_path = save_reconstructions(save_dir=dirs["log"], model=model, epoch=epoch)
                logger.print("train recontructions saved: {}".format(save_path))
        
        # keep track of loss values every epoch
        for loss_name, loss_val in losses_curr_epoch.items():
            if loss_name not in losses_all:
                losses_all[loss_name] = []
            losses_all[loss_name].append(loss_val)

        # log every certain epochs
        do_initial_checks = ((epoch > 0 and epoch <= 50) and (epoch % 10 == 0))
        if do_initial_checks or (epoch % params["train"]["log_freq"] == 0):
            loss_str = "epoch:{}/{} ".format(epoch, params["train"]["n_epoch"])
            for loss_name, loss_vals in losses_all.items():
                loss_str += "{}:{:.4f} ".format(loss_name, loss_vals[-1])
            logger.print(loss_str)
            
        # checkpoint every certain epochs
        if do_initial_checks or (epoch > 0 and epoch % params["train"]["save_freq"] == 0):
            model.eval()
            with torch.no_grad():
                # save loss plot
                json_path, save_path = save_losses(save_dir=dirs["log"], epoch=epoch, losses=losses_all)
                logger.print("train losses saved: {}, {}".format(json_path, save_path))

                # save model
                save_dir = save_model(save_dir=dirs["model"], model=model)
                logger.print("model saved: {}".format(dirs["model"]))

                # save y traverse
                traversed_y = traverse_y(model_name, model, x=model.x, y=model.y, y_mins=dataset.y_mins, y_maxs=dataset.y_maxs, n_samples=20)
                save_path = save_image(traversed_y.squeeze(), os.path.join(dirs["visualization"], "y_trav_{:05d}.png".format(epoch)))
                logger.print("y-traverse saved: {}".format(save_path))

                # save normalized mutual information as heatmap
                MI_score = save_mutual_information(dataloader, model)
                save_path = save_MI_score(save_dir=dirs["visualization"], MI=MI_score, model_name=model_name, epoch=epoch)
                logger.print("MI score saved: {}".format(save_path))
            model.train()
    logger.print("=============== DONE ================")