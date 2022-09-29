import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import itertools
import torch.distributions as dist

from utils.util_model import get_available_devices, View, reparameterize, kl_divergence
from utils.logger import LogLevel
from utils.util_io import as_np

class DuoVAE(nn.Module):
    def __init__(self, params, is_train, logger):
        super().__init__()

        self.device, self.gpu_ids = get_available_devices(logger)

        # parameters
        lr = params["train"]["lr"]
        z_dim = params["common"]["z_dim"]
        w_dim = params["common"]["w_dim"]
        hid_channel = params["common"]["hid_channel"]
        hid_dim_x = params["common"]["hid_dim_x"]
        hid_dim_y = params["common"]["hid_dim_y"]
        img_channel = params["common"]["img_channel"]
        y_dim = params["common"]["y_dim"]
        self.x_recon_weight = params["common"]["x_recon_weight"]
        self.y_recon_weight = params["common"]["y_recon_weight"]
        self.beta_z = params["common"]["beta_z"]
        self.beta_w = params["common"]["beta_w"]
        self.beta_w2 = params["duovae"]["beta_w2"]

        # define models
        self.encoder_x = torch.nn.DataParallel(EncoderX(img_channel, hid_channel, hid_dim_x, z_dim, w_dim)).to(self.device)
        self.decoder_x = torch.nn.DataParallel(DecoderX(img_channel, hid_channel, hid_dim_x, z_dim, w_dim)).to(self.device)
        self.encoder_y = torch.nn.DataParallel(EncoderY(y_dim, hid_dim_y, w_dim)).to(self.device)
        self.decoder_y = torch.nn.DataParallel(DecoderY(y_dim, hid_dim_y, w_dim)).to(self.device)
        self.model_names = ["encoder_x", "decoder_x", "encoder_y", "decoder_y"]

        if is_train:
            params_x = itertools.chain(self.encoder_x.parameters(), self.decoder_x.parameters(), self.decoder_y.parameters())
            params_y = itertools.chain(self.encoder_y.parameters(), self.decoder_y.parameters())

            self.optimizer_x = torch.optim.Adam(params_x, lr=lr)
            self.optimizer_y = torch.optim.Adam(params_y, lr=lr)

    def set_input(self, data):
        self.x = data['x'].to(self.device) # (B, img_channel, 64, 64)
        self.y = data['y'].to(self.device) # (B, y_dim)

    def encode_x(self, x, sample: bool):
        # encode
        (z_mean, w_mean), (z_logvar, w_logvar) = self.encoder_x(x)
        # sample w, z
        # print("Z:", z_mean.shape, z_mean.min().item(), z_mean.max().item(), z_logvar.shape, z_logvar.min().item(), z_logvar.max().item())
        # print("W:", w_mean.shape, w_mean.min().item(), w_mean.max().item(), w_logvar.shape, w_logvar.min().item(), w_logvar.max().item())
        # print()
        z, Qz = reparameterize(z_mean, z_logvar, sample)
        w, Qw = reparameterize(w_mean, w_logvar, sample)
        
        return (z, w), (Qz, Qw)

    def encode(self, x, sample: bool):
        # alias for encode_x
        return self.encode_x(x, sample)
        
    def decode_x(self, z, w):
        y_recon = self.decoder_y(w)
        x_logits, x_recon = self.decoder_x(z, w)
        return x_logits, x_recon, y_recon

    def backward_x(self):
        # encode
        (self.z, self.w), (Qz, Qw) = self.encode_x(self.x, sample=True)
        # decode
        x_logits, self.x_recon, self.y_recon = self.decode_x(self.z, self.w)

        # losses
        batch_size = self.x.shape[0]
        self.loss_x_recon = self.x_recon_weight*F.binary_cross_entropy_with_logits(x_logits, self.x, reduction="sum") / batch_size
        self.loss_y_recon = self.y_recon_weight*F.mse_loss(self.y_recon, self.y, reduction="sum") / batch_size

        Pz = dist.Normal(torch.zeros_like(self.z), torch.ones_like(self.z))
        self.loss_kl_div_z = self.beta_z*kl_divergence(Qz, Pz) / batch_size

        with torch.no_grad():
            # no backpropagation on the encoder q(y|w) in this step
            w_mean, w_logvar = self.encoder_y(self.y)
            w_std = torch.sqrt(torch.exp(w_logvar.detach()))
            Pw = dist.Normal(w_mean.detach(), w_std)
        self.loss_kl_div_w = self.beta_w*kl_divergence(Qw, Pw) / batch_size

        loss = self.loss_x_recon + self.loss_y_recon \
                + self.loss_kl_div_z + self.loss_kl_div_w

        loss.backward()

    def encode_y(self, y, sample: bool):
        w_mean, w_logvar = self.encoder_y(y)
        w2, Qw2 = reparameterize(w_mean, w_logvar, sample=sample)
        return w2, Qw2

    def decode_y(self, w2):
        return self.decoder_y(w2)

    def backward_y(self):
        # encode
        self.w2, Qw2 = self.encode_y(self.y, sample=True)
        # decode
        self.y_recon2 = self.decoder_y(self.w2)

        # losses
        batch_size = self.x.shape[0]
        self.loss_y_recon2 = self.y_recon_weight*F.mse_loss(self.y_recon2, self.y, reduction="sum") / batch_size

        Pw = dist.Normal(torch.zeros_like(self.w2), torch.ones_like(self.w2))
        self.loss_kl_div_w2 = self.beta_w2*kl_divergence(Qw2, Pw) / batch_size

        loss = self.loss_y_recon2 + self.loss_kl_div_w2
        loss.backward()

    def optimize_parameters(self):
        # main VAE
        self.optimizer_x.zero_grad()
        self.backward_x()
        self.optimizer_x.step()
        
        # auxiliary VAE
        self.optimizer_y.zero_grad()
        self.backward_y()
        self.optimizer_y.step()

    def get_losses(self):
        # define which losses to get
        loss_names = ["x_recon", "y_recon", "y_recon2", "kl_div_z", "kl_div_w", "kl_div_w2"]
        losses = {}
        for name in loss_names:
            losses[name] = getattr(self, "loss_{}".format(name))
        return losses

    def traverse_y(self, y_mins, y_maxs, n_samples):
        unit_range = torch.arange(0, 1+1e-5, 1.0/(n_samples-1))

        (z, _), _ = self.encode_x(self.x[[0]], sample=False)

        _, n_channel, h, w = self.x.shape
        vdivider = np.ones((1, n_channel, h, 1))
        hdivider = np.ones((1, n_channel, 1, w*n_samples + (n_samples-1)))
        # traverse
        x_recons_all = None
        for y_idx in range(len(y_mins)):
            x_recons = None
            for a in unit_range:
                y_new = torch.clone(self.y[[0]]).cpu() # had to move to cpu for some internal bug in the next line (Apple silicon-related)
                y_new[0, y_idx] = y_mins[y_idx]*(1-a) + y_maxs[y_idx]*a
                y_new = y_new.to(self.device)

                # decode
                w, _ = self.encoder_y(y_new)
                _, x_recon = self.decoder_x(z, w)
                x_recons = as_np(x_recon) if x_recons is None else np.concatenate((x_recons, vdivider, as_np(x_recon)), axis=-1)
            x_recons_all = x_recons if x_recons_all is None else np.concatenate((x_recons_all, hdivider, x_recons), axis=2)
        x_recons_all = np.transpose(x_recons_all, (0, 2, 3, 1))
        return x_recons_all
            
"""
Encoder q(z,w|x): Encode input x to latent variables (z, w)
"""  
class EncoderX(nn.Module):
    def __init__(self, img_channel, hid_channel, hid_dim, z_dim, w_dim):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channel, hid_channel, kernel_size=4, stride=2, padding=1), # (32, 32, 32)
            nn.BatchNorm2d(hid_channel),
            nn.ReLU(),
            nn.Conv2d(hid_channel, hid_channel, kernel_size=4, stride=2, padding=1), # (32, 16, 16)
            nn.BatchNorm2d(hid_channel),
            nn.ReLU(),
            nn.Conv2d(hid_channel, hid_channel, kernel_size=4, stride=2, padding=1), # (32, 8, 8)
            nn.BatchNorm2d(hid_channel),
            nn.ReLU(),
            View((-1, hid_channel*8*8)),
            nn.Linear(hid_channel*8*8, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 2*(z_dim+w_dim)),
        )

    def forward(self, x):
        mean_logvar = self.encoder(x)
        mean_logvar_z = mean_logvar[:, 0:2*self.z_dim]
        mean_logvar_w = mean_logvar[:, 2*self.z_dim:]

        # z
        z_mean, z_logvar = mean_logvar_z.view(-1, self.z_dim, 2).unbind(-1)

        # w
        w_mean, w_logvar = mean_logvar_w.view(-1, self.w_dim, 2).unbind(-1)

        return (z_mean, w_mean), (z_logvar, w_logvar)

"""
Decoder p(x|z,w): Recontruct input x from latent variables (z, w)
"""  
class DecoderX(nn.Module):
    def __init__(self, img_channel, hid_channel, hid_dim, z_dim, w_dim):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim+w_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_channel*8*8),
            nn.ReLU(),
            View((-1, hid_channel, 8, 8)),
            nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=4, stride=2, padding=1), # (32, 16, 16)
            nn.BatchNorm2d(hid_channel),
            nn.ReLU(),
            nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=4, stride=2, padding=1), # (32, 32, 32)
            nn.BatchNorm2d(hid_channel),
            nn.ReLU(),
            nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=4, stride=2, padding=1), # (32, 64, 64)
            nn.BatchNorm2d(hid_channel),
            nn.ReLU(),
            nn.ConvTranspose2d(hid_channel, img_channel, kernel_size=3, stride=1, padding=1) # (img_channel, 64, 64)
        )

    def forward(self, z, w):
        zw = torch.cat([z, w],dim=-1)

        # decode x
        x_logits = self.decoder(zw)
        x_recon = torch.sigmoid(x_logits)
        return x_logits, x_recon

"""
Encoder q(w|y): Recontruct input y from latent variables w
"""  
class EncoderY(nn.Module):
    def __init__(self, y_dim, hid_dim, w_dim):
        super().__init__()
        self.y_dim = y_dim

        self.encoder = nn.ModuleList()
        for y_idx in range(y_dim):
            self.encoder.append(nn.Sequential(
                nn.Linear(1, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, 2)
            ))

    def forward(self, y):
        mu, logvar = [], []
        for y_idx in range(self.y_dim):
            y_i = y[:, y_idx].unsqueeze(-1)
            h = self.encoder[y_idx](y_i)
            mu_i, logvar_i = h.view(-1, 1, 2).unbind(-1)
            mu.append(mu_i)
            logvar.append(logvar_i)
        mu = torch.cat(mu, dim=-1)
        logvar = torch.cat(logvar, dim=-1)

        return mu, logvar

"""
Decoder p(y|w): Recontruct input y from latent variables w
"""  
class DecoderY(nn.Module):
    def __init__(self, y_dim, hid_dim, w_dim):
        super().__init__()
        self.y_dim = y_dim

        self.decoder = nn.ModuleList()
        for y_idx in range(y_dim):
            self.decoder.append(nn.Sequential(
                nn.Linear(1, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, 1),
                nn.Sigmoid()
            ))
        
    def forward(self, w):
        y_recon = []
        for y_idx in range(self.y_dim):
            w_i = w[:, y_idx].unsqueeze(-1)
            y_recon.append(self.decoder[y_idx](w_i))
        y_recon = torch.cat(y_recon, dim=-1)
        return y_recon