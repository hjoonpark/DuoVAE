import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import itertools
import torch.distributions as dist
from utils.util_model import get_available_devices, View, reparameterize, kl_divergence
from utils.logger import LogLevel
from utils.util_io import as_np
import math

"""
Code references:
- https://github.com/xguo7/PCVAE
- https://github.com/YannDubs/disentangling-vae
"""
class PcVAE(nn.Module):
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
        self.img_channel = params["common"]["img_channel"]
        self.y_dim = params["common"]["y_dim"]
        self.x_recon_weight = params["common"]["x_recon_weight"]
        self.y_recon_weight = params["common"]["y_recon_weight"]
        self.beta = (params["common"]["beta_z"] + params["common"]["beta_w"])*0.5
        self.beta_groupwise = params["pcvae"]["beta_groupwise"]
        self.beta_pairwise = params["pcvae"]["beta_pairwise"]

        # define models
        self.encoder = Encoder(self.img_channel, hid_channel, hid_dim_x, z_dim, w_dim).to(self.device)
        self.decoder = Decoder(self.img_channel, hid_channel, hid_dim_x, z_dim, w_dim, self.y_dim, hid_dim_y).to(self.device)
        
        # used by util.model to save/load model
        self.model_names = ["encoder", "decoder"]
        
        # used by util.model to plot losses
        self.loss_names = ["x_recon", "y_recon", "kl_div", "groupwise_tc", "pairwise_tc"]
        
        if is_train:
            params = itertools.chain(self.encoder.parameters(), self.decoder.parameters())
            self.optimizer = torch.optim.Adam(params, lr=lr)

    def set_input(self, data):
        self.x = data['x'].to(self.device) # (B, img_channel, 64, 64)
        self.y = data['y'].to(self.device) # (B, y_dim)

    def encode(self, x, sample: bool):
        # encode
        (z_mean, w_mean), (z_logvar, w_logvar) = self.encoder(x)
        # sample w, z
        z, Qz = reparameterize(z_mean, z_logvar, sample)
        w, Qw = reparameterize(w_mean, w_logvar, sample)
        return (z, w), (Qz, Qw)

    def decode(self, z, w):
        x_logits, x_recon, y_recon = self.decoder(z, w)
        return x_logits, x_recon, y_recon

    def forward_backward(self):
        # encode
        (self.z, self.w), (Qz, Qw) = self.encode(self.x, sample=True)
        # decode
        x_logits, self.x_recon, self.y_recon = self.decode(self.z, self.w)

        # losses
        # * reconstruction losses are rescaled w.r.t. image and label dimensions so that hyperparameters are easier to tune and consistent regardless of their dimensions.
        # https://github.com/xguo7/PCVAE/blob/dd85743c148b86dd2b583cb074b819f51d6b7a48/disvae/models/losses.py#L676
        batch_size, _, h, w = self.x.shape
        if self.img_channel == 1:
            # treat black pixel as class 0 and white pixel as class 1
            self.loss_x_recon = self.x_recon_weight*F.binary_cross_entropy_with_logits(x_logits, self.x, reduction="sum") / (batch_size*h*w)
        else: 
            # RGB images
            self.loss_x_recon = self.x_recon_weight*F.l1_loss(self.x_recon, self.x, reduction="sum") / (batch_size*h*w)
        self.loss_y_recon = self.y_recon_weight*F.mse_loss(self.y_recon, self.y, reduction="sum") / (batch_size*self.y_dim)

        Pz = dist.Normal(torch.zeros_like(self.z), torch.ones_like(self.z))
        Pw = dist.Normal(torch.zeros_like(self.w), torch.ones_like(self.w))
        self.loss_kl_div = self.beta * (kl_divergence(Qz, Pz) + kl_divergence(Qw, Pw)) / batch_size

        #total correlation loss of all latents for pairwise disentangelment for mutiple properties
        w_mean, w_logvar = Qw.loc, (Qw.scale**2).log()
        log_pw, log_qw, log_prod_qwi, log_q_wCx = _get_log_pz_qz_prodzi_qzCx(latent_sample=self.w,
                                                                            latent_dist=(w_mean, w_logvar),
                                                                            n_data=None, # n_data is not used. Ideally, len(dataset)
                                                                            is_mss=False)
        # TC[z] = KL[q(z)||\prod_i z_i]
        self.loss_pairwise_tc = -self.beta_pairwise * (log_qw - log_prod_qwi).mean() 
        #total correlation loss between w and z (groupwise disentangelment)
        z_mean, z_logvar = Qz.loc, (Qz.scale**2).log()
        log_pwz, log_qwz, log_prod_qwqz, log_q_wzCx = _get_log_pzw_qzw_prodzw_qzwCx(latent_sample_z=self.z,
                                                                            latent_sample_w=self.w,
                                                                            latent_dist_z=(z_mean, z_logvar),
                                                                            latent_dist_w=(w_mean, w_logvar),
                                                                            n_data=None, # n_data is not used. Ideally, len(dataset)
                                                                            is_mss=False)
        #TC[z,w] = KL[q(z,w)||\z,w]
        self.loss_groupwise_tc = -self.beta_groupwise * (log_qwz - log_prod_qwqz).mean()        

        # total loss
        loss = self.loss_x_recon + self.loss_y_recon \
                + self.loss_kl_div + self.loss_pairwise_tc + self.loss_groupwise_tc
        loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward_backward()
        self.optimizer.step()

    def iterate_get_w(self, label, w_latent_idx, maxIter=20):
        # https://github.com/xguo7/PCVAE/blob/dd85743c148b86dd2b583cb074b819f51d6b7a48/disvae/models/vae.py#L318
        # get the w for a kind of given property
        w_n=label.view(-1,1).to(self.device).float() # [N]
        for iter_index in range(maxIter):
            summand = self.decoder.property_lin_list[w_latent_idx](w_n)
            w_n1 = label.view(-1,1).to(self.device).float() - summand
            w_n = w_n1.clone()
        return w_n1.view(-1) 

"""
Encoder: Encode input x to latent variables (z, w)
"""
class Encoder(nn.Module):
    # https://github.com/xguo7/PCVAE/blob/main/disvae/models/encoders.py
    def __init__(self, img_channel, hid_channel, hid_dim, z_dim, w_dim):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channel, hid_channel, kernel_size=4, stride=2, padding=1, bias=False), # (32, 32, 32)
            nn.BatchNorm2d(hid_channel),
            LeakyReLU(0.2, True),
            nn.Conv2d(hid_channel, hid_channel, kernel_size=4, stride=2, padding=1, bias=False), # (32, 16, 16
            nn.BatchNorm2d(hid_channel),
            LeakyReLU(0.2, True),
            nn.Conv2d(hid_channel, hid_channel, kernel_size=4, stride=2, padding=1, bias=False), # (32, 8, 8)
            nn.BatchNorm2d(hid_channel),
            LeakyReLU(0.2, True),
            View((-1, hid_channel*8*8)),
            nn.Linear(hid_channel*8*8, hid_dim),
            LeakyReLU(0.2, True),
            nn.Linear(hid_dim, hid_dim),
            LeakyReLU(0.2, True),
            nn.Linear(hid_dim, (z_dim+w_dim) * 2)
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
Decoder: Recontruct input x from latent variables (z, w)
"""
class Decoder(nn.Module):
    # https://github.com/xguo7/PCVAE/blob/main/disvae/models/decoders.py
    def __init__(self, img_channel, hid_channel, hid_dim, z_dim, w_dim, y_dim, hid_dim_y):
        super().__init__()

        self.y_dim = y_dim
        # decoder for the image
        self.decoder = nn.Sequential(
            nn.Linear(z_dim+w_dim, hid_channel),
            LeakyReLU(0.2, True),
            nn.Linear(hid_channel, hid_channel),
            LeakyReLU(0.2, True),
            nn.Linear(hid_channel, hid_channel*8*8),
            LeakyReLU(0.2, True),
            View((-1, hid_channel, 8, 8)),
            nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=4, stride=2, padding=1, bias=False), # (32, 16, 16)
            nn.BatchNorm2d(hid_channel),
            LeakyReLU(0.2, True),
            nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=4, stride=2, padding=1, bias=False), # (32, 32, 32)
            nn.BatchNorm2d(hid_channel),
            LeakyReLU(0.2, True),
            nn.ConvTranspose2d(hid_channel, hid_channel, kernel_size=4, stride=2, padding=1), # (32, 64, 64)
            LeakyReLU(0.2, True),
            nn.ConvTranspose2d(hid_channel, img_channel, kernel_size=3, stride=1, padding=1) # (img_channel, 64, 64)
        )

        # decoder for the property 
        self.property_lin_list = nn.ModuleList()
        for property_idx in range(y_dim):
            layers = []
            layers.append(spectral_norm_fc(nn.Linear(1, hid_dim_y)))
            layers.append(LeakyReLU(0.2, True))
            layers.append(spectral_norm_fc(nn.Linear(hid_dim_y, 1)))
            layers.append(nn.Sigmoid())
            self.property_lin_list.append(nn.Sequential(*layers))

    def forward(self, z, w):

        prop=[]
        #fully connected process for reconstruct the properties
        for idx in range(self.y_dim):
            w_=w[:,idx].view(-1,1)
            prop.append(self.property_lin_list[idx](w_)+w_)

        # decode
        zw = torch.cat([z, w],dim=-1)
        x_logits = self.decoder(zw)
        x_recon = torch.sigmoid(x_logits)
        y_recon = torch.cat(prop,dim=-1)

        return x_logits, x_recon, y_recon
        
"""
Helper math functions
"""
def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=False):
    # https://github.com/YannDubs/disentangling-vae/blob/7b8285baa19d591cf34c652049884aca5d8acbca/disvae/models/losses.py#L523
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)
    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx

def _get_log_pzw_qzw_prodzw_qzwCx(latent_sample_z,latent_sample_w, latent_dist_z, latent_dist_w, n_data, is_mss=False):
    # https://github.com/xguo7/PCVAE/blob/dd85743c148b86dd2b583cb074b819f51d6b7a48/disvae/models/losses.py#L640
    batch_size, hidden_dim_z = latent_sample_z.shape
    batch_size, hidden_dim_w = latent_sample_w.shape
    hidden_dim=hidden_dim_z+hidden_dim_w
    latent_dist=(torch.cat([latent_dist_z[0],latent_dist_w[0]],dim=-1), torch.cat([latent_dist_z[1],latent_dist_w[1]],dim=-1))
    latent_sample=torch.cat([latent_sample_z,latent_sample_w],dim=-1)
    
    # calculate log q(z,w|x)
    log_q_zwCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z,w)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pzw = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qzqw = matrix_log_density_gaussian(latent_sample, *latent_dist)
    mat_log_qz = matrix_log_density_gaussian(latent_sample_z, *latent_dist_z)
    mat_log_qw = matrix_log_density_gaussian(latent_sample_w, *latent_dist_w)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qzqw = mat_log_qzqw + log_iw_mat.view(batch_size, batch_size, 1)
        log_iw_mat_z = log_importance_weight_matrix(batch_size, n_data).to(latent_sample_z.device)
        mat_log_qz = mat_log_qz + log_iw_mat_z.view(batch_size, batch_size, 1)        
        log_iw_mat_w = log_importance_weight_matrix(batch_size, n_data).to(latent_sample_w.device)
        mat_log_qw = mat_log_qw + log_iw_mat_w.view(batch_size, batch_size, 1)

    log_qzw = torch.logsumexp(mat_log_qzqw.sum(2), dim=1, keepdim=False)
    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_qw = torch.logsumexp(mat_log_qw.sum(2), dim=1, keepdim=False)
    log_prod_qzqw = log_qz + log_qw

    return log_pzw, log_qzw, log_prod_qzqw, log_q_zwCx

def matrix_log_density_gaussian(x, mu, logvar):
    # https://github.com/YannDubs/disentangling-vae/blob/7b8285baa19d591cf34c652049884aca5d8acbca/disvae/utils/math.py#L8
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.
    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).
    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).
    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).
    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)

def log_density_gaussian(x, mu, logvar):
    # https://github.com/YannDubs/disentangling-vae/blob/7b8285baa19d591cf34c652049884aca5d8acbca/disvae/utils/math.py#L34
    """Calculates log density of a Gaussian.
    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.
    mu: torch.Tensor or np.ndarray or float
        Mean.
    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density

"""
Soft Spectral Normalization (use for fc-layers)
https://github.com/xguo7/PCVAE/blob/main/disvae/models/spectral_norm_fc.py
Adpated from: https://arxiv.org/abs/1802.05957
"""
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter

class SpectralNorm(object):
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.

    def __init__(self, coeff, name='weight', n_power_iterations=5, dim=0, eps=1e-12):
        self.coeff = coeff
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                            'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, do_power_iteration):
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important bahaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is alreay on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        sigma_log = getattr(module, self.name + '_sigma') # for logging
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor

        # for logging
        sigma_log.copy_(sigma.detach())

        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module, name, coeff, n_power_iterations, dim, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                    "the same parameter {}".format(name))

        fn = SpectralNorm(coeff, name, n_power_iterations, dim, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1).to(weight.device))

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormLoadStateDictPreHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(self, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        version = local_metadata.get('spectral_norm', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + fn.name + '_orig']
                weight = state_dict.pop(prefix + fn.name)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + fn.name + '_u']
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[prefix + fn.name + '_v'] = v

# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormStateDictHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError("Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.fn._version

def spectral_norm_fc(module, coeff=0.97, name='weight', n_power_iterations=5, eps=1e-12, dim=None):
    r"""Applies spectral normalization to a parameter in the given module.
    .. math::
        \mathbf{W} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
    Spectral normalization stabilizes the training of discriminators (critics)
    in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.
    See `Spectral Normalization for Generative Adversarial Networks`_ .
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectal norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is 0, except for modules that are instances of
            ConvTranspose1/2/3d, when it is 1
    Returns:
        The original module with the spectal norm hook
    Example::
        >>> m = spectral_norm(nn.Linear(20, 40))
        Linear (20 -> 40)
        >>> m.weight_u.size()
        torch.Size([20])
    """
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                                torch.nn.ConvTranspose2d,
                                torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, coeff, n_power_iterations, dim, eps)
    return module

def remove_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(
        name, module))