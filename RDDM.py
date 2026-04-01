import numpy as np
import torch
from torch import einsum, nn
import torch.nn.functional as F
from functools import partial
from collections import namedtuple
import math
import random
from einops import rearrange, reduce
from tqdm.auto import tqdm
import cv2

device_0 = 'cuda' if torch.cuda.is_available() else 'cpu'
# device_0 = 'cpu'

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])

def normalization_data(data):
    data_max = data.max()
    data_min = data.min()
    gap = data_max - data_min
    temp_data = data - data_min
    temp_data = temp_data / gap
    return temp_data

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    assert (alphas.sum()-1).abs() < 1e-6

    return alphas*sum_scale

def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    #余弦调度
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32) #返回betas噪声表。貌似并没有用到

def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1

def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class ResidualDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            # loss_type='l2',
            objective='pred_res_noise',
            ddim_sampling_eta=0.,
            condition=False,
            sum_scale=None,
            input_condition=False,
            input_condition_mask=False,
            test_res_or_noise="res_noise"
            # test_res_or_noise="noise"
    ):
        super().__init__()
        assert not (
                type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.input_condition = input_condition
        self.input_condition_mask = input_condition_mask
        self.test_res_or_noise = test_res_or_noise

        self.VFMupsample = nn.Upsample(scale_factor=64 // 16)
        # self.VFMup1 = nn.ConvTranspose2d(1,3,3,2,1,1)
        # self.VFMup2 = nn.ConvTranspose2d(3,1,3,2,1,1)



        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            ddim_sampling_eta = 0.
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        convert_to_ddim=True
        if convert_to_ddim:
            beta_schedule = "linear"
            beta_start = 0.0001
            beta_end = 0.02
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                        torch.linspace(beta_start**0.5, beta_end**0.5,
                                       timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = 0
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = 0
        else:
            alphas = gen_coefficients(timesteps, schedule="decreased")
            betas2 = gen_coefficients(
                timesteps, schedule="increased", sum_scale=self.sum_scale)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                                                 alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def init(self):
        timesteps = 1000

        convert_to_ddim = True
        if convert_to_ddim:
            beta_schedule = "linear"
            beta_start = 0.0001
            beta_end = 0.02
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                        torch.linspace(beta_start**0.5, beta_end**0.5,
                                       timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = alphas[1]
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = betas2[1]

            # adjust
            # alphas = gen_coefficients(timesteps, schedule="average", ratio=0)
            # alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            # alphas_cumsum_prev = F.pad(
            #     alphas_cumsum[:-1], (1, 0), value=alphas_cumsum[1])

            # betas2 = gen_coefficients(
            #     timesteps, schedule="average", sum_scale=self.sum_scale, ratio=0)
            # betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)
            # betas2_cumsum_prev = F.pad(
            #     betas2_cumsum[:-1], (1, 0), value=betas2_cumsum[1])
        else:
            alphas = gen_coefficients(timesteps, schedule="average", ratio=1)
            betas2 = gen_coefficients(
                timesteps, schedule="increased", sum_scale=self.sum_scale, ratio=3)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(
                alphas_cumsum[:-1], (1, 0), value=alphas_cumsum[1])
            betas2_cumsum_prev = F.pad(
                betas2_cumsum[:-1], (1, 0), value=betas2_cumsum[1])

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.alphas = alphas
        self.alphas_cumsum = alphas_cumsum
        self.one_minus_alphas_cumsum = 1-alphas_cumsum
        self.betas2 = betas2
        self.betas = torch.sqrt(betas2)
        self.betas2_cumsum = betas2_cumsum
        self.betas_cumsum = betas_cumsum
        self.posterior_mean_coef1 = betas2_cumsum_prev/betas2_cumsum
        self.posterior_mean_coef2 = (
                                            betas2 * alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum
        self.posterior_mean_coef3 = betas2/betas2_cumsum
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
                (x_t-x_input-(extract(self.alphas_cumsum, t, x_t.shape)-1)
                 * pred_res)/extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
                (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input -
                 extract(self.betas_cumsum, t, x_t.shape) * noise)/extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
                x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
                extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
                extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
                extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, x_input_condition=0, x_self_cond=None, clip_denoised=True):
        if not self.condition:
            x_in = x
        else:
            if self.input_condition:

                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in,
                                  [self.alphas_cumsum[t]*self.num_timesteps,
                                   self.betas_cumsum[t]*self.num_timesteps],
                                  x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            if self.test_res_or_noise == "res_noise":
                pred_res = model_output[0]
                pred_noise = model_output[1]
                pred_res = maybe_clip(pred_res)
                x_start = self.predict_start_from_res_noise(
                    x, t, pred_res, pred_noise)
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "res":
                pred_res = model_output[0]
                pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(
                    x, t, x_input, pred_res)
                x_start = x_input - pred_res
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "noise":
                pred_noise = model_output[1]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                x_start = maybe_clip(x_start)
                pred_res = x_input - x_start
                pred_res = maybe_clip(pred_res)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input-model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t, x_input_condition=0, x_self_cond=None):

        # x_input_condition = self.VFMupsample(x_input_condition)
        # x_input = self.VFMupsample(x_input)

        preds = self.model_predictions(
            x_input, x, t, x_input_condition, x_self_cond)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int, x_input_condition=0, x_self_cond=None):
        b, *_, device = *x.shape, device_0
        batched_times = torch.full(
            (x.shape[0],), t, device=device_0, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x_input, x=x, t=batched_times, x_input_condition=x_input_condition, x_self_cond=x_self_cond)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]
        # if(shape == (1, 1, 64, 64)):
            # x_input = self.VFMupsample(x_input)

        # out_x_start = normalization_data(x_input[0].cpu().numpy().reshape(64,64,1))*255
        # cv2.imwrite('predict//''x_start.png',out_x_start,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        batch, device = shape[0], self.betas.device

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                  torch.randn(shape, device=device)
            # img = x_input + torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        # print('1--->')
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(
                x_input, img, t, x_input_condition, self_cond)

            # out = normalization_data(img.detach().cpu().numpy().reshape(64,64,1))*255
            # cv2.imwrite('predict//'+str(t).zfill(6)+'.png',out,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            if not last:
                img_list.append(img)

        temp,start_temp = self.p_sample(
            x_input, img, t, x_input_condition, self_cond)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True):
        if self.input_condition:
            x_input_condition = x_input[1]
        else:
            x_input_condition = 0
        x_input = x_input[0]
        # if(shape == (1, 1, 64, 64)):
        #     x_input_condition = self.VFMupsample(x_input_condition)

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                  torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"

        if not last:
            img_list = []

        # j=0
        #
        # out = normalization_data(x_input.detach().cpu().numpy().reshape(64,64,1))*255
        # cv2.imwrite('predict//''input.png',out,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # out = normalization_data(img.detach().cpu().numpy().reshape(64,64,1))*255
        # cv2.imwrite('predict//''img.png',out,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # out = normalization_data(x_input_condition.detach().cpu().numpy().reshape(64,64,1))*255
        # # print(out)
        # cv2.imwrite('predict//''x_input_condition.png',out,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])


        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(
                x_input, img, time_cond, x_input_condition, self_cond)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                                                                                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)

            if type == "use_pred_noise":
                img = img - alpha*pred_res + sigma2.sqrt()*noise
            elif type == "use_x_start":
                img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*img + \
                      (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
                      (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
                      sigma2.sqrt()*noise
            if not last:
                img_list.append(img)

        #     out = normalization_data(img.detach().cpu().numpy().reshape(64,64,1))*255
        #     cv2.imwrite('predict//'+str(j).zfill(6)+'res.png',out,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        #     j=j+1
        #
        # j+1
        # i=0

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(
                x_input, img, time_cond, x_input_condition, self_cond)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                                                                                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)

            if type == "use_pred_noise":
                img = img - (betas_cumsum-(betas2_cumsum_next-sigma2).sqrt()) * \
                      pred_noise + sigma2.sqrt()*noise
            elif type == "use_x_start":
                img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*img + \
                      (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
                      (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
                      sigma2.sqrt()*noise

            if not last:
                img_list.append(img)
            # # # if(i==18):
            # # out = normalization_data(img.detach().cpu().numpy().reshape(64,64,1))*255
            # cv2.imwrite('predict//'+str(j).zfill(6)+'noise.png',out,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            # # print(i)
            # # i=i+1
            # j=j+1


        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            # return unnormalize_to_zero_to_one(img_list)
            return img_list
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            # return unnormalize_to_zero_to_one(img_list)
            return img_list




    # @torch.no_grad()
    # def ddim_sample(self, x_input, shape, last=True):
    #     if self.input_condition:
    #         x_input_condition = x_input[1]
    #     else:
    #         x_input_condition = 0
    #     x_input = x_input[0]

    #     batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
    #         0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

    #     # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = torch.linspace(-1, total_timesteps - 1,
    #                            steps=sampling_timesteps + 1)
    #     times = list(reversed(times.int().tolist()))
    #     # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
    #     time_pairs = list(zip(times[:-1], times[1:]))

    #     if self.condition:
    #         img = x_input+math.sqrt(self.sum_scale) * \
    #             torch.randn(shape, device=device)
    #         input_add_noise = img
    #     else:
    #         img = torch.randn(shape, device=device)

    #     x_start = None
    #     type = "use_pred_noise"

    #     if not last:
    #         img_list = []

    #     eta = 0

    #     for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
    #         time_cond = torch.full(
    #             (batch,), time, device=device, dtype=torch.long)
    #         self_cond = x_start if self.self_condition else None
    #         preds = self.model_predictions(
    #             x_input, img, time_cond, x_input_condition, self_cond)

    #         pred_res = preds.pred_res
    #         pred_noise = preds.pred_noise
    #         x_start = preds.pred_x_start

    #         if time_next < 0:
    #             img = x_start
    #             if not last:
    #                 img_list.append(img)
    #             continue

    #         alpha_cumsum = self.alphas_cumsum[time]
    #         alpha_cumsum_next = self.alphas_cumsum[time_next]
    #         alpha = alpha_cumsum-alpha_cumsum_next

    #         betas2_cumsum = self.betas2_cumsum[time]
    #         betas2_cumsum_next = self.betas2_cumsum[time_next]
    #         betas2 = betas2_cumsum-betas2_cumsum_next
    #         # betas2 = 1-(1-betas2_cumsum)/(1-betas2_cumsum_next)
    #         betas = betas2.sqrt()
    #         betas_cumsum = self.betas_cumsum[time]
    #         betas_cumsum_next = self.betas_cumsum[time_next]
    #         sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
    #         sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
    #             betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

    #         if eta == 0:
    #             noise = 0
    #         else:
    #             noise = torch.randn_like(img)

    #         if type == "use_pred_noise":
    #             img = img - alpha*pred_res - \
    #                 (betas_cumsum-(betas2_cumsum_next-sigma2).sqrt()) * \
    #                 pred_noise + sigma2.sqrt()*noise
    #         elif type == "use_x_start":
    #             img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*img + \
    #                 (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
    #                 (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
    #                 sigma2.sqrt()*noise

    #         if not last:
    #             img_list.append(img)

    #     if self.condition:
    #         if not last:
    #             img_list = [input_add_noise]+img_list
    #         else:
    #             img_list = [input_add_noise, img]
    #         return unnormalize_to_zero_to_one(img_list)
    #     else:
    #         if not last:
    #             img_list = img_list
    #         else:
    #             img_list = [img]
    #         return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if self.condition:
            if self.input_condition and self.input_condition_mask:
                x_input[0] = normalize_to_neg_one_to_one(x_input[0])
            else:
                x_input = normalize_to_neg_one_to_one(x_input)
            batch_size, channels, h, w = x_input[0].shape
            h, w =64, 64
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)
        return sample_fn(x_input, size, last=last)

    def q_sample(self, x_start, x_res, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                x_start+extract(self.alphas_cumsum, t, x_start.shape) * x_res +
                extract(self.betas_cumsum, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, imgs, t, noise=None):
        if isinstance(imgs, list):  # Condition
        # if self.condition:  # Condition
            if self.input_condition:
                x_input_condition = imgs[2]
            else:
                x_input_condition = 0
            x_input = imgs[1]
            x_start = imgs[0]  # gt = imgs[0], input = imgs[1]
        else:  # Generation 生成任务只需要输入gt
            x_input = 0
            x_start = imgs

        noise = default(noise, lambda: torch.randn_like(x_start))

        x_res = x_input - x_start

        b, c, h, w = x_start.shape

        # noise sample
        x = self.q_sample(x_start, x_res, t, noise=noise)
        # x为t时刻的图像

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(
                    x_input, x, t, x_input_condition if self.input_condition else 0).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        if not self.condition:
            x_in = x
        else:
            if self.input_condition:
                x_in = torch.cat((x, x_input, x_input_condition), dim=1)
            else:
                x_in = torch.cat((x, x_input), dim=1)

        model_out = self.model(x_in,
                               [self.alphas_cumsum[t]*self.num_timesteps,
                                self.betas_cumsum[t]*self.num_timesteps],
                               x_self_cond)
        # print(model_out[0].shape)
        target = []
        # print(self.objective)
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)
            # print(len(model_out))
            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)

            pred_res = x_input-model_out[0]
            pred_noise = model_out[1]
        elif self.objective == "pred_noise":
            target.append(noise)

            pred_noise = model_out[0]

        elif self.objective == "pred_res":
            target.append(x_res)

            pred_res = model_out[0]

        else:
            raise ValueError(f'unknown objective {self.objective}')

        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t)
            loss = 10000*self.loss_fn(x_u, u_gt, reduction='none')
            return [loss]
        else:
            loss_list = []
            for i in range(len(model_out)):
                print(i)
                loss = self.loss_fn(model_out[i], target[i], reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
                loss_list.append(loss)
            return loss_list

    # def forward(self, img, *args, **kwargs):
    #     if isinstance(img, list):
    #         b, c, h, w, device, img_size, = * \
    #             img[0].shape, img[0].device, self.image_size
    #     else:
    #         b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
    #     # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
    #     t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
    #
    #     if self.input_condition and self.input_condition_mask:
    #         img[0] = normalize_to_neg_one_to_one(img[0])
    #         img[1] = normalize_to_neg_one_to_one(img[1])
    #     else:
    #         img = normalize_to_neg_one_to_one(img)
    #
    #     return self.p_losses(img, t, *args, **kwargs)

    def forward(self, img, bg, condition, *args, **kwargs):
        # gt = imgs[0], input = imgs[1], condition = imgs[2]
        input = []
        input.append(img)
        input.append(bg)
        # temp_condition = self.VFMup1(condition)
        # temp_condition = self.VFMup2(temp_condition)
        # input.append(temp_condition)
        input.append(condition)
        # print(condition)

        # out_x_start = normalization_data(condition[0].cpu().numpy().reshape(64,64,1))*255
        # cv2.imwrite('predict//''condition.png',out_x_start,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        if isinstance(input, list):
            b, c, h, w, device, img_size, = * \
                input[0].shape, input[0].device, self.image_size
        else:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        if self.input_condition and self.input_condition_mask:
            input[0] = normalize_to_neg_one_to_one(input[0])
            input[1] = normalize_to_neg_one_to_one(input[1])
        else:
            # img = normalize_to_neg_one_to_one(img)
            input = normalize_to_neg_one_to_one(input)

        return self.p_losses(input, t, *args, **kwargs)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = device_0
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # x = torch.stack(x)
        # print(x.shape)
        # print(emb.shape)
        emb = x[:, None] * emb[None, :]
        # emb = x[0][:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            self_condition=False,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            condition=False,
            input_condition=False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels + channels * \
                         (1 if self_condition else 0) + channels * \
                         (1 if condition else 0) + channels * (1 if input_condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

class UnetRes(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            self_condition=False,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            num_unet=2,
            condition=False,
            input_condition=False,
            objective='pred_res_noise',
            test_res_or_noise="res_noise"
            # test_res_or_noise="noise"
    ):
        super().__init__()
        self.condition = condition
        self.input_condition = input_condition
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.self_condition = self_condition
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise
        # determine dimensions
        if self.num_unet == 2:
            self.unet0 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition)
            self.unet1 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition)
        elif self.num_unet == 1:
            self.unet0 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition)

    def forward(self, x, time, x_self_cond=None):
        if self.num_unet == 2:
            if self.test_res_or_noise == "res_noise":
                return self.unet0(x, time[0], x_self_cond=x_self_cond), self.unet1(x, time[1], x_self_cond=x_self_cond)
            elif self.test_res_or_noise == "res":
                return self.unet0(x, time[0], x_self_cond=x_self_cond), 0
            elif self.test_res_or_noise == "noise":
                return 0, self.unet1(x, time[1], x_self_cond=x_self_cond)
        elif self.num_unet == 1:
            if self.objective == 'pred_res_noise':
                # num_unet=2
                pass
            elif self.objective == 'pred_x0_noise':
                # num_unet=2
                pass
            elif self.objective == "pred_noise":
                time = time[1]
            elif self.objective == "pred_res":
                time = time[0]
            return [self.unet0(x, time, x_self_cond=x_self_cond)]
