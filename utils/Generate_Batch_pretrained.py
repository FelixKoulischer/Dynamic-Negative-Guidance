import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image

import random
import os

def set_seed(seed: int) -> None:
    ''' Function to set the seed for all pseudorandom number generators to ave reproducible results

    Input:
        - seed : The desired seed
    Output:
        - None
    '''
    # Python's built-in random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # For some libraries that use environment variables for seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Ensuring reproducibility of operations on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_batch(N_batch: int, diffusion: nn.Module, model: nn.Module, to_forget_model: nn.Module, guidance_type: str, device: str,
                   seed: int=0, initial_noise: torch.Tensor = None,
                   C: int=1, H:int = 32, W:int = 32, T: int=1000,
                   guidance_scale: float=1.,
                   threshold: float=0.04, s_s: float=100., s_m: float=0.1, beta_m: float = 0.2,
                   prior: float=0.2, Temp: float=1., p_min: float=1e-6, p_max: float=0.99, offset: float=0.) -> torch.Tensor:
    '''
    Function to generate a batch of images using a choosable guidance mechanism
    
    Input:
            - N_batch         : The number of desired images
            - diffusion       : The forward diffusion processs, containing the noising schedule
            - model           : The unconditional model
            - to_forget_model : The to-forget model, that can only generate forbidden images
            - guidance_type   : The guidance type, should be one of negative_prompting, safe_latent_diffusion, negative_guidance
            - device          : The devic eon which models should run
            - seed            : The fixed seed (Optional)
            - initial_noise   : The initial sampled noise from N(0,1) (Optional)
            - C, H, W         : The dimension of the to generate images (Default: 1,32,32)
            - guidance_scale  : The guidance scale $\lambda$ (Default: 1.)
            - threshold, s_s, beta_m, s_m : The additional hyperparameters for SLD (Required for SLD)
            - prior, Temp     : The additional hyperparameters for NG (Required for NG)
            - p_min, p_max    : Clamping of posterior (Optional for NG)
            - offset          : Additional bias for NG (Optional for NG)
    Output:
            - A torch.Tensor of size (N_batch,C,H,W) containing the N_batch generated images
    '''
    set_seed(seed)
    assert guidance_type in ['safe_latent_diffusion', 'negative_prompting', 'negative_guidance'], "Guidance type is not defined! Should be one of: safe_latent_diffusion, negative_prompting and negative_guidance "

    if initial_noise is None:
        initial_noise= torch.randn((N_batch, C, H, W)).to(device)
    
    # Negative prompting
    if guidance_type == 'negative_prompting':
        return gen_batch_NP(N_batch, diffusion, model, to_forget_model, device, initial_noise, guidance_scale)
        
    # Safe latent diffusion
    if guidance_type == 'safe_latent_diffusion':
        assert threshold is not None and s_s is not None and s_m is not None and beta_m is not None, "Wrong hyperparameters passed for Safe Latent Diffusion. Should be given: (threshold, s_s, beta_m, s_m)" 
        return gen_batch_SLD(N_batch, diffusion, model, to_forget_model, device, initial_noise, guidance_scale, threshold, s_s, beta_m, s_m)

    # Negative guidance
    if guidance_type == 'negative_guidance':
        assert prior is not None and Temp is not None, "Wrong hyperparameters passed for Negative Guidance. Should be given: (prior, Temp). Optional: (p_min, p_max)" 
        return gen_batch_NG(N_batch, diffusion, model, to_forget_model, device, initial_noise, guidance_scale, prior, Temp, p_min, p_max,offset)


def gen_batch_NP(N_batch: int, diffusion: nn.Module, model: nn.Module, to_forget_model: nn.Module, device: str, initial_noise: torch.Tensor,
                 guidance_scale: float):
    '''
    Function to generate a batch of images using Negative Prompting (NP) as guidance mechanism
    
    Input:
            - N_batch         : The number of desired images
            - diffusion       : The forward diffusion processs, containing the noising schedule
            - model           : The unconditional model
            - to_forget_model : The to-forget model, that can only generate forbidden images\
            - device          : The device on which models should run
            - initial_noise   : The initial sampled noise from N(0,1) (Optional)
            - guidance_scale  : The negative guidance scale $\lambda$ (Default: 1.)

    Output:
            - A torch.Tensor of size (N_batch,C,H,W) containing the N_batch generated images using Negative prompting
    '''
    batch = initial_noise.to(device)
    with torch.no_grad():
        noised_x = batch
    
        for ti in reversed(range(0,diffusion.T)):
            ti_axis = ti*torch.ones(N_batch).to(batch.device)
            beta, sqrt_1m_alphas_cumprod = torch.tensor([diffusion.discrete_betas.to(batch.device)[ti]]).to(batch.device), diffusion.sqrt_1m_alphas_cumprod.to(batch.device)[ti]

            # Compute new gradients
            noise_pred = model(noised_x, ti_axis).sample
            to_forget_noise_pred = to_forget_model(noised_x, ti_axis)

            # Apply the guidance
            guided_noise_pred = noise_pred - guidance_scale * (to_forget_noise_pred-noise_pred)

            # DDPM update rule
            new_noised_x = (noised_x - beta[:, None, None, None]/sqrt_1m_alphas_cumprod * guided_noise_pred) / torch.sqrt(1. - beta)[:, None, None, None] # after nosie_pred 
            if ti !=0:
                noise = torch.randn_like(batch)
                new_noised_x = new_noised_x + torch.sqrt(beta)[:, None, None, None] * noise

            # Update the state
            noised_x = new_noised_x
            
        denoised_x = noised_x  
        return denoised_x

def gen_batch_SLD(N_batch: int, diffusion: nn.Module, model: nn.Module, to_forget_model: nn.Module, device: str, initial_noise: torch.Tensor,
                  guidance_scale: float, threshold: float, s_s: float, beta_m: float, s_m: float) -> torch.Tensor:
    '''
    Function to generate a batch of images using Safe Latent Diffusion (SLD) as guidance mechanism
    
    Input:
            - N_batch         : The number of desired images
            - diffusion       : The forward diffusion processs, containing the noising schedule
            - model           : The unconditional model
            - to_forget_model : The to-forget model, that can only generate forbidden images\
            - device          : The device on which models should run
            - initial_noise   : The initial sampled noise from N(0,1) (Optional)
            - guidance_scale  : The negative guidance scale $\lambda$ (Default: 1.)
            - threshold       : The threshold value from which guidance is activated 
            - s_s             : The rescaling of the difference (afterwards clamped at 1.)
            - beta_m          : The damping factor for the momentum term
            - s_m             : The rescaling of the momentum term
    Output:
            - A torch.Tensor of size (N_batch,C,H,W) containing the N_batch generated images using Negative prompting
    '''
    batch = initial_noise.to(device)
    SLD_correc_with_mom = torch.zeros(batch.size()).to(batch.device)
    with torch.no_grad():
        noised_x = batch
    
        avg_diff = torch.zeros((N_batch)).to(batch.device)
        for ti in reversed(range(0,diffusion.T)):
            ti_axis = ti*torch.ones(N_batch).to(batch.device)
            beta, sqrt_1m_alphas_cumprod = torch.tensor([diffusion.discrete_betas.to(batch.device)[ti]]).to(batch.device), diffusion.sqrt_1m_alphas_cumprod.to(batch.device)[ti]
            
            # Compute new gradients
            noise_pred = model(noised_x, ti_axis).sample
            to_forget_noise_pred = to_forget_model(noised_x, ti_axis)
            
            # Apply the guidance
            # diff = to_forget_noise_pred - noise_pred
            diff = noise_pred - to_forget_noise_pred
            mask = torch.where(diff < threshold, torch.clamp(s_s*torch.abs(diff),max=1), torch.zeros_like(diff))
            SLD_correc = -mask*diff + s_m*SLD_correc_with_mom
            SLD_correc_with_mom = beta_m*SLD_correc_with_mom+(1-beta_m)*SLD_correc
            guided_noise_pred = noise_pred - guidance_scale * SLD_correc

            # DDPM update rule
            new_noised_x = (noised_x - beta[:, None, None, None]/sqrt_1m_alphas_cumprod * guided_noise_pred) / torch.sqrt(1. - beta)[:, None, None, None]
            if ti !=0:
                noise = torch.randn_like(batch)
                new_noised_x = new_noised_x + torch.sqrt(beta)[:, None, None, None] * noise

            # Update the state
            noised_x = new_noised_x
    
        denoised_x = noised_x  
        return denoised_x

def gen_batch_NG(N_batch: int, diffusion: nn.Module, model: nn.Module, to_forget_model: nn.Module, device: str, initial_noise: torch.Tensor,
                 guidance_scale: float, prior: float, Temp: float, p_min: float, p_max: float, offset: float) -> torch.Tensor:
    '''
    Function to generate a batch of images using Safe Latent Diffusion (SLD) as guidance mechanism
    
    Input:
            - N_batch         : The number of desired images
            - diffusion       : The forward diffusion processs, containing the noising schedule
            - model           : The unconditional model
            - to_forget_model : The to-forget model, that can only generate forbidden images\
            - device          : The device on which models should run
            - initial_noise   : The initial sampled noise from N(0,1) (Optional)
            - guidance_scale  : The guidance scale $\lambda$ (Default: 1.)
            - threshold       : The threshold value from which guidance is activated 
            - prior           : The initial prior for negative guidance
            - Temp            : The Temperature for negative guidance
            - p_min           : For clamping of posterior: minimum value (Optional, default = 1e-6)
            - p_max           : For clamping of posterior: maximum value (Optional, default = 0.99)
            - offset          : Additional offset for the added term (Optional, default = 0.0)
    Output:
            - A torch.Tensor of size (N_batch,C,H,W) containing the N_batch generated images using Negative prompting
    '''
    batch = initial_noise.to(device)
    
    p = prior*torch.ones(N_batch).to(batch.device)
    with torch.no_grad():
        noised_x = batch
        for ti in reversed(range(0,diffusion.T)):
            ti_axis = ti*torch.ones(N_batch).to(batch.device)
            beta, sqrt_1m_alphas_cumprod = torch.tensor([diffusion.discrete_betas[ti]]).to(batch.device), diffusion.sqrt_1m_alphas_cumprod[ti].to(batch.device)
    
            # Compute new gradients
            noise_pred = model(noised_x, ti_axis).sample   
            to_forget_noise_pred = to_forget_model(noised_x, ti_axis)
    
            # Apply the guidance
            guidance_scale_t = (guidance_scale*p/(1-p))[:,None,None,None]
            guided_noise_pred = noise_pred - guidance_scale_t*(to_forget_noise_pred-noise_pred)
    
            # DDPM update rule
            new_noised_x = (noised_x - beta[:, None, None, None]/sqrt_1m_alphas_cumprod * guided_noise_pred) / torch.sqrt(1. - beta)[:, None, None, None] # after nosie_pred 
            if ti!=0:
                noise_for_SDE = torch.sqrt(beta)[:, None, None, None] * torch.randn_like(noised_x)
                new_noised_x = new_noised_x + noise_for_SDE
    
            # Compute the posterior
            p = compute_posterior_NG(diffusion, ti, p, new_noised_x, noised_x, noise_pred, to_forget_noise_pred,
                                     Temp=Temp, p_min=p_min, p_max=p_max, offset = offset)
    
            # Update the state
            noised_x = new_noised_x
        return noised_x

def compute_posterior_NG(diffusion: nn.Module, t: int, p_prev: torch.Tensor, new_noised_x: torch.Tensor, noised_x: torch.Tensor,
                         noise_pred: torch.Tensor, to_forget_noise_pred: torch.Tensor, 
                         Temp: float=1.0, p_min: float=1e-6, p_max : float= 0.99, offset: float=0.) -> torch.Tensor:
    '''
    Function to compute the guidance scale for negative guidance
    
    Input:
            - diffusion             : The forward diffusion processs, containing the noising schedule
            - t                     : The current timestep
            - p_prev                : The previous posterior estimation, containing N_batch values
            - new_noised_x          : The noisy state at timestep t
            - noised_x              : The noisy state at timestep t+1
            - noise_pred            : The noise prediction of unconditional model at timestep t+1
            - to_forget_noise_pred  : The noise prediction of bad model at timestep t+1
            - Temp                  : The temperature hyperparameter for NG (Default = 1.)
            - p_min                 : For clamping of posterior: minimum value (Default = 1e-6)
            - p_max                 : For clamping of posterior: maximum value (Default = 0.99)
            - offset          : Additional offset for the added term (Optional, default = 0.0)

    Output:
            - A torch.Tensor of size (N_batch) containing the estimated posterior value for each of the images being generated
    '''
    a_t, a_bar_t = diffusion.alphas[t], diffusion.alphas_cumprod[t]          # Diffusion constants required for Negative Guidance
    a_bar_tmin1 = diffusion.alphas_cumprod[t-1] if t>0 else a_bar_t
    sigma_square_t = (1-a_bar_tmin1)/(1-a_bar_t)*(1-a_t)
    
    # Flatten for easy MSE loss computation
    noised_x = torch.flatten(noised_x, start_dim=1)                                      # Noisy state at t+1
    new_noised_x = torch.flatten(new_noised_x, start_dim=1)                              # Noisy state at t
    to_forget_predicted_noise  = torch.flatten(to_forget_noise_pred, start_dim=1)        # Bad noise prediction at t+1
    predicted_noise = torch.flatten(noise_pred,start_dim=1)                              # Unconditional noise prediction at t+1

    # Compute mu's
    predicted_mean = 1/torch.sqrt(a_t)*(noised_x-(1-a_t)/torch.sqrt(1-a_bar_t)*predicted_noise)                 # Predicted unconditional mean at t
    bad_predicted_mean = 1/torch.sqrt(a_t)*(noised_x-(1-a_t)/torch.sqrt(1-a_bar_t)*to_forget_predicted_noise)   # Predicted bad mean at t

    # Compute error difference
    distance_from_bad = torch.sum((new_noised_x - bad_predicted_mean)**2,dim=1)        # Euclidean distance between x_t and mu_c,t+1
    distance_from_uncond = torch.sum((new_noised_x - predicted_mean)**2,dim=1)         # Euclidean distance between x_t and mu_t+1
    diff = distance_from_bad-distance_from_uncond                                      # Difference between the two terms

    p = p_prev*torch.exp(1/(2*sigma_square_t)*(-Temp*diff+offset))                     # Compute the new posterior
    assert not torch.isnan(p).any(), "Estimated posterior contains NaN values! Most likely is that p_max was too close to 1 at previous timestep. Consider reducing p_max."
    p = torch.clamp(p,min=p_min,max=p_max)                                             # Clamp the posterior between [p_min,p_max]
    return p







