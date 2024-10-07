"""
Library with the samplers and functions related to the sampling processes both forward and backwards
"""
import abc
import torch
import numpy as np
from tqdm import tqdm
from .losses import get_model_fn

import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image

_SAMPLERS = {}                                                       # Global dictoinarry containing all the samplers

def register_sampler(cls=None, *, name=None):
    """A decorator for registering sampler classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__                                # Take name of sampler
        else:
            local_name = name                                        # If explicitly given
        if local_name in _SAMPLERS:
            raise ValueError(f'Already registered model with name: {local_name}')     # Check whether already in the dictionary
        _SAMPLERS[local_name] = cls                                  # If not in dictionarry yet => add it under the chosen name
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_sampler(name):
    """ Function that returns the desired sample 
            => simply takes it from the global dictionaary "_SAMPLERS" """
    return _SAMPLERS[name]


# Define Abstract classes that take the different parts of the models and groups them together
class Sampler(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, diffusion, model_fn):
        super().__init__()
        self.diffusion = diffusion
        self.model_fn = model_fn

    @abc.abstractmethod               # Gets the update_fn function from the samplers
    def update_fn(self, x, t):
        """One update of the sampler.

        Args:
            x: A PyTorch tensor representing the current state        [N_batch,C,H,W]
            t: A PyTorch tensor representing the current time step.   [N_batch]

        Returns:
            x: A PyTorch tensor of the next state.                    [N_batch,C,H,W]
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.     [N_batch,C,H,W]
        """
        pass


@register_sampler(name='ancestral_sampling')    # Registers the DDPM sampler in the global dictionarry
class AncestralSampling(Sampler):
    """The ancestral sampler used in the DDPM paper"""

    def __init__(self, diffusion, model_fn):
        super().__init__(diffusion, model_fn)   # Initializes the given sub-models

    def update_fn(self, x, t):
        diffusion = self.diffusion

        beta = diffusion.discrete_betas.to(t.device)[t.long()]
        std  = diffusion.sqrt_1m_alphas_cumprod.to(t.device)[t.long()]

        # set the model either for training or evaluation
        predicted_noise = self.model_fn(x, t.to(dtype=torch.float32))    # Compute predicted noise: forward pass through the UNet WATCH OUT ADDED .to(dtype=torch.float32)
        score = - predicted_noise / std[:, None, None, None]             # Rescale by the std and change sign => score

        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]   # One backward step (without noise => ODE!)
        noise = torch.randn_like(x)                                                                     # Sample some noise
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise                                      # Add noise to current state => SDE!
        return x, x_mean                                              # Returns both SDE and ODE!! => chose which one you want to use
    
# Define Abstract classes that take the different parts of the models and groups them together
class HopSampler(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, diffusion, model_fn, Hopfield):
        super().__init__()
        self.diffusion = diffusion
        self.model_fn = model_fn
        self.Hopfield = Hopfield

    @abc.abstractmethod               # Gets the update_fn function from the samplers
    def update_fn(self, x, t):
        """One update of the sampler.

        Args:
            x: A PyTorch tensor representing the current state        [N_batch,C,H,W]
            t: A PyTorch tensor representing the current time step.   [N_batch]

        Returns:
            x: A PyTorch tensor of the next state.                    [N_batch,C,H,W]
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.     [N_batch,C,H,W]
        """
        pass

@register_sampler(name='guided_ancestral_sampling')
class Guided_AncestralSampling(HopSampler):
    """The MHN guided DDPM sampler"""
    def __init__(self, diffusion, model_fn, Hopfield):
        super().__init__(diffusion, model_fn, Hopfield)

    def update_fn(self, x, t, t_next, eta=0):
        diffusion = self.diffusion

        beta = diffusion.discrete_betas.to(t.device)[t.long()]
        std  = diffusion.sqrt_1m_alphas_cumprod.to(t.device)[t.long()]

        # set the model either for training or evaluation
        predicted_noise = self.model_fn(x, t.to(dtype=torch.float32))    # Compute predicted noise: forward pass through the UNet WATCH OUT ADDED .to(dtype=torch.float32)
        score = - predicted_noise / std[:, None, None, None]             # Rescale by the std and change sign => score

        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]   # One backward step (without noise => ODE!)

        # Compute the guidance
        lam = 0.01                                                                                              # Guidance scale
        p = self.Hopfield.probability(t[0],x).unsqueeze(dim=1)                                                  # Unsqueeze such that can be multiplied with the gradient which has size [36,784]
        print('(prob of shown, Avg prob): ',(p[0].item(),p.mean().item()))                                                   # Extra logging
        print(p.size())
        Hop_grad = self.Hopfield.gradient(t[0],x)

        MHN_Guidance = p/(1-p)*Hop_grad(t[0],x)                                                             # Compute correctly weighted complement energy gradient WATCH OUT SHOULD BE -
        x_mean -= lam*MHN_Guidance.view(-1, 1, 28, 28)                                                           # Gradient DESCENT and not ascent

        noise = torch.randn_like(x)                                                                     # Sample some noise
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise                                      # Add noise to current state => SDE!
        return x, x_mean



def sampling_fn(sampler_name,diffusion, model, shape, device, T=None, denoise=True):
    """If T is given then starts from that diffusion time step"""
    model_fn = get_model_fn(model, train=False)                       # Gets the desired model from losses.py
    sampler_method = get_sampler(sampler_name.lower())     # Gets the desired sampler "method" (.lower() to make sure 'DDPM' = 'ddpm') NO YET INITIALISED
    sampler = sampler_method(diffusion, model_fn)                     # Load the modules into the desired sampler

    if T is None:                    # if no time is given => simply assume max time step is desired!
        T = diffusion.T

    with torch.no_grad():            # No need for gradients!
        x = diffusion.prior_sampling(shape).to(device)                           # Sample x from the prior
        timesteps = torch.flip(torch.arange(0, T, device=device), dims=(0,))     # reverse time partition [T, 0] => we go backwards in time

        for timestep in tqdm(timesteps):
            t = torch.ones(shape[0], device=device) * timestep                   # t is a tensor consisting of [N_batch] identical times t
            x, x_mean = sampler.update_fn(x, t)                                         # Apply one step trough the sampler
        return x_mean if denoise else x                                 # If you're denoising return ODE => else SDE, inverse_scaler??











################################ Fast Samplers ########################################
    
# You need a new abstract class because fast samplers coded such that they also return the position of the next time_step! => could be modified I guess
class FastSampler(abc.ABC):
    """The abstract class for a Fast Sampler algorithm."""

    def __init__(self, diffusion, model_fn):
        super().__init__()
        self.diffusion = diffusion
        self.model_fn = model_fn

    @abc.abstractmethod
    def update_fn(self, x, t, t_next):
        """One update of the sampler.
        Args:
            x: A PyTorch tensor representing the current state
            t: A PyTorch tensor representing the current time step.
            t_next: A PyTorch tensor representing the next time step.
        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_sampler(name='ddim')
class DDIM(FastSampler):
    """The DDIM sampler"""

    def __init__(self, diffusion, model_fn):
        super().__init__(diffusion, model_fn)

    def update_fn(self, x, t, t_next, eta=0):
        # NOTE: We are producing each predicted x0, not x_{t-1} at timestep t. 

        t = t.long().to(x.device)                        # Integer values of the timesteps
        t_next = t_next.long().to(x.device)              # Integer values of the timesteps for the NEXT STEP
        
        at = self.diffusion.alphas_cumprod.to(x.device)[t][:, None, None, None]                      # Compute alpha for current timestep
        at_next = self.diffusion.alphas_cumprod.to(x.device)[t_next][:, None, None, None]            # Compute alpha for NEXT timestep

        # noise estimation
        et = self.model_fn(x, t.float())                 # U-Net needs a float tensor

        # predicts x_0 by direct substitution
        x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()

        # noise controlling the Markovia/Non-Markovian property
        sigma_t = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()

        # update using forward posterior q(x_t-1|x_t, x0_t)
        x = at_next.sqrt() * x0_t + (1 - at_next- sigma_t**2).sqrt() * et + sigma_t * torch.randn_like(x) 
        return x, x0_t


################################################ Utils for PSDM sampler #########################################


def get_fast_sampler(config, diffusion, model, inverse_scaler, sampler_name="ddim"):
    """ Creates a fast sampling function
        Note that DDPM fast sampler runs DDIM with eta=1.0
    """
    model_fn = get_model_fn(model, train=False)  # get noise predictor model in evaluation mode
    # sampler_method = get_sampler(config.sampling.sampler.lower())
    if sampler_name == "ddpm":
        sampler_method = get_sampler("ddim")
    else:
        sampler_method = get_sampler(sampler_name)
    sampler = sampler_method(diffusion, model_fn)

    def ddim_sampling_fn(x, ts, t_nexts, denoise=True):
        with torch.no_grad():
            # reverse time partition [T, 0]
            for i, j in tqdm(zip(reversed(ts), reversed(t_nexts)), total=len(ts)):
                t = (torch.ones(x.shape[0]) * i).to(config.device)
                t_next = (torch.ones(x.shape[0]) * j).to(config.device)
                x, x_mean = sampler.update_fn(x, t, t_next)
            return inverse_scaler(x_mean if denoise else x)

    def ddpm_sampling_fn(x, ts, t_nexts, denoise=True):
        with torch.no_grad():
            # reverse time partition [T, 0]
            for i, j in tqdm(zip(reversed(ts), reversed(t_nexts)), total=len(ts)):
                t = (torch.ones(x.shape[0]) * i)
                t_next = (torch.ones(x.shape[0]) * j)
                x, x_mean = sampler.update_fn(x, t, t_next, eta=1.0)
            return inverse_scaler(x_mean if denoise else x)

    if sampler_name == "ddim":
        return ddim_sampling_fn
    elif sampler_name == "ddpm":
        return ddpm_sampling_fn
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")



############################################           UTILS            ###############################################
    
def get_time_sequence(denoising_steps=10, T=1000, skip_type="uniform", late_t=None):

    if late_t is None:
        # evenly spaced numbers over a half open interval
        if skip_type == "uniform":
            skip = T // denoising_steps
            seq = np.arange(0, T, skip)
        elif skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(T * 0.8), denoising_steps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
    else:
        # evenly spaced numbers over a specified closed interval.
        seq = np.linspace(0, late_t, num=denoising_steps, dtype=int)

    seq_next = [-1] + list(seq[:-1])

    return seq, seq_next
