"""
All functions related to loss computation and optimization.
"""
import torch


def get_model_fn(model, train=False):
    """
    Returns a function that runs the model in either training or evaluation mode.
    """
    def model_fn(x, labels):
        if not train:
            model.eval()                             # Set model to eval mode
            return model(x, labels)
        else:
            model.train()                            # Set model to train mode
            return model(x, labels)
    return model_fn

def SNR_loss_weighting(diffusion:torch.nn.Module, t: torch.Tensor,T:int =1000) -> torch.Tensor:
    sqrt_alphas_cumprod    = diffusion.sqrt_alphas_cumprod[t.long()]
    sqrt_1m_alphas_cumprod = diffusion.sqrt_1m_alphas_cumprod[t.long()]
    return sqrt_alphas_cumprod/sqrt_1m_alphas_cumprod

def get_ddpm_loss_fn(diffusion, train=True, loss_weight=True):
    """ Returns a function that corresponds to the DDPM MSE-loss """
    def loss_fn(model, batch, old_loss=None, alpha = 0.1):
        model_fn = get_model_fn(model, train=train) # set model either in train of evaluation mode
        time_steps = torch.randint(0, diffusion.T, (batch.shape[0],), device=batch.device)            # Select random time step
        sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(batch.device)                          # Weighting factor 1
        sqrt_1m_alphas_cumprod = diffusion.sqrt_1m_alphas_cumprod.to(batch.device)                    # Weighting factor 2

        noise = torch.randn_like(batch)                                                               # Random Noise
        perturbed_data = sqrt_alphas_cumprod[time_steps, None, None, None] * batch + sqrt_1m_alphas_cumprod[time_steps, None, None, None] * noise     # LinComb of Noise + info
        predicted_noise = model_fn(perturbed_data, time_steps.to(dtype=torch.float32))                                        # Computed predicted noise at t
        losses = torch.square(predicted_noise - noise)
        
        if loss_weight:
            weighting_factor = SNR_loss_weighting(diffusion, time_steps.to(dtype=torch.float32))[:,None,None,None]
            losses = weighting_factor*losses
        loss   = torch.mean(losses)                                                              # Avg over all space dimensions + over all batches
        if old_loss != None:                                                                     
            new_loss = (1-alpha)*old_loss + alpha*loss                                               # APPLY EMA
            return new_loss
        else:
            return loss
    return loss_fn