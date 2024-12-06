import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run 'main.py' with specified hyperparameters. Computes FID, KL-divergence and safety level. Arguments that should always be passed: N_tot, N_batch, guidance_type, guidance_scale. Look at which parameters are required for each guidance type.")

    # Set random seed for reproducible results
    parser.add_argument('--seed',type=int,default=0, help='Random seed to reproduce results')

    # Set device
    parser.add_argument('--device',type=str,default='cuda', help='Device on which everything should be run')
    
    # Define hyperparameters that should always be passed when running the code
    parser.add_argument('--N_tot', type=int, default=2048, help='Total number of images generated')
    parser.add_argument('--N_batch', type=int, default=512, help='Batch size of images generated')
    parser.add_argument('--guidance_type', type=str, default='negative_prompting', help='Guidance type to avoid forbidden regions. Can be set to "negative_prompting, "safe_latent_diffusion" or "negative_guidance" (ours)')
    parser.add_argument('--to_remove_class', type=int, default=0,help='The CIFAR10 class number that should be removed using the negative guidance scheme (should be between 0 and 4)')
    parser.add_argument('--guidance_scale', type=float, default=1., help='The guidance scale $\lambda$.')

    # Define hyperparameters that need to be passed if using SFD
    parser.add_argument('--threshold', type=float, default=0.04, help='Threshold value for SFD.')
    parser.add_argument('--s_s', type=float, default=100., help='Rescaling of guidance for SFD.')
    parser.add_argument('--beta_m', type=float, default=0.1, help='Exponential decay parameter for momentum in SFD.')
    parser.add_argument('--s_m', type=float, default=0.2, help='Rescaling of momentum in SFD.')
    
    # Define hyperparameters that need to be passed if using NG
    parser.add_argument('--prior', type=float, default=0.2, help='Initial prior value for NG.')
    parser.add_argument('--Temp', type=float, default=1., help='The temperature value for NG')
    parser.add_argument('--offset', type=float, default=0., help='The offset value for NG')
    parser.add_argument('--p_min', type=float, default=1e-6, help='Minimal value for posterior')
    parser.add_argument('--p_max', type=float, default=0.99, help='Maximal value for posterior')

    # Define more general hyperparameters that are not constantly passed explicitly
    parser.add_argument('--C', type=int, default=3, help='Number of channels in the input image.')
    parser.add_argument('--H', type=int, default=32, help='Height of the input image.')
    parser.add_argument('--W', type=int, default=32, help='Width of the input image.')
    parser.add_argument('--n_feat', type=int, default=8, help='Number of feature channels for the first CNN inside the UNet.')
    parser.add_argument('--t_emb_dim', type=int, default=32, help='Size of the positional time embeddings for the UNet.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Size of the kernel for the CNNs inside the UNet.')
    parser.add_argument('--stride', type=int, default=1, help='Stride for the CNNs inside the UNet.')
    parser.add_argument('--padding', type=int, default=1, help='Padding for the CNNs inside the UNet.')

    parser.add_argument('--T', type=int, default=1000, help='Number of diffusion timesteps.')
    parser.add_argument('--rho', type=int, default=1, help='Exponent for the EDM noising schedule. Setting to 1 is linear schedule and to 7 is EDM schedule.')
    parser.add_argument('--beta_min', type=int, default=1e-4, help='Smallest noise added, present at timestep 0.')
    parser.add_argument('--beta_max', type=int, default=2e-2, help='Largest noise added, present at timestep T.')

    parser.add_argument('--data_batch_size', type=int, default=64, help='Batch size of training data.')
    parser.add_argument('--data_batch_size_test', type=int, default=1024, help='Batch size of test data.')

    return parser.parse_args()
