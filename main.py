# Import all the required modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import torchvision
from torchvision import transforms
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image

import random
import csv
import os

# Import the hyperparameter
### TO WRITE!!!

# Import the useful models 
from models.ddpm import UNet
from models.Inception import InceptionV3

# Import all the useful features
from utils.Generate_Batch import generate_batch
from utils.Metrics import extract_features, get_images_stats, KL_with_ground_truth, KL_with_uniform, calculate_frechet_distance, obtain_generated_distribution
from utils.Generate_Batch import generate_batch

# Load model and set diffusion class
from diffusion.losses import get_model_fn
from diffusion.forward_diffusion import GaussianDiffusion
from diffusion.sampling import sampling_fn

# For the classifier from hugging face
import torchvision.transforms as transforms
from transformers import pipeline

# Load the hyperparameters
from Hyperparameters.parse_hyperparameters import parse_args

# Initialise the config
from utils.default_cifar10_configs import get_config

def main():
    # Parse arguments
    args = parse_args()

    # Set device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Print the hyperparameters to verify
    print(f'\nTotal number of images generated for analysis: {args.N_tot}')
    print(f'Size of batch: {args.N_batch}')
    print(f'Everything should be running on: {args.device}')
    print(f'\nGuidance type: {args.guidance_type}')
    #print(f'Guidance scale: {args.guidance_scale}')
    if args.guidance_type == 'negative_guidance':
        print(f'Prior: {args.prior}')
        print(f'Temp: {args.Temp}')
    if args.guidance_type == 'safe_latent_diffusion':
        print(f'Threshold: {args.threshold}')
        print(f's_s: {args.s_s}')
        print(f'beta_m: {args.beta_m}')
        print(f's_m: {args.s_m}')
        
    print(f'Guidance scale: {args.guidance_scale}')
    Run_Analysis(args)

def filter_loader(data_loader, exclude_label=0):
    print(f'Watch out: filtering class number {exclude_label} for FID computation. If this is not the one you wanted, specify so in main.py')
    for real_imgs, labels in data_loader:
        mask = labels != exclude_label
        filtered_imgs = real_imgs[mask]
        filtered_labels = labels[mask]
        yield filtered_imgs, filtered_labels

def Run_Analysis(args):
    # Load the diffusion module containg the forward process
    diffusion = GaussianDiffusion(beta_min = args.beta_min, beta_max = args.beta_max, rho= args.rho, T=args.T)  # defines the diffusion process

    # Load the required data (in this case MNIST)
    transform = transforms.Compose([
    transforms.Resize((32, 32)),          # Resize the image to 32x32
    transforms.ToTensor(),                # Convert the image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.data_batch_size, shuffle=True, drop_last=True)
    loader_test_shuffled = torch.utils.data.DataLoader(dataset, batch_size=args.data_batch_size_test, shuffle=True, drop_last=True)
    #print('\nMNIST-data loaded')
    
    # Filter out examples containing only the digit "0"
    filtered_loader = filter_loader(loader, exclude_label=0)

    #Load default config for the models
    config = get_config()

    # Load the models: First unconditional model
    model = UNet(config)
    model.load_state_dict(torch.load(f'models/checkpoint_CIFAR10_unconditional.pth')['model'])
    model.to(args.device)
    model.eval()
    model_fn = get_model_fn(model, train=False) 
    
    to_forget_model = UNet(config)
    #loaded_state = torch.load(ckpt_dir, map_location=device)
    #model.load_state_dict(loaded_state['net'])
    to_forget_model.load_state_dict(torch.load(f'models/checkpoint_CIFAR10_only_planes.pt')['net'])
    to_forget_model.to(args.device)
    to_forget_model.eval()
    to_forget_model_fn = get_model_fn(to_forget_model, train=False)
    
    # Load the models: Third the classifier to label the generated samples
    # Initialize the Hugging Face pipeline for image classification
    classifier_pipeline = pipeline("image-classification", model="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10", device=args.device)
    pretrained_classifier = classifier_pipeline.model
    pretrained_classifier.to(args.device)
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),   # Resize images to 224x224
        #transforms.ToPILImage(),        # Convert tensors to PIL images
        transforms.ToTensor(),           # Convert PIL images to tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [0, 1]
    ])
        
    # Load the models: Fourth the InceptionV3 network to compute the FID
    incept = InceptionV3().to(args.device)
    incept.eval()
    
    # Load local name for the hyperparameters
    device = args.device
    C, H, W = args.C, args.H, args.W
    guidance_scale = args.guidance_scale
    threshold, s_s, s_m, beta_m = args.threshold, args.s_s, args.s_m, args.beta_m
    prior, Temp = args.prior, args.Temp
    
    guidance_type = args.guidance_type
    
    N_batch, N_tot = args.N_batch, args.N_tot
    
    # Compute statistics of entire CIFAR10 data
    real_imgs_features = []
    for k, (real_imgs, labels) in enumerate(loader):
        real_imgs_features_i = extract_features(real_imgs, incept, device)
        real_imgs_features.append(real_imgs_features_i.cpu().numpy())
    real_imgs_features = np.vstack(real_imgs_features)
    mu_real, cov_real = np.mean(real_imgs_features,axis=0),  np.cov(real_imgs_features.T)
    print('\nFinished analysing full CIFAR dataset')

    # Compute statistics of filtered MNIST data (no "0"s)
    real_imgs_features_filt = []
    for k, (real_imgs_filt, labels) in enumerate(filtered_loader):
        real_imgs_features_filt_i = extract_features(real_imgs_filt, incept, device)
        real_imgs_features_filt.append(real_imgs_features_filt_i.cpu().numpy())
    real_imgs_features_filt = np.vstack(real_imgs_features_filt)
    mu_filt, cov_filt = np.mean(real_imgs_features_filt,axis=0),  np.cov(real_imgs_features_filt.T)
    print('\nFinished analysing filtered CIFAR dataset (without class "0")')
    
    # Compute statistics of generated MNIST data
    fake_imgs_features = []
    distr, label_list = torch.zeros((10,2)), torch.tensor([0,1,2,3,4,5,6,7,8,9])
    num_batches_needed = N_tot//N_batch
    for i in range(0,num_batches_needed):
        fake_imgs = generate_batch(N_batch, diffusion, model, to_forget_model, guidance_type, args.device, seed=args.seed+i, 
                                  C = args.C, H = args.H, W = args.W, T = args.T,
                                  guidance_scale = args.guidance_scale,
                                  threshold=threshold, s_s=args.s_s, beta_m=args.beta_m, s_m = args.s_m,
                                  prior=args.prior, Temp=args.Temp, p_min=args.p_min, p_max=args.p_max)
        distr += obtain_generated_distribution(label_list,(fake_imgs+1)/2, classifier_pipeline)/num_batches_needed # Very important rescale between [0,1] instead of [-1,1]
        fake_imgs_features_i = extract_features(fake_imgs, incept, device)
        fake_imgs_features.append(fake_imgs_features_i.cpu().numpy())
        torch.cuda.empty_cache()
        print(f'Generated {i+1} batches.') 
    fake_imgs_features = np.vstack(fake_imgs_features)

    print(f'Watch out: analysing the removal of class 0. If this is not the one you wanted, specify so in main.py')
    num_wrong = distr[0,1].item()
    
    ground_truth_distr = 1/9*torch.ones((10,2))
    ground_truth_distr[:,0] = torch.linspace(0,9,10)
    ground_truth_distr[0,1] = 0.

    plt.figure()
    plt.bar(distr[:,0].numpy(), distr[:,1].numpy(), color='blue',label='Measured')
    plt.bar(ground_truth_distr[:,0].numpy(), ground_truth_distr[:,1].numpy(), color='red', alpha=0.4,label='Ground Truth')
    plt.legend()
    plt.xlabel(' Image label ')
    plt.xticks(label_list)
    plt.ylabel(' Occurence in denoised output ')
    plt.title(f' Distribution of generated images with lambda = {guidance_scale}')
    plt.ylim(0,1)
    if guidance_type == 'negative_prompting':
        plt.savefig(f'Results/Figures/NP/Distribution_negative_prompt_lam_{guidance_scale}.png')
    if guidance_type == 'safe_latent_diffusion':
        plt.savefig(f'Results/Figures/SLD/Distribution_safe_latent_diffusion_lam_{guidance_scale}_thresh_{threshold}_ss_{s_s}_betam_{beta_m}_sm_{s_m}.png')
    if guidance_type == 'negative_guidance':
        plt.savefig(f'Results/Figures/NG/Distribution_negative_guidance_lam_{guidance_scale}_prior_{prior}_Temp_{Temp}.png')
    plt.close()
    
    KL_div = KL_with_ground_truth(ground_truth_distr,distr)
    KL_div_uniform = KL_with_uniform(distr)

    mu_fake, cov_fake = np.mean(fake_imgs_features,axis=0),  np.cov(fake_imgs_features.T)
    fid = calculate_frechet_distance(mu_fake, cov_fake, mu_real, cov_real, eps=1e-6)
    fid_filt = calculate_frechet_distance(mu_fake, cov_fake, mu_filt, cov_filt)
    
    print('     - Num_wrong          : ', num_wrong)
    print('     - FID (with guidance): ', fid)
    print('     -       (w.r.t. filt): ', fid_filt)
    print('     - KL-div             : ', KL_div)
    
    if guidance_type == 'negative_prompting':
        column_labels = ['Number of generated samples','Initial seed','Guidance scale', 'Number of forbidden images (%)', 'KL divergence to ground truth', 'KL divergence to uniform','FID (without zeros)','FID (with zeros)']
        row = [args.N_tot,args.seed,args.guidance_scale, round(num_wrong,4), round(KL_div,4), round(KL_div_uniform,4), round(fid_filt,4), round(fid,4)]
    if guidance_type == 'safe_latent_diffusion':
        column_labels = ['Number of generated samples','Initial seed','Guidance scale', 'Threshold','s_s', 'beta_m', 's_m', 'Number of forbidden images (%)', 'KL divergence to ground truth', 'KL divergence to uniform','FID (without zeros)','FID (with zeros)']
        row = [args.N_tot,args.seed,args.guidance_scale, args.threshold, args.s_s, args.beta_m, args.s_m, round(num_wrong,4), round(KL_div,4), round(KL_div_uniform,4), round(fid_filt,4), round(fid,4)]
    if guidance_type == 'negative_guidance':
        column_labels = ['Number of generated samples','Initial seed','Guidance scale', 'Prior', 'Temp', 'Number of forbidden images (%)', 'KL divergence to ground truth', 'KL divergence to uniform','FID (without zeros)','FID (with zeros)']
        row = [args.N_tot,args.seed,args.guidance_scale, args.prior, args.Temp, round(num_wrong,4), round(KL_div,4), round(KL_div_uniform,4), round(fid_filt,4), round(fid,4)]
    
    file_name = f'Results/CSVs/{guidance_type}.csv'
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the column names only if the file is empty
        if csvfile.tell() == 0:
            writer.writerow(column_labels)
        writer.writerow(row)

if __name__ == '__main__':
    main()
