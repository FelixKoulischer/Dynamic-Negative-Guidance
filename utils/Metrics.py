import os
import io
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torchvision
from torchvision import transforms

from scipy import linalg
from models.Inception import InceptionV3

# CIFAR 10 labels, class names
cifar10_classes = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
# Inverting the dictionary
cifar10_numbers = {v: k for k, v in cifar10_classes.items()}

# Use classifier to obtain the generated distribution
def obtain_generated_distribution(label_list:torch.Tensor,denoised_x:torch.Tensor, classifier_pipeline)->list:
    distr = []
    y = []
    for i in range(0,denoised_x.size(0)):
        img = transforms.ToPILImage()(denoised_x[i])
        result = classifier_pipeline(img)
        prediction = result[0]['label']
        y.append(cifar10_numbers[prediction])
    y = torch.tensor(y)
    for i, label in enumerate(label_list):
        indices = torch.where(y==label)[0]
        if indices.size() != torch.Size([]):
            distr.append([i,indices.size()[0]/denoised_x.size()[0]])
        else:
            distr.append([i,1/denoised_x.size()[0]])
    return torch.tensor(distr)

### First metric => KL-Div with target distribution        ###
def KL_with_ground_truth(ground_truth_distr: torch.Tensor,obtained_distr:torch.Tensor, precision: int = 4) -> float:
    N_tot = ground_truth_distr.size()[0]
    KL_div_terms = torch.zeros(N_tot)
    for i in range(0,N_tot):
        if ground_truth_distr[i,1] == 0:
            KL_div_terms[i] = 0
        else:
            KL_div_terms[i] = ground_truth_distr[i,1]*torch.log(ground_truth_distr[i,1]/obtained_distr[i,1])
    return round(KL_div_terms.sum().item(),precision)

###             => KL-Div with target  uniform distritbution ###
def KL_with_uniform(obtained_distr:torch.Tensor, precision: int = 4) -> float:
    N_tot = obtained_distr.size()[0]
    ground_truth_distr = 1/N_tot*torch.ones((10,2))
    ground_truth_distr[:,0] = torch.linspace(0,N_tot-1,N_tot)
    KL_div_terms = torch.zeros(N_tot)
    for i in range(0,N_tot):
        if ground_truth_distr[i,1] == 0:
            KL_div_terms[i] = 0
        else:
            KL_div_terms[i] = ground_truth_distr[i,1]*torch.log(ground_truth_distr[i,1]/obtained_distr[i,1])
    return round(KL_div_terms.sum().item(),precision)

### Second Mwtric: FID ###
def get_images_stats(features, device):
    """
    Compute stats for a dataset.
    Args:
        train_ds: training dataset  Input tensor of shape BxCxHxW. Values are expected to be in range (0, 1)
        model: the inception model
        device: computation device (cpu or cuda)
    Returns:
        mu :  The sample mean over activations for the entire dataset.
        cov : The covariance matrix over activations for the entire dataset.
    """
    activations = np.array([])
    with torch.no_grad():
        batch = images.to(device)
        activations = model(batch)[0].squeeze(3).squeeze(2)

    # Compute statistical measures (mean and covariance) of the activations.
    mu, cov = np.mean(activations,axis=0), np.cov(activations.T)
    return mu, cov
def extract_features(images,model, device):
    """
    Compute stats for a dataset.
    Args:
        train_ds: training dataset  Input tensor of shape BxCxHxW. Values are expected to be in range (0, 1)
        model: the inception model
        device: computation device (cpu or cuda)
    Returns:
        mu :  The sample mean over activations for the entire dataset.
        cov : The covariance matrix over activations for the entire dataset.
    """
    activations = np.array([])
    with torch.no_grad():
        batch = images.to(device)
        activations = model(batch)[0].squeeze(3).squeeze(2)
    return activations



#### NOTE: Below adapted from
# https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        log_and_print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
        #    raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def mse_from_forbidden(denoised_x, all_zeros_for_Hop,device):
    # Read of sizes of tensors
    N_batch, N_pats = denoised_x.size()[0], all_zeros_for_Hop.size()[0]
    C, H, W = denoised_x.size()[1], denoised_x.size()[2], denoised_x.size()[3]
    
    # Randomly generated input tensors for demonstration
    generated_images = denoised_x.to(device)
    forbidden_zeros = all_zeros_for_Hop.reshape(N_pats, C, H, W).to(device)    # Tensor of size (M, C, H, W)
    
    # Expand tensors to allow broadcasting and compute MSE
    # Expand batch_tensors to (N_batch, M, C, H, W)
    generated_images_expanded = generated_images.unsqueeze(1).expand(-1, 4, -1, -1, -1)
    # Expand reference_tensors to (N_batch, M, C, H, W)
    forbidden_zeros = forbidden_zeros.unsqueeze(0).expand(len(Gen_forbidden), -1, -1, -1, -1)
    
    # Compute the squared differences
    squared_diff = (generated_images_expanded - forbidden_zeros) ** 2
    
    # Sum over the last three dimensions (C, H, W)
    sum_squared_diff = torch.sum(squared_diff, dim=(2, 3, 4))
    
    # Compute the mean squared error
    mse = sum_squared_diff / (args.C * args.H * args.W)

    #Closest mse
    closest_mse = torch.min(mse,dim=1).values
    
    return closest_mse

def CosineSim_from_forbidden(denoised_x: torch.Tensor, Hopfield_patterns: torch.Tensor,device) -> torch.Tensor:
    # Read sizes of tensors
    N_batch, C, H, W = denoised_x.size()
    N_pats = Hopfield_patterns.size(0)
    
    # Flatten the tensors
    batch_flat = denoised_x.view(N_train, -1)
    Hopfield_patterns_flat = Hopfield_patterns.view(N_pats, -1)
    
    # Normalize the tensors to unit vectors
    batch_normalized = torch.nn.functional.normalize(batch_flat, p=2, dim=1)
    Hopfield_patterns_normalized = torch.nn.functional.normalize(Hopfield_patterns_flat, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.matmul(batch_normalized, Hopfield_patterns_normalized.t())
    cosine_sim_excluded = torch.where(cosine_sim == 1.0, 0.0, cosine_sim)
    
    # We want to find the closest (most similar) pattern, i.e., the highest cosine similarity
    closest_cosine_sim, _ = torch.max(cosine_sim_excluded, dim=1)
    
    return closest_cosine_sim
    
def Closest_MSE_from_training_data(training_data: torch.Tensor, Hopfield_patterns: torch.Tensor) -> torch.Tensor:
    # Read sizes of tensors
    N_train, C, H, W = training_data.size()
    N_pats = Hopfield_patterns.size(0)
    
    # Expand training_data tensor to (N_train, N_pats, C, H, W)
    training_data_expanded = training_data.unsqueeze(1).expand(-1, N_pats, -1, -1, -1)
    # Expand Hopfield patterns to (N_train, N_pats, C, H, W)
    Hopfield_patterns_expanded = Hopfield_patterns.unsqueeze(0).expand(N_train, -1, -1, -1, -1)
    # Compute the squared differences
    squared_diff = (training_data_expanded - Hopfield_patterns_expanded) ** 2
    
    # Sum over the last three dimensions (C, H, W)
    sum_squared_diff = torch.sum(squared_diff, dim=(2, 3, 4))
    
    # Compute the mean squared error
    mse = sum_squared_diff / (C * H * W)
    
    # Remove the pattern itself => remove zeros from mse! (equivalently compute second closest pattern => closest patternn is pattern itself!)
    mse_exclude_pats = torch.where(mse == 0.0,1.0,mse)
    
    #Closest mse
    closest_mse = torch.min(mse_exclude_pats,dim=0).values
    avg_mse = torch.mean(mse,dim=0)*N_pats/(N_pats-1) # Extra factor to compensate for the extra zero present in the sum
    
    return closest_mse, avg_mse

def Closest_CosineSim_from_training_data(training_data: torch.Tensor, Hopfield_patterns: torch.Tensor) -> torch.Tensor:
    # Read sizes of tensors
    N_train, C, H, W = training_data.size()
    N_pats = Hopfield_patterns.size(0)
    
    # Flatten the tensors
    training_data_flat = training_data.view(N_train, -1)
    Hopfield_patterns_flat = Hopfield_patterns.view(N_pats, -1)
    
    # Normalize the tensors to unit vectors
    training_data_normalized = torch.nn.functional.normalize(training_data_flat, p=2, dim=1)
    Hopfield_patterns_normalized = torch.nn.functional.normalize(Hopfield_patterns_flat, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.matmul(training_data_normalized, Hopfield_patterns_normalized.t())
    cosine_sim_excluded = torch.where(cosine_sim == 1.0, 0.0, cosine_sim)
    
    # We want to find the closest (most similar) pattern, i.e., the highest cosine similarity
    closest_cosine_sim, _ = torch.max(cosine_sim_excluded, dim=1)
    
    # Find the average cosine similarity
    avg_cosine_sim = (torch.mean(cosine_sim, dim=1)-1/N_pats)*N_pats/(N_pats-1)
    
    return closest_cosine_sim, avg_cosine_sim
