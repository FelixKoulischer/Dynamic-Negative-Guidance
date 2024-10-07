import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()

    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 128       # batch size per device for training the ET model
    training.numworkers = 0         # number of workers for training the ET model
    training.n_epochs = 800       # number of epochs for training
    training.eval_freq = 10         # epoch interval between two evaluations
    training.snapshot_freq = 50    # number of epochs to sample
    training.distributed = True     # train the model in distributed gpus


    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.generated_batch = 36   # batch size for sampling process

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.checkpoint = 25
    evaluate.batch_size = 250
    evaluate.enable_sampling = True
    evaluate.num_samples = 50000

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'CIFAR10'
    data.image_size = 32
    data.random_flip = True
    data.centered = True
    data.uniform_dequantization = False
    data.num_classes = 10
    data.num_channels = 3

    # model
    config.model = model = ml_collections.ConfigDict()
    model.beta_min = 0.0001
    model.beta_max = 0.020
    model.dropout = 0.1
    model.embedding_type = 'fourier'
    model.denoiser = True

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.multiplier = 2.5
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.weight_decay = 0.
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config

def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.reduce_mean = True

  # sampling
  sampling = config.sampling
  sampling.sampler = 'ancestral_sampling'

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True

  return config