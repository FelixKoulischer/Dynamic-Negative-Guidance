"""
Abstract Class for the forward and reverse process.

    It explains what we need in the diffusion process
"""
import abc
import torch


class DiffusionProcess(abc.ABC):
    """ Diffusion process abstract class. Functions are designed for a mini-batch of inputs """

    def __init__(self, T):
        """ Construct a Discrete Diffusion process.

        Args:
            N: number of time steps
        """
        super().__init__()
        self.N = T

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the DP."""
        pass

    @abc.abstractmethod
    def transition_prob(self, x, t):
        pass

    @abc.abstractmethod
    def forward_step(self, x, t):
        """return sample x_t ~ q(x_t|x_{t-1})"""
        pass

    @abc.abstractmethod
    def t_step_transition_prob(self, x, t):
        """Computes the the t-step forward distribution q(x_t|x_0) """
        pass

    @abc.abstractmethod
    def t_forward_steps(self, x, t):
        """return sample x_t ~ q(x_t|x_{t-1})"""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$."""
        pass

class GaussianDiffusion(DiffusionProcess):
    def __init__(self, beta_min:float=1e-4, beta_max:float=2e-2, rho:int=7,T:int=1000) -> None:
        """Construct a Gaussian diffusion model.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            T: number of timesteps
            rho: power in noise scheduling: ==> '1' for linear, '2' for quadratic and '7' for the one proposed in EDM paper (default)
        """
        super().__init__(T)
        assert beta_max >= beta_min
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.rho = rho
        self.N = T
        self.discrete_betas = self.generate_betas()    # LINEAR NOISING SCHEDULE => POOR!!! => SCHOULD USE EDM NOISING SCHEDULE!!
        self.alphas = 1. - self.discrete_betas                                             # $$ \alpha_t = 1 - \beta_t $$
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)                            # $$ \bar{\alpha}_t = \prod_{i=0}^t \aplha_i $$
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)                         # $$ \sqrt{\bar{\alpha}} $$
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)                 # $$ \sqrt{1 - \bar{\alpha}} $$

    @property
    def T(self):                         # For easy access
        return self.N

    def generate_betas(self):
        t = torch.linspace(0,1,self.N)
        return (self.beta_min**(1/self.rho)+t*(self.beta_max**(1/self.rho)-self.beta_min**(1/self.rho)))**(self.rho) # You just switched max and min
    
    # Transition prob of FORWARD Markov Chain!
    def transition_prob(self, x, t):   
        """Forward step as a result of the forward transition distribution $q(x_t|x_t-1)$
            $$ q(x_t|x_{t-1}) = \mathcal{N}(x_t|\sqrt{1-beta_t}x_{t-1}, \beta_t*I) $$
        Args:
            x: tensor in range [-1,1]  => size [N_batch,C,H,W]
            t: 1D tensor of            => size [N_batch]
        Returns:
            mean: average of forward gassian (rescaled x) => size [N_batch,C,H,W]
            std:  Scale of the "to-be-added" noise        => scalar
        """
        beta = self.discrete_betas.to(x.device)[t]
        mean = torch.sqrt(1 - beta[:, None, None, None].to(x.device)) * x         # Rescale the input: VP
        std = torch.sqrt(beta)                                                    # Std of the "to-be-added"-noise
        return mean, std
   
    # Applies the FORWARD Markov Chain!
    def forward_step(self, x, t):
        """return sample $x_t ~ q(x_t|x_{t-1})$
          $$x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t} * z ; z~N(0,I)$$
        Args:
            x: tensor in range [-1,1]  => size [N_batch,C,H,W]
            t: 1D tensor of            => size [N_batch]
        Returns:
            x:  The "noised" sample    => size [N_batch,C,H,W]
            z:  The added noise        => size [N_batch,C,H,W]
        """
        mean, std = self.transition_prob(x, t)                        # Looks at the probability of forward step
        z = torch.randn_like(x)                                       # The random noise that will be added
        x = mean + std[:, None, None, None]*z                         # Noising the sample! (sqrt(1-beta) already included in the mean)
        return x, z

    # Transition prob of multiple step FORWARD Markov Chain! => for DDIM
    def t_step_transition_prob(self, x, t):
        """ Computes the the t-step forward distribution $q(x_t|x_0)$
            $$ q(x_t|x_0) = \mathcal{N}(x_t; sqrt{\bar{\alpha_t}}x_0, (1-\bar{\alpha_t})I) $$
        Args:
            x: tensor in range [-1,1]  => size [N_batch,C,H,W]
            t: 1D tensor of            => size [N_batch]
        Returns:
            mean: average of forward gassian (rescaled x) => size [N_batch,C,H,W]
            std:  Scale of the "to-be-added" noise        => scalar
        """
        mean = self.sqrt_alphas_cumprod.to(x.device)[t, None, None, None] * x      # Rescale the input: VP => multiple step use \bar{\alpha}
        std  = self.sqrt_1m_alphas_cumprod.to(x.device)[t]                         # Std of the "to-be-added"-noise
        return mean, std

    # Applies the multiple step FORWARD Markov Chain!
    def t_forward_steps(self, x, t):
        """return sample $x_t ~ q(x_t|x_0)$
        Basically reparameterize the t-step distribution
        $$x_t = sqrt{\bar{\alpha_t}}x_0 + \sqrt{(1-\bar{\alpha_t})} * z; z \sim  z~N(0,I)$$
        Args:
            x: tensor in range [-1,1]  => size [N_batch,C,H,W]
            t: 1D tensor of            => size [N_batch]
        Returns:
            x:  The "noised" sample    => size [N_batch,C,H,W]
            z:  The added noise        => size [N_batch,C,H,W]
        """
        mean, std = self.t_step_transition_prob(x, t)                          # Looks at the probability of forward step
        z = torch.randn_like(x)                                                # The random noise that will be added
        x_t = mean + std[:, None, None, None]*z                                # Noising the sample! (sqrt(1-beta) already included in the mean)
        return x_t, z

    # Sample from Gaussian prior!!
    def prior_sampling(self, shape): # maybe instead of SHAPE give x => and just do randn_like(x)
        return torch.randn(*shape)        # Gaussian prior: $$ \mathcal{N}(\bm{x}|0,\bm{I}) $$ 