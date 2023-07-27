# fmt: off
from tkinter import Variable
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Distribution, Independent, MixtureSameFamily, MultivariateNormal, Normal
)
from torch.distributions.kumaraswamy import Kumaraswamy
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli, RelaxedBernoulli
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli

from torch.distributions.categorical import Categorical

from deep_traffic_generation.core.abstract import LSR

from deep_traffic_generation.core.datasets import TrafficDataset

SMALL = 1e-16

# fmt:on
class CustomMSF(MixtureSameFamily):
    """MixtureSameFamily with `rsample()` method for reparametrization.

    Args:
        mixture_distribution (Categorical): Manages the probability of
            selecting component. The number of categories must match the
            rightmost batch dimension of the component_distribution.
        component_distribution (Distribution): Define the distribution law
            followed by the components. Right-most batch dimension indexes
            component.
    """

    def rsample(self, sample_shape=torch.Size()):
        """Generates a sample_shape shaped reparameterized sample or
        sample_shape shaped batch of reparameterized samples if the
        distribution parameters are batched.

        Method:

            - Apply `Gumbel Sotmax
              <https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html>`_
              on component weights to get a one-hot tensor;
            - Sample using rsample() from the component distribution;
            - Use the one-hot tensor to select samples.

        .. note::
            The component distribution of the mixture should implements a
            rsample() method.

        .. warning::
            Further studies should be made on this method. It is highly
            possible that this method is not correct.
        """
        assert (
            self.component_distribution.has_rsample
        ), "component_distribution attribute should implement rsample() method"

        weights = self.mixture_distribution._param
        comp = nn.functional.gumbel_softmax(weights, hard=True).unsqueeze(-1)
        samples = self.component_distribution.rsample(sample_shape)
        return (comp * samples).sum(dim=1)


class NormalLSR(LSR):
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__(input_dim, out_dim)

        self.z_loc = nn.Linear(input_dim, out_dim)
        z_log_var_layers = []
        z_log_var_layers.append(nn.Linear(input_dim, out_dim))
        z_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=2.0))
        self.z_log_var = nn.Sequential(*z_log_var_layers)

        self.out_dim = out_dim
        self.dist = Normal

        self.prior_loc = nn.Parameter(
            torch.zeros((1, out_dim)), requires_grad=False
        )
        self.prior_log_var = nn.Parameter(
            torch.zeros((1, out_dim)), requires_grad=False
        )
        self.register_parameter("prior_loc", self.prior_loc)
        self.register_parameter("prior_log_var", self.prior_log_var)

    def forward(self, hidden) -> Distribution:
        loc = self.z_loc(hidden)
        log_var = self.z_log_var(hidden)
        return Independent(self.dist(loc, (log_var / 2).exp()), 1)

    def dist_params(self, p: Independent) -> List[torch.Tensor]:
        return [p.base_dist.loc, p.base_dist.scale]

    def get_posterior(self, dist_params: List[torch.Tensor]) -> Independent:
        return Independent(self.dist(dist_params[0], dist_params[1]), 1)

    def get_prior(self) -> Independent:
        return Independent(
            self.dist(self.prior_loc, (self.prior_log_var / 2).exp()), 1
        )


class VampPriorLSR(LSR):
    """VampPrior Latent Space Regularization. https://arxiv.org/pdf/1705.07120.pdf

    Args:
        original_dim(int): number of features for each trajectory (usually 4)
        original_seq_len(int): sequence length of one trajectory (usually 200)
        input_dim (int): size of each input sample after the encoder NN
        out_dim (int):size of each output sample, dimension of the latent distributions
        encoder (nn.Module) : Neural net used for the encoder
        n_components (int, optional): Number of components in the Gaussian
            Mixture of the VampPrior. Defaults to ``500``.
    """

    def __init__(
        self,
        original_dim: int,
        original_seq_len: int,
        input_dim: int,
        out_dim: int,
        encoder: nn.Module,
        n_components: int,
        encoder_traj2: Optional[nn.Module] = None,
        encoder_delta_t: Optional[nn.Module] = None,
    ):
        super().__init__(input_dim, out_dim)

        self.original_dim = original_dim
        self.seq_len = original_seq_len
        self.encoder = encoder
        self.encoder2 = encoder_traj2
        self.encoder_delta_t = encoder_delta_t
        self.n_components = n_components

        # We don't use customMSF here because we don't need to chose one component of the prior when sampling
        self.dist = MixtureSameFamily
        self.comp = Normal
        self.mix = Categorical

        # Posterior Parameters
        z_loc_layers = []
        z_loc_layers.append(nn.Linear(input_dim, out_dim))
        # z_loc_layers.append(nn.BatchNorm1d(out_dim)) #add
        # z_loc_layers.append(nn.ReLU()) #add
        # z_loc_layers.append(nn.Linear(out_dim, out_dim)) #add
        # z_loc_layers.append(nn.BatchNorm1d(out_dim)) #add
        self.z_loc = nn.Sequential(*z_loc_layers)

        z_log_var_layers = []
        z_log_var_layers.append(nn.Linear(input_dim, out_dim))
        # z_loc_layers.append(nn.BatchNorm1d(out_dim)) #add
        # z_log_var_layers.append(nn.Linear(out_dim, out_dim)) #add
        # z_loc_layers.append(nn.BatchNorm1d(out_dim)) #add
        z_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=6.0))
        self.z_log_var = nn.Sequential(*z_log_var_layers)

        # prior parameters
        # Input to the NN that will produce the pseudo inputs
        self.idle_input = torch.autograd.Variable(
            torch.eye(n_components, n_components), requires_grad=False
        )

        # NN that transform the idle_inputs into the pseudo_inputs that will be transformed
        # by the encoder into the means of the VampPrior
        pseudo_inputs_layers = []
        pseudo_inputs_layers.append(nn.Linear(n_components, n_components))
        # pseudo_inputs_layers.append(nn.BatchNorm1d(n_components)) #add
        pseudo_inputs_layers.append(nn.ReLU())
        # pseudo_inputs_layers.append(nn.Linear(n_components, n_components)) #add
        # pseudo_inputs_layers.append(nn.BatchNorm1d(n_components)) #add
        # pseudo_inputs_layers.append(nn.ReLU()) #add
        
        if  self.encoder2:
            pseudo_inputs_layers.append(
                nn.Linear(
                    n_components,
                    # (2*original_dim * original_seq_len + 1),
                    (2*original_dim * original_seq_len),
                )
            )
        else :
            pseudo_inputs_layers.append(
                nn.Linear(
                    n_components,
                    (original_dim * original_seq_len),
                )
            )
        
        pseudo_inputs_layers.append(nn.Hardtanh(min_val=-1.0, max_val=1.0))
        self.pseudo_inputs_NN = nn.Sequential(*pseudo_inputs_layers)

        # decouple variances of posterior and prior componenents
        prior_log_var_layers = []
        prior_log_var_layers.append(nn.Linear(input_dim, out_dim))
        prior_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=6.0))
        self.prior_log_var_NN = nn.Sequential(*z_log_var_layers)

        # In Vamprior, the weights of the GM are all equal
        # Here they are trained
        self.prior_weights = nn.Parameter(
            torch.ones((1, n_components)), requires_grad=True
        )

        self.register_parameter("prior_weights", self.prior_weights)

    def forward(self, hidden: torch.Tensor) -> Distribution:
        """[summary]

        Args:
            hidden (torch.Tensor): output of encoder

        Returns:
            Distribution: corresponding posterior distribution
        """

        # Calculate the posterior parameters :
        loc = self.z_loc(hidden)
        log_var = self.z_log_var(hidden)
        scales = (log_var / 2).exp()

        # calculate the prior paramters :
        X = self.pseudo_inputs_NN(self.idle_input)
        
        if self.encoder2:
            # delta_t = X[:,0]
            # X1 = X[:,1:(self.original_dim *self.seq_len +1)].view((X.shape[0], self.original_dim, self.seq_len))
            # X2 = X[:,(self.original_dim *self.seq_len +1):].view((X.shape[0], self.original_dim, self.seq_len))
            X1 = X[:,:(self.original_dim *self.seq_len)].view((X.shape[0], self.original_dim, self.seq_len))
            X2 = X[:,(self.original_dim *self.seq_len):].view((X.shape[0], self.original_dim, self.seq_len))
            pseudo_h1 = self.encoder(X1)
            pseudo_h2 = self.encoder2(X2)
            
            # pseudo_h = torch.cat((pseudo_h1, pseudo_h2,torch.unsqueeze(delta_t, 1)), dim = 1)
            pseudo_h = torch.cat((pseudo_h1, pseudo_h2), dim = 1)
            # pseudo_h = self.encoder_delta_t(pseudo_h)

        else :
            X = X.view((X.shape[0], self.original_dim, self.seq_len))
            pseudo_h = self.encoder(X)


        self.prior_means = self.z_loc(pseudo_h)
        # self.prior_log_vars = self.z_log_var(pseudo_h)
        self.prior_log_vars = self.prior_log_var_NN(pseudo_h)

        # return the posterior : a single multivariate normal
        return Independent(self.comp(loc, scales), 1)

    # Only for the posterior distribution
    def dist_params(self, p: MixtureSameFamily) -> Tuple:
        return [p.base_dist.loc, p.base_dist.scale]

    # Is a signle multivariate normal
    def get_posterior(self, dist_params: Tuple) -> Distribution:
        return Independent(self.comp(dist_params[0], dist_params[1]), 1)

    def get_prior(self) -> MixtureSameFamily:
        return self.dist(
            self.mix(logits=self.prior_weights.view(self.n_components)),
            Independent(
                self.comp(
                    self.prior_means,
                    (self.prior_log_vars / 2).exp(),
                ),
                1,
            ),
        )

class factorized_VampPriorLSR(LSR):
    """Factorized VampPrior LSR. We learn a gaussian mixture per dimension of the latent space. 

    Args:
        original_dim(int): number of features for each trajectory (usually 4)
        original_seq_len(int): sequence length of one trajectory (usually 200)
        input_dim (int): size of each input sample after the encoder NN
        out_dim (int):size of each output sample, dimension of the latent distributions
        encoder (nn.Module) : Neural net used for the encoder
        n_components (int, optional): Number of components in the Gaussian
            Mixture of the VampPrior
    """

    def __init__(
        self,
        original_dim: int,
        original_seq_len: int,
        input_dim: int,
        out_dim: int,
        encoder: nn.Module,
        n_components: int,
        encoder_traj2: Optional[nn.Module] = None,
        encoder_delta_t: Optional[nn.Module] = None,
    ):
        super().__init__(input_dim, out_dim)

        self.original_dim = original_dim
        self.seq_len = original_seq_len
        self.encoder = encoder
        self.encoder2 = encoder_traj2
        self.encoder_delta_t = encoder_delta_t
        self.n_components = n_components

        # We don't use customMSF here because we don't need to chose one component of the prior when sampling
        self.dist = MixtureSameFamily
        self.comp = Normal
        self.mix = Categorical

        # Posterior Parameters
        z_loc_layers = []
        z_loc_layers.append(nn.Linear(input_dim, out_dim))
        self.z_loc = nn.Sequential(*z_loc_layers)

        z_log_var_layers = []
        z_log_var_layers.append(nn.Linear(input_dim, out_dim))
        z_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=6.0))
        self.z_log_var = nn.Sequential(*z_log_var_layers)

        # prior parameters
        # Input to the NN that will produce the pseudo inputs
        self.idle_input = torch.autograd.Variable(
            torch.eye(n_components, n_components), requires_grad=False
        )

        # NN that transform the idle_inputs into the pseudo_inputs that will be transformed
        # by the encoder into the means of the VampPrior
        pseudo_inputs_layers = []
        pseudo_inputs_layers.append(nn.Linear(n_components, n_components))
        pseudo_inputs_layers.append(nn.ReLU())
        
        if  self.encoder2:
            pseudo_inputs_layers.append(
                nn.Linear(
                    n_components,
                    (2*original_dim * original_seq_len),
                )
            )
        else :
            pseudo_inputs_layers.append(
                nn.Linear(
                    n_components,
                    (original_dim * original_seq_len),
                )
            )
        
        pseudo_inputs_layers.append(nn.Hardtanh(min_val=-1.0, max_val=1.0))
        self.pseudo_inputs_NN = nn.Sequential(*pseudo_inputs_layers)

        # decouple variances of posterior and prior componenents
        prior_log_var_layers = []
        prior_log_var_layers.append(nn.Linear(input_dim, out_dim))
        prior_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=6.0))
        self.prior_log_var_NN = nn.Sequential(*z_log_var_layers)

        # In Vamprior, the weights of the GM are all equal
        # Here they are trained
        self.prior_weights = nn.Parameter(
            torch.ones((out_dim, n_components)), requires_grad=True
        )

        self.register_parameter("prior_weights", self.prior_weights)

    def forward(self, hidden: torch.Tensor) -> Distribution:
        """[summary]

        Args:
            hidden (torch.Tensor): output of encoder

        Returns:
            Distribution: corresponding posterior distribution
        """

        # Calculate the posterior parameters :
        loc = self.z_loc(hidden)
        log_var = self.z_log_var(hidden)
        scales = (log_var / 2).exp()

        # calculate the prior paramters :
        X = self.pseudo_inputs_NN(self.idle_input)
        
        if self.encoder2:
            X1 = X[:,:(self.original_dim *self.seq_len)].view((X.shape[0], self.original_dim, self.seq_len))
            X2 = X[:,(self.original_dim *self.seq_len):].view((X.shape[0], self.original_dim, self.seq_len))
            pseudo_h1 = self.encoder(X1)
            pseudo_h2 = self.encoder2(X2)
            
            pseudo_h = torch.cat((pseudo_h1, pseudo_h2), dim = 1)

        else :
            X = X.view((X.shape[0], self.original_dim, self.seq_len))
            pseudo_h = self.encoder(X)


        self.prior_means = self.z_loc(pseudo_h)
        self.prior_log_vars = self.prior_log_var_NN(pseudo_h)

        # return the posterior : a single multivariate normal
        return Independent(self.comp(loc, scales), 1)

    # Only for the posterior distribution
    def dist_params(self, p: MixtureSameFamily) -> Tuple:
        return [p.base_dist.loc, p.base_dist.scale]

    # Is a signle multivariate normal
    def get_posterior(self, dist_params: Tuple) -> Distribution:
        return Independent(self.comp(dist_params[0], dist_params[1]), 1)

    def get_prior(self) -> MixtureSameFamily:
        cat = self.mix(logits=self.prior_weights)
        means = self.prior_means.view(self.out_dim, self.n_components, 1)
        sigma = (self.prior_log_vars.view(self.out_dim, self.n_components, 1) / 2).exp()
        dist = self.dist(cat, Independent(self.comp(means, sigma), 1))
        return Independent(dist, 1)


class factorized_GMM_LSR(LSR):
    """Factorized GMM LSR. We learn a gaussian mixture per dimension of the latent space. 

    Args:
        original_dim(int): number of features for each trajectory (usually 4)
        original_seq_len(int): sequence length of one trajectory (usually 200)
        input_dim (int): size of each input sample after the encoder NN
        out_dim (int):size of each output sample, dimension of the latent distributions
        n_components (int, optional): Number of components in the Gaussian
            Mixture
    """

    def __init__(
        self,
        original_dim: int,
        original_seq_len: int,
        input_dim: int,
        out_dim: int,
        n_components: int,
    ):
        super().__init__(input_dim, out_dim)

        self.original_dim = original_dim
        self.seq_len = original_seq_len
        self.n_components = n_components

        # We don't use customMSF here because we don't need to chose one component of the prior when sampling
        self.dist = MixtureSameFamily
        self.comp = Normal
        self.mix = Categorical

        # Posterior Parameters
        z_loc_layers = []
        z_loc_layers.append(nn.Linear(input_dim, out_dim))
        self.z_loc = nn.Sequential(*z_loc_layers)

        z_log_var_layers = []
        z_log_var_layers.append(nn.Linear(input_dim, out_dim))
        z_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=6.0))
        self.z_log_var = nn.Sequential(*z_log_var_layers)

        # prior parameters
        # Input to the NN that will produce the prior parameters
        self.idle_input = torch.autograd.Variable(
            torch.eye(n_components, n_components), requires_grad=False
        )

        # idle_inputs transformed into the prior parameters
        pseudo_inputs_layers = []
        pseudo_inputs_layers.append(nn.Linear(n_components, n_components))
        pseudo_inputs_layers.append(nn.ReLU())
        pseudo_inputs_layers.append(nn.Linear(n_components, input_dim))
        self.pseudo_inputs_NN = nn.Sequential(*pseudo_inputs_layers)
        
        #prior means
        prior_means_layers = []
        prior_means_layers.append(nn.Linear(input_dim, out_dim))
        self.prior_means_NN = nn.Sequential(*prior_means_layers)

        prior_log_var_layers = []
        prior_log_var_layers.append(nn.Linear(input_dim, out_dim))
        prior_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=6.0))
        self.prior_log_var_NN = nn.Sequential(*prior_log_var_layers)

        # Weights of the mixture
        self.prior_weights = nn.Parameter(
            torch.ones((out_dim, n_components)), requires_grad=True
        )

        self.register_parameter("prior_weights", self.prior_weights)

    def forward(self, hidden: torch.Tensor) -> Distribution:
        """[summary]

        Args:
            hidden (torch.Tensor): output of encoder

        Returns:
            Distribution: corresponding posterior distribution
        """

        # Calculate the posterior parameters :
        loc = self.z_loc(hidden)
        log_var = self.z_log_var(hidden)
        scales = (log_var / 2).exp()

        # calculate the prior paramters :
        X = self.pseudo_inputs_NN(self.idle_input)
        self.prior_means = self.prior_means_NN(X)
        self.prior_log_vars = self.prior_log_var_NN(X)

        # return the posterior : a single multivariate normal
        return Independent(self.comp(loc, scales), 1)

    # Only for the posterior distribution
    def dist_params(self, p: MixtureSameFamily) -> Tuple:
        return [p.base_dist.loc, p.base_dist.scale]

    # Is a signle multivariate normal
    def get_posterior(self, dist_params: Tuple) -> Distribution:
        return Independent(self.comp(dist_params[0], dist_params[1]), 1)

    def get_prior(self) -> MixtureSameFamily:
        cat = self.mix(logits=self.prior_weights)
        means = self.prior_means.view(self.out_dim, self.n_components, 1)
        sigma = (self.prior_log_vars.view(self.out_dim, self.n_components, 1) / 2).exp()
        dist = self.dist(cat, Independent(self.comp(means, sigma), 1))
        return Independent(dist, 1)

class IBP_LSR(LSR):
    """Beta-Bernoulli-Process LSR : https://github.com/rachtsingh/ibp_vae

    Args:
        original_dim(int): number of features for each trajectory (usually 4)
        original_seq_len(int): sequence length of one trajectory (usually 200)
        input_dim (int): size of each input sample after the encoder NN
        max_trunc (int):maximum size of the latent space
    """

    def __init__(
        self,
        input_dim: int,
        max_trunc: int,
        alpha0: float,
    ):
        super().__init__(input_dim, max_trunc)
        self.trunc = max_trunc
        self.alpha0 = alpha0
        self.comp = Normal

        # Posterior Parameters: mean of continuous_z
        z_loc_layers = []
        z_loc_layers.append(nn.Linear(input_dim, max_trunc))
        self.z_loc = nn.Sequential(*z_loc_layers)
        
        # Posterior Parameters: logvar of continuous_z
        z_log_var_layers = []
        z_log_var_layers.append(nn.Linear(input_dim, max_trunc))
        z_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=6.0))
        self.z_log_var = nn.Sequential(*z_log_var_layers)
        
        # Posterior Parameters: logit of discrete_z
        z_logit_layers = []
        z_logit_layers.append(nn.Linear(input_dim, max_trunc))
        self.z_logit = nn.Sequential(*z_logit_layers)

        # posterior parameters : (for Kumaraswamy distribution)
        a_val = np.log(np.exp(self.alpha0) - 1) # inverse softplus
        b_val = np.log(np.exp(1.) - 1)
        self.beta_a = nn.Parameter(torch.Tensor(self.trunc).zero_() + a_val, requires_grad=True)
        self.beta_b = nn.Parameter(torch.Tensor(self.trunc).zero_() + b_val, requires_grad=True)

        self.register_parameter("beta_a", self.beta_a)
        self.register_parameter("beta_b", self.beta_b)
    
    # Calculate the pis from the Kumaraswamy distribution
    def reparametrize(self, a: torch.Tensor, b: torch.Tensor, log = False) -> torch.Tensor:
        v = Kumaraswamy(a, b).sample()
        v_term = (v+SMALL).log()
        logpis = torch.cumsum(v_term, dim=1)
        if log:
            return logpis
        else:
            return logpis.exp()
    
    

    def forward(self, hidden: torch.Tensor) -> Tuple[Distribution, Distribution]:
        """[summary]

        Args:
            hidden (torch.Tensor): output of encoder

        Returns:
            Distribution: corresponding posterior distribution
        """
        
        self.batch_size = hidden.shape[0]

        # Calculate the posterior parameters :
        loc = self.z_loc(hidden)
        log_var = self.z_log_var(hidden)
        scales = (log_var / 2).exp()
        
        # Calculation of pi's for the concrete distribution
        #Look at how is calculated the concrete distribution
        logit_x = self.z_logit(hidden)
        beta_a = F.softplus(self.beta_a) + 0.01
        beta_b = F.softplus(self.beta_b) + 0.01
        #logit_post = d(x_n) + logit(pi_n), pi_n = prod(v_n_i), v_n_i ~ Kumaraswamy(a_n_i, b_n_i)
        #Why this ? why log prior here ?
        beta_a_ext = beta_a.view(1, self.trunc).expand(hidden.shape[0], self.trunc)
        beta_b_ext = beta_b.view(1, self.trunc).expand(hidden.shape[0], self.trunc)
        self.log_prior = self.reparametrize(beta_a_ext, beta_b_ext, log=True)
        self.logit_post = logit_x + torch.logit(self.log_prior.exp(), SMALL)

        # Note: The discrete distribution returns the logit : -> we have to apply sigmoid to ge the true samples
        #Warning: Maybe LogitRelaxedBernoulli is not the good distribution for ExpContrete of the paper https://arxiv.org/pdf/1909.01839.pdf
        # return Independent(self.comp(loc, scales), 1), Independent(LogitRelaxedBernoulli(temperature = torch.Tensor([1.]), logits=self.logit_post), 1), Independent(Kumaraswamy(beta_a_ext, beta_b_ext), 1)
        return Independent(self.comp(loc, scales), 1), Independent(RelaxedBernoulli(temperature = torch.Tensor([2.]), logits=self.logit_post), 1), Independent(Kumaraswamy(beta_a_ext, beta_b_ext), 1)

    # Only for the continuous posterior distribution
    def get_params_continuous(self, p: Independent) -> Tuple:
        return [p.base_dist.loc, p.base_dist.scale]
    
    def get_params_discrete(self, p: Independent) -> Tuple:
        return [torch.Tensor([p.base_dist.temperature]), p.base_dist.logits]
    
    def get_params_bernoullis(self) -> Tuple:
        return [self.log_prior.exp(), self.logit_post.exp()]
    
    def get_params_kumaraswamy(self, p: Independent) -> Tuple:
        #concentration1 = beta_a, concentration0 = beta_b
        return [p.base_dist.concentration1, p.base_dist.concentration0]

    # Get kumaraswamy distribution
    def get_kumaraswamy(self, dist_params: Tuple) -> Distribution:
        return Independent(Kumaraswamy(dist_params[0], dist_params[1]), 1)
    
    def get_posterior_continuous(self, dist_params: Tuple) -> Distribution:
        return Independent(self.comp(dist_params[0], dist_params[1]), 1)
    
    def get_posterior_discrete(self, dist_params: Tuple) -> Distribution:
        # return  Independent(LogitRelaxedBernoulli(temperature = dist_params[0], logits=dist_params[1]), 1)
        return  Independent(RelaxedBernoulli(temperature = dist_params[0], logits=dist_params[1]), 1)

    # Not sure about this one though
    def get_prior(self) -> Tuple[Distribution, Distribution, Distribution]:
        continuous_dist = Independent(Normal(torch.zeros(self.trunc), torch.ones(self.trunc)), 1)
        discrete_dist = Bernoulli(self.log_prior.exp()) # In https://arxiv.org/pdf/1909.01839.pdf they use for the training the concrete distribution and not the bernoulli as prior. KL between discrete and continuous is not possible ?
        #discrete approx of the Bernoulli prior
        #why logit prior is taken from the kumaraswamy distrib and not from the beta ? 
        # continuous_discrete_dist = Independent(LogitRelaxedBernoulli(temperature = torch.Tensor([1.]), logits=torch.logit(self.log_prior.exp(), SMALL)), 1) 
        continuous_discrete_dist = Independent(RelaxedBernoulli(temperature = torch.Tensor([2.]), logits=torch.logit(self.log_prior.exp(), SMALL)), 1) 
        
        alpha0_ext = torch.Tensor(self.trunc).zero_() + self.alpha0
        alpha0_ext = alpha0_ext.view(1, self.trunc).expand(self.batch_size, self.trunc)
        beta0_ext = torch.Tensor(self.trunc).zero_() + 1
        beta0_ext = beta0_ext.view(1, self.trunc).expand(self.batch_size, self.trunc)
        beta_dist = Independent(Beta(alpha0_ext, beta0_ext), 1)
        
        return continuous_dist, discrete_dist, continuous_discrete_dist, beta_dist
