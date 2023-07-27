# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, Tuple, Union
from numbers import Number
from xmlrpc.client import boolean

import numpy as np
import math
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
# from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.distributions.distribution import Distribution
from torch.distributions import Normal, Independent, MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from traffic.core.projection import EuroPP

from deep_traffic_generation.core.datasets import DatasetParams

from .builders import (
    CollectionBuilder, IdentifierBuilder, LatLonBuilder, TimestampBuilder
)
from .utils import plot_traffic, traffic_from_data


# fmt: on
class LSR(nn.Module):
    """Abstract for Latent Space Regularization Networks.

    Args:
        input_dim (int): size of each input sample.
        out_dim (int): size pf each output sample.
        fix_prior (bool, optional): Whether to optimize the prior distribution.
            Defaults to ``True``.
    """

    def __init__(self, input_dim: int, out_dim: int, fix_prior: bool = False):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.fix_prior = fix_prior

    def forward(self, hidden: torch.Tensor) -> Distribution:
        """Defines the computation performed at every call.

        Returns a Distribution object according to a tensor. Ideally the
        Distribution object implements a rsample() method.

        .. note::
            Should be overriden by all subclasses.
        """
        raise NotImplementedError()

    def dist_params(self, p: Distribution) -> Tuple:
        """Returns a tuple of tensors corresponding to the parameters of
        a given distribution.

        .. note::
            Should be overriden by all subclasses.
        """
        raise NotImplementedError()

    def get_posterior(self, dist_params: Tuple) -> Distribution:
        """Returns a Distribution object according to a tuple of parameters.
        Inverse method of dist_params().

        Args:
            dist_params (Tuple): tuple of tensors corresponding to distribution
                parameters.

        .. note::
            Should be overriden by all subclasses.
        """
        raise NotImplementedError()

    def get_prior(self, batch_size: int) -> Distribution:
        """Returns the prior distribution we want the posterior distribution
        to fit.

        .. note::
            Should be overriden by all subclasses.
        """
        raise NotImplementedError()


class Abstract(LightningModule):
    """Abstract class for deep models."""

    _required_hparams = [
        "lr",
        "lr_step_size",
        "lr_gamma",
        "dropout",
    ]

    def __init__(
        self, dataset_params: DatasetParams, config: Union[Dict, Namespace]
    ) -> None:
        super().__init__()

        self._check_hparams(config)
        self.save_hyperparameters(config)

        self.dataset_params = dataset_params

        # self.criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    @classmethod
    def network_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def add_model_specific_args(
        cls,
        parent_parser: ArgumentParser,
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        parser = parent_parser.add_argument_group(f"{cls.network_name()}")
        parser.add_argument(
            "--name",
            dest="network_name",
            default=f"{cls.network_name()}",
            type=str,
            help="network name",
        )
        parser.add_argument(
            "--lr",
            dest="lr",
            default=1e-3,
            type=float,
            help="learning rate",
        )
        parser.add_argument(
            "--lrstep",
            dest="lr_step_size",
            default=100,
            type=int,
            help="period of learning rate decay (in epochs)",
        )
        parser.add_argument(
            "--lrgamma",
            dest="lr_gamma",
            default=1.0,
            type=float,
            help="multiplicative factor of learning rate decay",
        )
        parser.add_argument(
            "--dropout", dest="dropout", type=float, default=0.0
        )

        return parent_parser, parser

    def get_builder(self, nb_samples: int, length: int) -> CollectionBuilder:
        builder = CollectionBuilder(
            [
                IdentifierBuilder(nb_samples, length),
                TimestampBuilder(),
            ]
        )
        if "track_unwrapped" in self.dataset_params["features"]:
            if self.dataset_params["info_params"]["index"] == 0:
                builder.append(LatLonBuilder(build_from="azgs"))
            elif self.dataset_params["info_params"]["index"] == -1:
                builder.append(LatLonBuilder(build_from="azgs_r"))
        elif "track" in self.dataset_params["features"]:
            if self.dataset_params["info_params"]["index"] == 0:
                builder.append(LatLonBuilder(build_from="azgs"))
            elif self.dataset_params["info_params"]["index"] == -1:
                builder.append(LatLonBuilder(build_from="azgs_r"))
        elif "x" in self.dataset_params["features"]:
            builder.append(LatLonBuilder(build_from="xy", projection=EuroPP()))

        return builder

    def _check_hparams(self, hparams: Union[Dict, Namespace]):
        for hparam in self._required_hparams:
            if isinstance(hparams, Namespace):
                if hparam not in vars(hparams).keys():
                    raise AttributeError(
                        f"Can't set up network, {hparam} is missing."
                    )
            elif isinstance(hparams, dict):
                if hparam not in hparams.keys():
                    raise AttributeError(
                        f"Can't set up network, {hparam} is missing."
                    )
            else:
                raise TypeError(f"Invalid type for hparams: {type(hparams)}.")


class AE(Abstract):
    """Abstract class for Autoencoders.

    Usage Example:
        .. code:: python

            import torch.nn as nn
            from deep_traffic_generation.core import AE

            class YourAE(AE):
                def __init__(self, dataset_params, config):
                    super().__init__(dataset_params, config)

                    # Define encoder
                    self.encoder = nn.Linear(64, 16)

                    # Define decoder
                    self.decoder = nn.Linear(16, 64)
    """

    _required_hparams = Abstract._required_hparams + [
        "encoding_dim",
        "h_dims",
    ]

    def __init__(
        self, dataset_params: DatasetParams, config: Union[Dict, Namespace]
    ) -> None:
        super().__init__(dataset_params, config)

        self.encoder: nn.Module
        self.decoder: nn.Module
        self.out_activ: nn.Module

    def configure_optimizers(self) -> dict:
        """Optimizers."""
        # optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_step_size,
            gamma=self.hparams.lr_gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams, {"hp/valid_loss": 1, "hp/test_loss": 1}
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.out_activ(self.decoder(z))
        return z, x_hat

    def training_step(self, batch, batch_idx):
        """Training step.

        The validation loss is the Mean Square Error
        :math:`\\mathcal{L}_{MSE}(x_{i}, \\hat{x_{i}})`.
        """
        x, _ = batch
        _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.

        The validation loss is the Mean Square Error
        :math:`\\mathcal{L}_{MSE}(x_{i}, \\hat{x_{i}})`.
        """
        x, _ = batch
        _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        """Test step.

        The test loss is the Mean Square Error
        :math:`\\mathcal{L}_{MSE}(x_{i}, \\hat{x_{i}})`.
        """
        x, info = batch
        _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return x, x_hat, info

    # def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
    #     """FIXME: too messy."""
    #     idx = 0
    #     original = outputs[0][0][idx].unsqueeze(0).cpu()
    #     reconstructed = outputs[0][1][idx].unsqueeze(0).cpu()
    #     data = torch.cat((original, reconstructed), dim=0)
    #     data = data.reshape((data.shape[0], -1))
    #     # unscale the data
    #     if self.dataset_params["scaler"] is not None:
    #         data = self.dataset_params["scaler"].inverse_transform(data)

    #     if isinstance(data, torch.Tensor):
    #         data = data.numpy()
    #     # add info if needed (init_features)
    #     if len(self.dataset_params["info_params"]["features"]) > 0:
    #         info = outputs[0][2][idx].unsqueeze(0).cpu().numpy()
    #         info = np.repeat(info, data.shape[0], axis=0)
    #         data = np.concatenate((info, data), axis=1)
    #     # get builder
    #     builder = self.get_builder(
    #         nb_samples=2, length=self.dataset_params["seq_len"]
    #     )
    #     features = [
    #         "track" if "track" in f else f for f in self.hparams.features
    #     ]
    #     # build traffic
    #     traffic = traffic_from_data(
    #         data,
    #         features,
    #         self.dataset_params["info_params"]["features"],
    #         builder=builder,
    #     )
    #     # generate plot then send it to logger
    #     self.logger.experiment.add_figure(
    #         "original vs reconstructed", plot_traffic(traffic)
    #     )

    @classmethod
    def add_model_specific_args(
        cls,
        parent_parser: ArgumentParser,
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds AE arguments to ArgumentParser.

        List of arguments:

            * ``--encoding_dim``: Latent space size. Default to :math:`32`.
            * ``--h_dims``: List of dimensions for hidden layers. Default to
              ``[]``.

        Args:
            parent_parser (ArgumentParser): ArgumentParser to update.

        Returns:
            Tuple[ArgumentParser, _ArgumentGroup]: updated ArgumentParser with
            TrafficDataset arguments and _ArgumentGroup corresponding to the
            network.
        """
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--encoding_dim",
            dest="encoding_dim",
            type=int,
            default=32,
        )
        parser.add_argument(
            "--h_dims",
            dest="h_dims",
            nargs="+",
            type=int,
            default=[],
        )

        return parent_parser, parser


class VAE(AE):
    """Abstract class for Variational Autoencoder. Adaptation of the VAE
    presented by William Falcon in `Variational Autoencoder Demystified With
    PyTorch Implementation
    <https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed>`_.

    Usage Example:
        .. code:: python

            import torch.nn as nn
            from deep_traffic_generation.core import NormalLSR, VAE

            class YourVAE(VAE):
                def __init__(self, dataset_params, config):
                    super().__init__(dataset_parms, config)

                    # Define encoder
                    self.encoder = nn.Linear(64, 32)

                    # Example of latent space regularization
                    self.lsr = NormalLSR(
                        input_dim=32,
                        out_dim=16
                    )

                    # Define decoder
                    self.decoder = nn.Sequential(
                        nn.Linear(16, 32),
                        nn.ReLU(),
                        nn.Linear(32, 64)
                    )
    """

    _required_hparams = AE._required_hparams + [
        "kld_coef",
        "llv_coef",
        "scale",
        "fix_prior",
    ]

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        # Auto balancing between kld and llv with the decoder scale
        # Diagnosing and Enhancing VAE Models
        self.scale = nn.Parameter(
            torch.Tensor([self.hparams.scale]), requires_grad=True
        )

        # Latent Space Regularization
        self.lsr: LSR

    def forward(self, x) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        # encode x to get the location and log variance parameters
        h = self.encoder(x)
        # When batched, q is a collection of normal posterior
        q = self.lsr(h)
        z = q.rsample()
        # decode z
        x_hat = self.out_activ(self.decoder(z))
        return self.lsr.dist_params(q), z, x_hat

    def training_step(self, batch, batch_idx):
        """Training step.

        Computes the gaussian likelihood and the Kullback-Leibler divergence
        to get the ELBO loss function.

        .. math::

            \\mathcal{L}_{ELBO} = \\alpha \\times KL(q(z|x) || p(z))
            - \\beta \\times \\sum_{i=0}^{N} log(p(x_{i}|z_{i}))
        """
        x, _ = batch
        dist_params, z, x_hat = self.forward(x)

        # Regular VAE LOSS
        # log likelihood loss (reconstruction loss)
        llv_loss = -self.gaussian_likelihood(x, x_hat)
        llv_coef = self.hparams.llv_coef
        # kullback-leibler divergence (regularization loss)
        q_zx = self.lsr.get_posterior(dist_params)
        p_z = self.lsr.get_prior()
        kld_loss = self.kl_divergence(z, q_zx, p_z)
        kld_coef = self.hparams.kld_coef

        # elbo with beta hyperparameter:
        #   Higher values enforce orthogonality between latent representation.
        elbo = kld_coef * kld_loss + llv_coef * llv_loss
        elbo = elbo.mean()

        self.log_dict(
            {
                "train_loss": elbo,
                "kl_loss": kld_loss.mean(),
                "recon_loss": llv_loss.mean(),
            }
        )
        return elbo

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        _, _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, info = batch
        _, _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return x, x_hat, info

    def gaussian_likelihood(self, x: torch.Tensor, x_hat: torch.Tensor):
        """Computes the gaussian likelihood.

        Args:
            x (torch.Tensor): input data
            x_hat (torch.Tensor): mean decoded from :math:`z`.

        .. math::

            \\sum_{i=0}^{N} log(p(x_{i}|z_{i}))
            \\text{ with } p(.|z_{i})
            \\sim \\mathcal{N}(\\hat{x_{i}},\\,\\sigma^{2})

        .. note::
            The scale :math:`\\sigma` can be defined in config and will be
            accessible with ``self.scale``.
        """
        mean = x_hat
        dist = torch.distributions.Normal(mean, self.scale)
        # measure prob of seeing trajectory under p(x|z)
        log_pxz = dist.log_prob(x)
        dims = [i for i in range(1, len(x.size()))]
        return log_pxz.sum(dim=dims)

    def kl_divergence(
        self, z: torch.Tensor, p: Distribution, q: Distribution
    ) -> torch.Tensor:
        """Computes Kullback-Leibler divergence :math:`KL(p || q)` between two
        distributions, using Monte Carlo Sampling.
        Evaluate every z of the batch in its corresponding posterior (1st z with 1st post, etc..)
        and every z in the prior

        Args:
            z (torch.Tensor): A sample from p (the posterior).
            p (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the posetrior)
            q (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the prior)

        Returns:
            torch.Tensor: A batch of KL divergences of shape `z.size(0)`.

        .. note::
            Make sure that the `log_prob()` method of both Distribution
            objects returns a 1D-tensor with the size of `z` batch size.
        """
        log_p = p.log_prob(z)
        log_q = q.log_prob(z)
        return log_p - log_q

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds VAE arguments to ArgumentParser.

        List of arguments:

            * ``--llv_coef``: Coefficient for the gaussian log likelihood
              (reconstruction loss): :math:`\\beta`.
            * ``--kld_coef``: Coefficient for the Kullback-Leibler divergence
              (regularization loss): :math:`\\alpha`.
            * ``--scale``: Define the scale :math:`\\sigma` of the Normal law
              used to sample the reconstruction.
            * `fix-prior`: Whether the prior is learnable or not. Default to
              ``False``.

                * ``--fix-prior``: is not learnable;
                * ``--no-fix-prior``: is learnable.

        .. note::
            It adds also the argument of the inherited class `AE`.

        Args:
            parent_parser (ArgumentParser): ArgumentParser to update.

        Returns:
            Tuple[ArgumentParser, _ArgumentGroup]: updated ArgumentParser with
            TrafficDataset arguments and _ArgumentGroup corresponding to the
            network.
        """
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--llv_coef",
            dest="llv_coef",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--kld_coef", dest="kld_coef", type=float, default=1.0
        )

        parser.add_argument("--scale", dest="scale", type=float, default=1.0)
        parser.add_argument(
            "--fix-prior", dest="fix_prior", action="store_true"
        )
        parser.add_argument(
            "--no-fix-prior", dest="fix_prior", action="store_false"
        )
        parser.set_defaults(fix_prior=True)

        return parent_parser, parser

class VAEPairs(AE):
    """Abstract class for Variational Autoencoder. Adaptation of the VAE 
    to genreate pairs of trajectories
    """

    _required_hparams = AE._required_hparams + [
        "kld_coef",
        "llv_coef",
        "scale"
    ]

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        self.encoder_traj1: nn.Module
        self.encoder_traj2: nn.Module
        self.decoder: nn.Module

        # Auto balancing between kld and llv with the decoder scale
        # Diagnosing and Enhancing VAE Models
        #Scales for the decoders
        self.scale_traj1 = nn.Parameter(
            torch.Tensor([self.hparams.scale]), requires_grad=True
        )
        self.scale_traj2 = nn.Parameter(
            torch.Tensor([self.hparams.scale]), requires_grad=True
        )

        # Latent Space Regularization
        self.lsr: LSR

    def forward(self, x1, x2) -> Tuple[Tuple, torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.encoder_traj1(x1)
        h2 = self.encoder_traj2(x2)
        
        h = torch.cat((h1, h2), dim = 1)
        # h = torch.cat((h1.unsqueeze(2),h2.unsqueeze(2)), dim = 2)
        # h = nn.MaxPool2d((1, 2), stride=(1,1))(h).squeeze(2)
       
        # q is a collection of normal posterior for each traj of the batch
        q = self.lsr(h)
        z = q.rsample()

        # decode z into reconstructed pair and delta_t
        x1_hat, x2_hat = self.decoder(z)
        x1_hat = self.out_activ(x1_hat)
        x2_hat = self.out_activ(x2_hat)
        return self.lsr.dist_params(q), z, x1_hat, x2_hat


    def training_step(self, batch, batch_idx):
        """Training step.

        Computes the gaussian likelihood and the Kullback-Leibler divergence
        to get the ELBO loss function.

        """
        
        x1, x2 = batch
        dist_params, z, x1_hat, x2_hat = self.forward(x1, x2)

        # log likelihood loss for pairs of trajectories
        llv_loss1 = -self.gaussian_likelihood(x1, x1_hat, self.scale_traj1)
        llv_loss2 = -self.gaussian_likelihood(x2, x2_hat, self.scale_traj2)
        llv_coef = self.hparams.llv_coef

        # kullback-leibler divergence
        q_zx = self.lsr.get_posterior(dist_params)
        p_z = self.lsr.get_prior()
        kld_loss = self.kl_divergence(z, q_zx, p_z)
        kld_coef = self.hparams.kld_coef

        # elbo with beta hyperparameter:
        #   Higher values enforce orthogonality between latent representation.
        elbo = kld_coef * kld_loss + llv_coef * (llv_loss1 + llv_loss2)
        elbo = elbo.mean()

        self.log_dict(
            {
                "train_loss": elbo,
                "kl_loss": kld_loss.mean(),
                "recon_loss1": llv_loss1.mean(),
                "recon_loss2": llv_loss2.mean(),
            }
        )
        return elbo

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch
        _, _, x1_hat, x2_hat = self.forward(x1, x2)
        loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x1, x2 = batch
        _, _, x1_hat, x2_hat = self.forward(x1, x2)
        loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
        self.log("hp/test_loss", loss)
        return x1, x1_hat, x2, x2_hat

    def gaussian_likelihood(self, x: torch.Tensor, x_hat: torch.Tensor, scale):
        """Computes the gaussian likelihood.

        Args:
            x (torch.Tensor): input data
            x_hat (torch.Tensor): mean decoded from :math:`z`.

        .. math::

            \\sum_{i=0}^{N} log(p(x_{i}|z_{i}))
            \\text{ with } p(.|z_{i})
            \\sim \\mathcal{N}(\\hat{x_{i}},\\,\\sigma^{2})

        .. note::
            The scale :math:`\\sigma` can be defined in config and will be
            accessible with ``self.scale``.
        """
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)
        # measure prob of seeing trajectory under p(x|z)
        log_pxz = dist.log_prob(x)
        dims = [i for i in range(1, len(x.size()))]
        return log_pxz.sum(dim=dims)

    def kl_divergence(
        self, z: torch.Tensor, p: Distribution, q: Distribution
    ) -> torch.Tensor:
        """Computes Kullback-Leibler divergence :math:`KL(p || q)` between two
        distributions, using Monte Carlo Sampling.
        Evaluate every z of the batch in its corresponding posterior (1st z with 1st post, etc..)
        and every z in the prior

        Args:
            z (torch.Tensor): A sample from p (the posterior).
            p (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the posetrior)
            q (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the prior)

        Returns:
            torch.Tensor: A batch of KL divergences of shape `z.size(0)`.

        .. note::
            Make sure that the `log_prob()` method of both Distribution
            objects returns a 1D-tensor with the size of `z` batch size.
        """
        log_p = p.log_prob(z)
        log_q = q.log_prob(z)
        return log_p - log_q
        
        

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds VAE arguments to ArgumentParser.

        List of arguments:

            * ``--llv_coef``: Coefficient for the gaussian log likelihood
              (reconstruction loss): :math:`\\beta`.
            * ``--kld_coef``: Coefficient for the Kullback-Leibler divergence
              (regularization loss): :math:`\\alpha`.
            * ``--scale``: Define the scale :math:`\\sigma` of the Normal law
              used to sample the reconstruction.

        .. note::
            It adds also the argument of the inherited class `AE`.

        Args:
            parent_parser (ArgumentParser): ArgumentParser to update.

        Returns:
            Tuple[ArgumentParser, _ArgumentGroup]: updated ArgumentParser with
            TrafficDataset arguments and _ArgumentGroup corresponding to the
            network.
        """
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--llv_coef",
            dest="llv_coef",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--kld_coef", dest="kld_coef", type=float, default=1.0
        )

        parser.add_argument(
            "--reg_pseudo", dest="reg_pseudo", type=boolean, default=False
        )

        parser.add_argument("--scale", dest="scale", type=float, default=1.0)

        return parent_parser, parser
    
class VAEPairs_disent(AE):
    """Abstract class for Variational Autoencoder. Adaptation of the VAE 
    to genreate pairs of trajectories in a disentangled manner.
    inspired by: https://arxiv.org/abs/1802.04942
    
    Works only for a factorised gaussian posterior and a factorised prior
    """

    _required_hparams = AE._required_hparams + [
        "llv_coef",
        "tc_coef",
        "kld_coef",
        # "post_coef",
        # "prior_coef",
        "scale"
    ]

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        self.encoder_traj1: nn.Module
        self.encoder_traj2: nn.Module
        self.decoder: nn.Module

        # Auto balancing between kld and llv with the decoder scale
        # Diagnosing and Enhancing VAE Models
        # Scales for the decoders
        self.scale_traj1 = nn.Parameter(
            torch.Tensor([self.hparams.scale]), requires_grad=True
        )
        self.scale_traj2 = nn.Parameter(
            torch.Tensor([self.hparams.scale]), requires_grad=True
        )

        # Latent Space Regularization
        self.lsr: LSR

    def forward(self, x1, x2) -> Tuple[Tuple, torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.encoder_traj1(x1)
        h2 = self.encoder_traj2(x2)
        h = torch.cat((h1, h2), dim = 1)
        # h = torch.cat((h1.unsqueeze(2),h2.unsqueeze(2)), dim = 2)
        # h = nn.MaxPool2d((1, 2), stride=(1,1))(h).squeeze(2)
    
        # q is a collection of normal posterior for each traj of the batch
        q = self.lsr(h)
        z = q.rsample()

        # decode z into reconstructed pair
        x1_hat, x2_hat = self.decoder(z)
        x1_hat = self.out_activ(x1_hat)
        x2_hat = self.out_activ(x2_hat)
        return self.lsr.dist_params(q), z, x1_hat, x2_hat


    def training_step(self, batch, batch_idx):
        """Training step.

        Computes the ELBO according to the decomposition in https://arxiv.org/abs/1802.04942

        """
        
        x1, x2 = batch
        dist_params, z, x1_hat, x2_hat = self.forward(x1, x2)
        
        # prior
        pz = self.lsr.get_prior()
        if len(pz.event_shape) > 1:
            logpz = pz.log_prob(z.unsqueeze(2))
        else:
            logpz = pz.log_prob(z)

        # log likelihood loss for pairs of trajectories
        llv_loss1 = -self.gaussian_likelihood(x1, x1_hat, self.scale_traj1)
        llv_loss2 = -self.gaussian_likelihood(x2, x2_hat, self.scale_traj2)
        llv_coef = self.hparams.llv_coef
        
        # Calculation of log q(z_i|x_i) (the x_i used to generate z_i)
        q_zx = self.lsr.get_posterior(dist_params)
        logq_zx = q_zx.log_prob(z) #size (n_batch,)
        
        # kullback-leibler divergence  in the regular elbo
        q_zx = self.lsr.get_posterior(dist_params)
        kld_loss = self.kl_divergence(z, q_zx, pz)
        
        # Calculation of log q(z|x_i) for i = 1, ..., n_batch with z fixed
        # logq_zx corresponds to the diagonal of logq_zx_extended
        logqz_extended = q_zx.log_prob(z.view(-1, 1, z.shape[1])) # size (n_batch, n_batch)
        
        # Estimation of q(z) and product of the marginals of q(z) with minibatch weighted sampling: logq(z) ~= - log(MN) + logsumexp_m(q(z|x_m))
        marginals = Normal(dist_params[0], dist_params[1]).log_prob(z.view(-1, 1, z.shape[1])) # size (n_batch, n_batch, latent_dim)
        logqz_prodmarginals = (self.logsumexp(marginals, dim = 1) - math.log(z.shape[0] * self.dataset_params["n_samples"])).sum(1) 
        logqz = self.logsumexp(logqz_extended, dim = 1) - math.log(z.shape[0] * self.dataset_params["n_samples"])
        
        # # We introduce a regularization on the prior to force it to be factorized (case when the prior is VampPrior)
        # # The marginals of a Gaussian mixture with factorized components are the mixture of the marginals
        if self.hparams.post_coef > 0 or self.hparams.prior_coef > 0:
            pz_cat_probs = Categorical(probs = pz.mixture_distribution.probs.repeat(z.shape[1],1))
            pz_mu = pz.component_distribution.base_dist.loc.T.unsqueeze(2)
            pz_sigma = pz.component_distribution.base_dist.scale.T.unsqueeze(2)
            pz_marginals_comp = Independent(Normal(pz_mu, pz_sigma), 1)
            pz_marginals = MixtureSameFamily(pz_cat_probs, pz_marginals_comp)
            logpz_prodmarginals = self.logsumexp(pz_marginals.log_prob(z.unsqueeze(2)), dim = 1)
        
        # Mutual information loss
        mi_loss = (logq_zx - logqz)
        
        # Total Correlation loss
        tc_loss = (logqz - logqz_prodmarginals)
        tc_coef = self.hparams.tc_coef
        
        # Element-wise KL divergence loss
        kld_ew_loss = (logqz_prodmarginals - logpz) 
        # #If pz is not factorized: problem. We would prefer to use logpz_prodmarginals if we keep the "dimension-wise kl" characteristic
        # kld_ew_loss = (logqz_prodmarginals - logpz_prodmarginals)
        kld_coef = self.hparams.kld_coef
        
        # elbo
        elbo = kld_loss + llv_coef * (llv_loss1 + llv_loss2)
        elbo = elbo.detach().mean()
        modified_elbo = llv_coef * (llv_loss1 + llv_loss2) + mi_loss + tc_coef * tc_loss + kld_coef * kld_ew_loss 
        
        # Aggregated posterior regularization: https://arxiv.org/abs/1812.02833
        if self.hparams.post_coef > 0:
            post_loss = (logqz - logpz)
            post_coef = self.hparams.post_coef
            modified_elbo = modified_elbo + post_coef * post_loss
        else:
            post_loss = torch.zeros_like(logqz).detach()
            post_coef = 0
        
        # Factorized prior regularization (force VampPrior to be factorized)
        if self.hparams.prior_coef > 0:
            prior_loss = -(logpz - logpz_prodmarginals)
            prior_coef = self.hparams.prior_coef 
            modified_elbo = modified_elbo + prior_coef * prior_loss
        else:
            prior_loss = torch.zeros_like(logpz).detach()
            prior_coef = 0

        modified_elbo = modified_elbo.mean()

        self.log_dict(
            {
                "train_loss": modified_elbo,
                "elbo": elbo,
                "recon_loss1": llv_loss1.mean(),
                "recon_loss2": llv_loss2.mean(),
                "mutual_information": mi_loss.mean(),
                "total_correlation": tc_loss.mean(),
                "element_wise_kl_divergence": kld_ew_loss.mean(),
                "error_est_kld": torch.abs(kld_loss.mean() - (mi_loss + tc_loss + kld_ew_loss).mean()),
                "posterior_reg": post_loss.mean(),
                "prior_reg": prior_loss.mean(),
            }
        )
        return modified_elbo

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch
        _, _, x1_hat, x2_hat = self.forward(x1, x2)
        loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x1, x2 = batch
        _, _, x1_hat, x2_hat = self.forward(x1, x2)
        loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
        self.log("hp/test_loss", loss)
        return x1, x1_hat, x2, x2_hat
    
    def logsumexp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0),
                                        dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            if isinstance(sum_exp, Number):
                return m + math.log(sum_exp)
            else:
                return m + torch.log(sum_exp)

    def gaussian_likelihood(self, x: torch.Tensor, x_hat: torch.Tensor, scale):
        """Computes the gaussian likelihood.

        Args:
            x (torch.Tensor): input data
            x_hat (torch.Tensor): mean decoded from :math:`z`.

        .. math::

            \\sum_{i=0}^{N} log(p(x_{i}|z_{i}))
            \\text{ with } p(.|z_{i})
            \\sim \\mathcal{N}(\\hat{x_{i}},\\,\\sigma^{2})

        .. note::
            The scale :math:`\\sigma` can be defined in config and will be
            accessible with ``self.scale``.
        """
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)
        # measure prob of seeing trajectory under p(x|z)
        log_pxz = dist.log_prob(x)
        dims = [i for i in range(1, len(x.size()))]
        return log_pxz.sum(dim=dims)

    def kl_divergence(
        self, z: torch.Tensor, p: Distribution, q: Distribution
    ) -> torch.Tensor:
        """Computes Kullback-Leibler divergence :math:`KL(p || q)` between two
        distributions, using Monte Carlo Sampling.
        Evaluate every z of the batch in its corresponding posterior (1st z with 1st post, etc..)
        and every z in the prior

        Args:
            z (torch.Tensor): A sample from p (the posterior).
            p (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the posetrior)
            q (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the prior)

        Returns:
            torch.Tensor: A batch of KL divergences of shape `z.size(0)`.

        .. note::
            Make sure that the `log_prob()` method of both Distribution
            objects returns a 1D-tensor with the size of `z` batch size.
        """
        if len(p.event_shape) > 1:
            log_p = p.log_prob(z.unsqueeze(2))
        else:
            log_p = p.log_prob(z)
        if len(q.event_shape) > 1:
            log_q = q.log_prob(z.unsqueeze(2))
        else:
            log_q = q.log_prob(z)
        return log_p - log_q
        
        

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds VAE arguments to ArgumentParser.

        List of arguments:

            * ``--llv_coef``: Coefficient for the gaussian log likelihood
            * ``--kld_coef``: Coefficient for the dimension-wise Kullback-Leibler divergence
            * ``--tc_coef``: Coefficient for the total correlation
            * ``--post_coef``: Coefficient for the divergence between the aggregated posterior and the prior
            * ``--prior_coef``: Coefficient for the total correlation of the prior
            * ``--scale``: Define the scale :math:`\\sigma` of the Normal law
            used to sample the reconstruction.

        .. note::
            It adds also the argument of the inherited class `AE`.

        Args:
            parent_parser (ArgumentParser): ArgumentParser to update.

        Returns:
            Tuple[ArgumentParser, _ArgumentGroup]: updated ArgumentParser with
            TrafficDataset arguments and _ArgumentGroup corresponding to the
            network.
        """
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--llv_coef",
            dest="llv_coef",
            type=float,
            default=1.0,
        )
        
        parser.add_argument(
            "--kld_coef", dest="kld_coef", type=float, default=1.0
        )
        
        parser.add_argument(
            "--tc_coef", dest="tc_coef", type=float, default=1.0
        )
        
        parser.add_argument(
            "--post_coef", dest="post_coef", type=float, default=0.0
        )
        
        parser.add_argument(
            "--prior_coef", dest="prior_coef", type=float, default=0.0
        )

        parser.add_argument("--scale", dest="scale", type=float, default=1.0)

        return parent_parser, parser
    
class IBP_VAE(AE):
    """Abstract class for the Beta-Bernoulli Process Variationa Autoencoder.
    Adaptation of the VAE to genreate pairs of trajectories in a disentangled manner.
    inspired by: http://approximateinference.org/2017/accepted/SinghEtAl2017.pdf
    """

    _required_hparams = AE._required_hparams + [
        "llv_coef",
        "scale"
    ]

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        self.encoder_traj1: nn.Module
        self.encoder_traj2: nn.Module
        self.decoder: nn.Module

        # Scales for the decoders
        self.scale_traj1 = nn.Parameter(
            torch.Tensor([self.hparams.scale]), requires_grad=True
        )
        self.scale_traj2 = nn.Parameter(
            torch.Tensor([self.hparams.scale]), requires_grad=True
        )

        # Latent Space Regularization
        self.lsr: LSR

    def forward(self, x1, x2) -> Tuple[Tuple, Tuple, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.encoder_traj1(x1)
        h2 = self.encoder_traj2(x2)
        h = torch.cat((h1, h2), dim = 1)
    
        # q is a collection of normal posterior for each traj of the batch
        q_continuous, q_discrete, q_kumaraswamy = self.lsr(h)
        # z_discrete = F.sigmoid(q_discrete.rsample()) #q_discrete samples logits, we have to apply the inverse: sigmoid 
        z_discrete = q_discrete.rsample()
        z_continuous = q_continuous.rsample()
        z_kumaraswamy = q_kumaraswamy.rsample() #Sampled used to calculte discrete distrib is in lsr.py. Here that's to calculate the KL divergence
        z = z_discrete * z_continuous

        # decode z into reconstructed pair
        x1_hat, x2_hat = self.decoder(z)
        x1_hat = self.out_activ(x1_hat)
        x2_hat = self.out_activ(x2_hat)
        return self.lsr.get_params_continuous(q_continuous), self.lsr.get_params_discrete(q_discrete), self.lsr.get_params_kumaraswamy(q_kumaraswamy), z_continuous, z_discrete, z_kumaraswamy, x1_hat, x2_hat


    def training_step(self, batch, batch_idx):
        """Training step.

        Computes the ELBO according to the decomposition in https://arxiv.org/abs/1909.01839

        """
        
        x1, x2 = batch
        dist_params_continous, dist_params_discrete, dist_params_kumaraswamy, z_continuous, z_discrete, z_kumaraswamy, x1_hat, x2_hat = self.forward(x1, x2)
        
        # priors distributions
        p_continuous, _, p_continuous_discrete, p_beta = self.lsr.get_prior()

        # Ici il faut réécrire ELBO pour la faire coller à IBP 
        
        # log likelihood loss for pairs of trajectories
        llv_loss1 = -self.gaussian_likelihood(x1, x1_hat, self.scale_traj1)
        llv_loss2 = -self.gaussian_likelihood(x2, x2_hat, self.scale_traj2)
        llv_coef = self.hparams.llv_coef
        
        # kullback-leibler divergence
        q_continous = self.lsr.get_posterior_continuous(dist_params_continous)
        q_discrete = self.lsr.get_posterior_discrete(dist_params_discrete)
        q_kumaraswamy = self.lsr.get_kumaraswamy(dist_params_kumaraswamy)
        kld_continuous = self.kl_divergence(z_continuous, q_continous, p_continuous)
        # kld_beta = self.kl_divergence(z_kumaraswamy, q_kumaraswamy, p_beta)
        kld_discrete = self.kl_divergence(z_discrete, q_discrete, p_continuous_discrete)
        
        #The KL divergence for discrete distribution : https://github.com/rachtsingh/ibp_vae/blob/master/src/training/common.py
        # dist_params_discrete_prior = self.lsr.get_params_discrete(p_continuous_discrete)
        # kld_discrete = self.kl_discrete(dist_params_discrete[1], dist_params_discrete_prior[1], torch.logit(z_discrete), dist_params_discrete[0], dist_params_discrete_prior[0])

        # Adapter ELBO for IBP
        # elbo = kld_continuous + kld_discrete + kld_beta + llv_coef * (llv_loss1 + llv_loss2)
        elbo = kld_continuous + kld_discrete + llv_coef * (llv_loss1 + llv_loss2)
        elbo = elbo.mean()
        
        print("")
        print("elbo: ", elbo)
        print("reco_loss: ", llv_loss1.mean()+ llv_loss2.mean())
        print("kld_continuous: ", kld_continuous.mean())
        print("kld_discrete: ", kld_discrete.mean())
        # print("kld_beta: ", kld_beta.mean())
        print("")

        self.log_dict(
            {
                "train_loss": elbo,
                "recon_loss1": llv_loss1.mean(),
                "recon_loss2": llv_loss2.mean(),
                "kld_continuous": kld_continuous.mean(),
                "kld_discrete": kld_discrete.mean(),
                # "kld_beta": kld_beta.mean(),
            }
        )
        return elbo

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch
        _, _, _, _, _, _, x1_hat, x2_hat = self.forward(x1, x2)
        loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x1, x2 = batch
        _, _, _, _, _, _, x1_hat, x2_hat = self.forward(x1, x2)
        loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
        self.log("hp/test_loss", loss)
        return x1, x1_hat, x2, x2_hat
    
    #The KL between two exp concrete is the same as the KL between two concrete: https://arxiv.org/pdf/1611.00712.pdf
    # We can calculate the KL on the exp concrete
    def log_density_expconcrete(self, logalphas, logsample, temp):
        """Log-density of the expconcrete according to https://arxiv.org/pdf/1611.00712.pdf
        Log alpha is a logit"""
        exp_term = logalphas + logsample.mul(-temp)
        log_prob = exp_term + np.log(temp) - 2. * F.softplus(exp_term)
        return log_prob
    
    def kl_discrete(self, logit_post, logit_prior, logsample, temp_post, temp_prior):
        """KL between two discrete distributions"""
        log_post = self.log_density_expconcrete(logit_post, logsample, temp_post)
        log_prior = self.log_density_expconcrete(logit_prior, logsample, temp_prior)
        return (log_post - log_prior).sum(-1)
    
    def gaussian_likelihood(self, x: torch.Tensor, x_hat: torch.Tensor, scale):
        """Computes the gaussian likelihood.

        Args:
            x (torch.Tensor): input data
            x_hat (torch.Tensor): mean decoded from :math:`z`.

        .. math::

            \\sum_{i=0}^{N} log(p(x_{i}|z_{i}))
            \\text{ with } p(.|z_{i})
            \\sim \\mathcal{N}(\\hat{x_{i}},\\,\\sigma^{2})

        .. note::
            The scale :math:`\\sigma` can be defined in config and will be
            accessible with ``self.scale``.
        """
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)
        # measure prob of seeing trajectory under p(x|z)
        log_pxz = dist.log_prob(x)
        dims = [i for i in range(1, len(x.size()))]
        return log_pxz.sum(dim=dims)

    def kl_divergence(
        self, z: torch.Tensor, p: Distribution, q: Distribution
    ) -> torch.Tensor:
        """Computes Kullback-Leibler divergence :math:`KL(p || q)` between two
        distributions, using Monte Carlo Sampling.
        Evaluate every z of the batch in its corresponding posterior (1st z with 1st post, etc..)
        and every z in the prior

        Args:
            z (torch.Tensor): A sample from p (the posterior).
            p (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the posetrior)
            q (Distribution): A :class:`~torch.distributions.Distribution`
                object. (the prior)

        Returns:
            torch.Tensor: A batch of KL divergences of shape `z.size(0)`.

        .. note::
            Make sure that the `log_prob()` method of both Distribution
            objects returns a 1D-tensor with the size of `z` batch size.
        """
        log_p = p.log_prob(z)
        log_q = q.log_prob(z)
        return log_p - log_q
        
        

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds VAE arguments to ArgumentParser.

        List of arguments:

            * ``--llv_coef``: Coefficient for the gaussian log likelihood
            * ``--kld_coef``: Coefficient for the dimension-wise Kullback-Leibler divergence
            * ``--tc_coef``: Coefficient for the total correlation
            * ``--post_coef``: Coefficient for the divergence between the aggregated posterior and the prior
            * ``--prior_coef``: Coefficient for the total correlation of the prior
            * ``--scale``: Define the scale :math:`\\sigma` of the Normal law
            used to sample the reconstruction.

        .. note::
            It adds also the argument of the inherited class `AE`.

        Args:
            parent_parser (ArgumentParser): ArgumentParser to update.

        Returns:
            Tuple[ArgumentParser, _ArgumentGroup]: updated ArgumentParser with
            TrafficDataset arguments and _ArgumentGroup corresponding to the
            network.
        """
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--llv_coef",
            dest="llv_coef",
            type=float,
            default=1.0,
        )

        parser.add_argument("--scale", dest="scale", type=float, default=1.0)

        return parent_parser, parser