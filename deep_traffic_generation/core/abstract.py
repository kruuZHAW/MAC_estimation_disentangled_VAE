# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, Tuple, Union
from xmlrpc.client import boolean

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.distributions.distribution import Distribution
from torch.distributions import Independent, Normal
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

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """FIXME: too messy."""
        idx = 0
        original = outputs[0][0][idx].unsqueeze(0).cpu()
        reconstructed = outputs[0][1][idx].unsqueeze(0).cpu()
        data = torch.cat((original, reconstructed), dim=0)
        data = data.reshape((data.shape[0], -1))
        # unscale the data
        if self.dataset_params["scaler"] is not None:
            data = self.dataset_params["scaler"].inverse_transform(data)

        if isinstance(data, torch.Tensor):
            data = data.numpy()
        # add info if needed (init_features)
        if len(self.dataset_params["info_params"]["features"]) > 0:
            info = outputs[0][2][idx].unsqueeze(0).cpu().numpy()
            info = np.repeat(info, data.shape[0], axis=0)
            data = np.concatenate((info, data), axis=1)
        # get builder
        builder = self.get_builder(
            nb_samples=2, length=self.dataset_params["seq_len"]
        )
        features = [
            "track" if "track" in f else f for f in self.hparams.features
        ]
        # build traffic
        traffic = traffic_from_data(
            data,
            features,
            self.dataset_params["info_params"]["features"],
            builder=builder,
        )
        # generate plot then send it to logger
        self.logger.experiment.add_figure(
            "original vs reconstructed", plot_traffic(traffic)
        )

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
    to genreate pairs of trajectories + delta_t

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
        "llv_coef_delta_t",
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
        self.encoders_delta_t: nn.Module
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
        #Scale for the decoder of delta_t
        # self.scale_delta_t = nn.Parameter(
        #     torch.Tensor([self.hparams.scale]), requires_grad=True
        # )

        # Latent Space Regularization
        self.lsr: LSR

    # def forward(self, x1, x2, delta_t) -> Tuple[Tuple, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def forward(self, x1, x2) -> Tuple[Tuple, torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.encoder_traj1(x1)
        h2 = self.encoder_traj2(x2)
        
        # h = torch.cat((h1, h2, torch.unsqueeze(delta_t, 1)), dim = 1)
        h = torch.cat((h1, h2), dim = 1)
        # h = torch.cat((h1.unsqueeze(2),h2.unsqueeze(2)), dim = 2)
        # h = nn.MaxPool2d((1, 2), stride=(1,1))(h).squeeze(2)
       
        # h = self.encoder_delta_t(h)
        # q is a collection of normal posterior for each traj of the batch
        q = self.lsr(h)
        z = q.rsample()

        # decode z into reconstructed pair and delta_t
        # x1_hat, x2_hat, delta_t_hat = self.decoder(z)
        x1_hat, x2_hat = self.decoder(z)
        x1_hat = self.out_activ(x1_hat)
        x2_hat = self.out_activ(x2_hat)
        return self.lsr.dist_params(q), z, x1_hat, x2_hat#, delta_t_hat


    def training_step(self, batch, batch_idx):
        """Training step.

        Computes the gaussian likelihood and the Kullback-Leibler divergence
        to get the ELBO loss function.
        A Gaussian liklihood term was added to take into account differently the delta_t

        """
        # x1, x2, delta_t = batch
        # dist_params, z, x1_hat, x2_hat, delta_t_hat = self.forward(x1, x2, delta_t)
        x1, x2 = batch
        dist_params, z, x1_hat, x2_hat = self.forward(x1, x2)

        # log likelihood loss for pairs of trajectories
        llv_loss1 = -self.gaussian_likelihood(x1, x1_hat, self.scale_traj1)
        llv_loss2 = -self.gaussian_likelihood(x2, x2_hat, self.scale_traj2)
        llv_coef = self.hparams.llv_coef

        # log likelihood loss for delta_t
        # llv_loss_delta_t = -self.gaussian_likelihood(delta_t, delta_t_hat, self.scale_delta_t)
        # llv_coef_delta_t = self.hparams.llv_coef_delta_t

        # kullback-leibler divergence / MMD
        q_zx = self.lsr.get_posterior(dist_params)
        p_z = self.lsr.get_prior()
        kld_loss = self.kl_divergence(z, q_zx, p_z)
        # mmd_loss = self.mmd(p_z, z)
        kld_coef = self.hparams.kld_coef

        # elbo with beta hyperparameter:
        #   Higher values enforce orthogonality between latent representation.
        elbo = kld_coef * kld_loss + llv_coef * (llv_loss1 + llv_loss2) #+ llv_coef_delta_t * llv_loss_delta_t
        # elbo = kld_coef * mmd_loss + llv_coef * (llv_loss1 + llv_loss2) #+ llv_coef_delta_t * llv_loss_delta_t
        elbo = elbo.mean()

        self.log_dict(
            {
                "train_loss": elbo,
                "kl_loss": kld_loss.mean(),
                # "mmd_loss": mmd_loss.mean(),
                "recon_loss1": llv_loss1.mean(),
                "recon_loss2": llv_loss2.mean(),
                # "recon_loss_delta_t": llv_loss_delta_t.mean(),
            }
        )
        return elbo

    def validation_step(self, batch, batch_idx):
        # x1, x2, delta_t = batch
        # _, _, x1_hat, x2_hat, delta_t_hat = self.forward(x1, x2, delta_t)
        # loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2) + F.mse_loss(delta_t_hat, delta_t)
        x1, x2 = batch
        _, _, x1_hat, x2_hat = self.forward(x1, x2)
        loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        # x1, x2, delta_t = batch
        # _, _, x1_hat, x2_hat, delta_t_hat = self.forward(x1, x2, delta_t)
        # loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2) + F.mse_loss(delta_t_hat, delta_t)
        x1, x2 = batch
        _, _, x1_hat, x2_hat = self.forward(x1, x2)
        loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
        self.log("hp/test_loss", loss)
        return x1, x1_hat, x2, x2_hat#, delta_t, delta_t_hat

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
    
    def compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)
    
    def mmd(self, p: Distribution, z: torch.Tensor) -> torch.Tensor:
    
        """Maximum Mean Discrepancy (MMD) to replace KLD (according to https://arxiv.org/pdf/1706.02262.pdf)
        Between samples from the prior and sample from the posterior.
        Gaussian kernel is used for the kernel trick.
        Base idea: two distributions are identical iif all their moments are identical
        
        Args: 
            p: prior distribution
            z: A sample from q (the posterior)
            
        """
        bs = z.size(0)
        z_prior = p.sample(torch.Size([bs])).squeeze(1)
        x_kernel = self.compute_kernel(z_prior, z_prior)
        y_kernel = self.compute_kernel(z, z)
        xy_kernel = self.compute_kernel(z_prior, z)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd
        
        

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
            * ``--llv_coef_delta_t``: Coefficient for the gaussian log likelihood
            (reconstruction loss) of delta_t.
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
            "--llv_coef_delta_t",
            dest="llv_coef_delta_t",
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


class HVAE(AE):
    """Abstract class for a 2 layers Hierarchical VAE.
    Adaptation of the code presented in VAE with VampPrior
    PyTorch Implementation
    https://github.com/jmtomczak/vae_vampprior/blob/master/models/HVAE_2level.py
    """

    _required_hparams = AE._required_hparams + [
        "kld_coef",
        "llv_coef",
        "scale",
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

        # Distribution q(z1|x,z2)
        z1_loc_layers = []
        z1_loc_layers.append(
            nn.Linear(self.hparams.encoding_dim, self.hparams.encoding_dim)
        )
        self.z1_loc_NN = nn.Sequential(*z1_loc_layers)

        z1_log_var_layers = []
        z1_log_var_layers.append(
            nn.Linear(self.hparams.encoding_dim, self.hparams.encoding_dim)
        )
        z1_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=2.0))
        self.z1_log_var_NN = nn.Sequential(*z1_log_var_layers)

        # Distribution p(z1|z2)
        p_z1_layers = []
        p_z1_layers.extend(
            [
                nn.Linear(self.hparams.encoding_dim, 300),
                nn.ReLU(),
                nn.Linear(300, 300),
            ]
        )
        # p_z1_layers.append(nn.Sigmoid())
        self.p_z1_NN = nn.Sequential(*p_z1_layers)

        self.p_z1_loc_NN = nn.Linear(300, self.hparams.encoding_dim)

        p_z1_log_var_layers = []
        p_z1_log_var_layers.append(nn.Linear(300, self.hparams.encoding_dim))
        p_z1_log_var_layers.append(nn.Hardtanh(min_val=-6.0, max_val=2.0))
        self.p_z1_log_var_NN = nn.Sequential(*p_z1_log_var_layers)

    # encoding distribution of z_1
    def q_z1(self, h1):

        loc_z1 = self.z1_loc_NN(h1)
        scales_z1 = (self.z1_log_var_NN(h1) / 2).exp()

        return Independent(Normal(loc_z1, scales_z1), 1)

    # decoding distribution of z_1
    def p_z1(self, z2):

        z2 = self.p_z1_NN(z2)
        mean = self.p_z1_loc_NN(z2)
        scales = (self.p_z1_log_var_NN(z2) / 2).exp()

        return mean, scales

    def forward(self, x) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:

        # get z2 : from x to h2 and sampling
        h2 = self.encoder_z2(x)
        q2 = self.lsr(h2)
        z2 = q2.rsample()

        # get z1 : from x and h2 to h1
        h1_x = self.encoder_z1_x(x)
        h1_z2 = self.encoder_z1_z2(z2)
        h1 = torch.cat((h1_x, h1_z2), 1)
        h1 = self.encoder_z1_joint(h1)
        q1 = self.q_z1(h1)
        z1 = q1.rsample()

        # Parameters of P(z1|z2)
        mean_pz1, scales_pz1 = self.p_z1(z2)

        # decode z
        x_hat = self.out_activ(self.decoder(torch.cat((z1, z2), 1)))
        return (
            self.lsr.dist_params(q2),
            [q1.base_dist.loc, q1.base_dist.scale],
            [mean_pz1, scales_pz1],
            z2,
            z1,
            x_hat,
        )

    def training_step(self, batch, batch_idx):
        """Training step.

        Computes the gaussian likelihood and the Kullback-Leibler divergence
        to get the ELBO loss function.
        The KL divergence part of ELBO is slightly different for HVAE

        """
        x, _ = batch
        q_z2_params, q_z1_params, p_z1_params, z2, z1, x_hat = self.forward(x)

        # log likelihood loss (reconstruction loss)
        llv_loss = -self.gaussian_likelihood(x, x_hat)
        llv_coef = self.hparams.llv_coef

        # kullback-leibler divergence (regularization loss)
        q_z2 = self.lsr.get_posterior(q_z2_params)
        p_z2 = self.lsr.get_prior()
        q_z1 = Independent(Normal(q_z1_params[0], q_z1_params[1]), 1)
        p_z1 = Independent(Normal(p_z1_params[0], p_z1_params[1]), 1)
        kld_loss_z2 = self.kl_divergence(z2, q_z2, p_z2)
        kld_loss_z1 = self.kl_divergence(z1, q_z1, p_z1)
        kld_coef = self.hparams.kld_coef

        # elbo with beta hyperparameter:
        # Higher values enforce orthogonality between latent representation.
        elbo = kld_coef * (kld_loss_z1 + kld_loss_z2) + llv_coef * llv_loss
        elbo = elbo.mean()

        self.log_dict(
            {
                "train_loss": elbo,
                "kl_loss": (kld_loss_z1 + kld_loss_z2).mean(),
                "recon_loss": llv_loss.mean(),
            }
        )
        return elbo

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        _, _, _, _, _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, info = batch
        _, _, _, _, _, x_hat = self.forward(x)
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
