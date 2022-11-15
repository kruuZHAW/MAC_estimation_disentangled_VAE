# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from base64 import decode
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from os import walk

import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler

from deep_traffic_generation.core import TCN, VAE, meta_cli_main
from deep_traffic_generation.core.datasets import MetaDatasetPairs, TrafficDataset
from deep_traffic_generation.tcvae import TCVAE
from deep_traffic_generation.core.lsr import VampPriorLSR, NormalLSR


# fmt: on
class TCDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        h_dims: List[int],
        seq_len: int,
        kernel_size: int,
        dilation_base: int,
        sampling_factor: int,
        h_activ: Optional[nn.Module] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.sampling_factor = sampling_factor

        self.decode_entry = nn.Linear(
            input_dim, h_dims[0] * int(seq_len / sampling_factor)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=sampling_factor),
            TCN(
                h_dims[0],
                out_dim,
                h_dims[1:],
                kernel_size,
                dilation_base,
                h_activ,
                dropout,
            ),
        )

    def forward(self, x):
        x = self.decode_entry(x)
        b, _ = x.size()
        x = x.view(b, -1, int(self.seq_len / self.sampling_factor))
        x_hat = self.decoder(x)
        return x_hat


class Meta_TCVAE(VAE):
    """Meta Temporal Convolutional Variational Autoencoder
    
    Usage example:
        A good model was achieved using this command to train it:

        .. code-block:: console

            python tcvae.py --encoding_dim 32 --h_dims 16 16 16 --features \
track groundspeed altitude timedelta --info_features latitude \
longitude --info_index -1
    """

    _required_hparams = VAE._required_hparams + [
        "sampling_factor",
        "kernel_size",
        "dilation_base",
        "n_components",
    ]

    def __init__(
        self,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(config)

        self.example_input_array = torch.rand(
            (
                1,
                self.dataset_params["input_dim"],
                self.dataset_params["seq_len"],
            )
        )

        self.encoder = nn.Sequential(
            TCN(
                input_dim=self.dataset_params["input_dim"],
                out_dim=self.hparams.h_dims[-1],
                h_dims=self.hparams.h_dims[:-1],
                kernel_size=self.hparams.kernel_size,
                dilation_base=self.hparams.dilation_base,
                dropout=self.hparams.dropout,
                h_activ=nn.ReLU(),
                # h_activ=None,
            ),
            nn.AvgPool1d(self.hparams.sampling_factor),
            nn.Flatten(),
        )

        h_dim = self.hparams.h_dims[-1] * (
            int(self.dataset_params["seq_len"] / self.hparams.sampling_factor)
        )

        if self.hparams.prior == "vampprior":
            self.lsr = VampPriorLSR(
                original_dim=self.dataset_params["input_dim"],
                original_seq_len=self.dataset_params["seq_len"],
                input_dim=h_dim,
                out_dim=self.hparams.encoding_dim,
                encoder=self.encoder,
                n_components=self.hparams.n_components,
            )

        elif self.hparams.prior == "standard":
            self.lsr = NormalLSR(
                input_dim=h_dim,
                out_dim=self.hparams.encoding_dim,
            )

        else:
            raise Exception("Wrong name of the prior for this VAE!")

        self.decoder = TCDecoder(
            input_dim=self.hparams.encoding_dim,
            out_dim=self.dataset_params["input_dim"],
            h_dims=self.hparams.h_dims[::-1],
            seq_len=self.dataset_params["seq_len"],
            kernel_size=self.hparams.kernel_size,
            dilation_base=self.hparams.dilation_base,
            sampling_factor=self.hparams.sampling_factor,
            dropout=self.hparams.dropout,
            h_activ=nn.ReLU(),
            # h_activ=None,
        )

        # non-linear activation after decoder
        self.out_activ = nn.Identity()
        # self.out_activ = nn.Tanh()

    def test_step(self, batch, batch_idx):
        x, info = batch
        _, _, x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("hp/test_loss", loss)
        return torch.transpose(x, 1, 2), torch.transpose(x_hat, 1, 2), info

    @classmethod
    def network_name(cls) -> str:
        return "meta_tcvae"

    @classmethod
    def add_model_specific_args(
        cls, parent_parser: ArgumentParser
    ) -> Tuple[ArgumentParser, _ArgumentGroup]:
        """Adds TCVAE arguments to ArgumentParser.

        List of arguments:

            * ``--dilation``: Dilation base. Default to :math:`2`.
            * ``--kernel``: Size of the kernel to use in Temporal Convolutional
              layers. Default to :math:`16`.
            * ``--prior``: choice of the prior (standard or vampprior). Default to
            "vampprior".
            * ``--n_components``: Number of components for the Gaussian Mixture
              modelling the prior. Default to :math:`300`.
            * ``--sampling_factor``: Sampling factor to reduce the sequence
              length after Temporal Convolutional layers. Default to
              :math:`10`.

        .. note::
            It adds also the argument of the inherited class `VAE`.

        Args:
            parent_parser (ArgumentParser): ArgumentParser to update.

        Returns:
            Tuple[ArgumentParser, _ArgumentGroup]: updated ArgumentParser with
            TrafficDataset arguments and _ArgumentGroup corresponding to the
            network.
        """
        _, parser = super().add_model_specific_args(parent_parser)
        parser.add_argument(
            "--sampling_factor",
            dest="sampling_factor",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--kernel",
            dest="kernel_size",
            type=int,
            default=16,
        )
        parser.add_argument(
            "--dilation",
            dest="dilation_base",
            type=int,
            default=2,
        )
        parser.add_argument(
            "--n_components", dest="n_components", type=int, default=500
        )

        parser.add_argument(
            "--prior",
            dest="prior",
            choices=["vampprior", "standard"],
            default="vampprior",
        )
        
        #Add argument for VAE path
        parser.add_argument(
            "--vae_paths",
            dest="vae_paths",
            type=Path,
            nargs=2,
            default=None,
        )
        
        parser.add_argument(
            "--data_path",
            dest="data_path",
            type=Path,
            nargs=2,
            default=None,
        )
        
        return parent_parser, parser


if __name__ == "__main__":
    #Load datasets1 and datasets2 here
    #Load VAE1 and VAE2 here
    #Pass dataset1, dataset2, VAE1, VAE2 as arguments of meta_cli_main
    #Cannot use arg parser here: Essayer en utilisant hparams
    
    print(Meta_TCVAE.hparams.prior) #h_params ne marche pas en dehors de la classe parce que non initialisée
    
    #Warning: Call of dataset1, dataset2, VAE1 and VAE2 is hardcoded
    dataset_VAE1 = TrafficDataset.from_file(
    Meta_TCVAE.hparams.data_path[0],
    features=["track", "groundspeed", "altitude", "timedelta"],
    scaler=MinMaxScaler(feature_range=(-1,1)),
    shape="image",
    info_params={"features": ["latitude", "longitude"], "index": -1},
    )

    dataset_VAE2 = TrafficDataset.from_file(
        Meta_TCVAE.hparams.data_path[0],
        features=["track", "groundspeed", "altitude", "timedelta"],
        scaler=MinMaxScaler(feature_range=(-1,1)),
        shape="image",
        info_params={"features": ["latitude", "longitude"], "index": -1},
    )
    
    filenames1 = next(walk(Meta_TCVAE.hparams.vae_paths[0]+ "checkpoints/"), (None, None, []))[2]
    VAE1 = TCVAE.load_from_checkpoint(
        Meta_TCVAE.hparams.vae_paths[0] + "checkpoints/" + filenames1[0],
        dataset_params=dataset_VAE1.parameters,
    )
    VAE1.eval()

    filenames2 = next(walk(Meta_TCVAE.hparams.vae_paths[1]+ "checkpoints/"), (None, None, []))[2]
    VAE2 = TCVAE.load_from_checkpoint(
        Meta_TCVAE.hparams.vae_paths[1] + "checkpoints/" + filenames2[0],
        dataset_params=dataset_VAE2.parameters,
    )
    VAE2.eval()
    
    meta_cli_main(Meta_TCVAE, MetaDatasetPairs, dataset_VAE1, dataset_VAE2, VAE1, VAE2, "image", seed=42)
