# fmt: off
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from base64 import decode
from cmath import tanh
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from deep_traffic_generation.core import TCN, VAEPairs, cli_main
from deep_traffic_generation.core.datasets import DatasetParams, TrafficDatasetPairs, TrafficDataset, TrafficDatasetPairsRandom 
from deep_traffic_generation.core.lsr import GaussianMixtureLSR, VampPriorLSR, NormalLSR, ExemplarLSR


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
        self.input_dim = input_dim

        self.decode_entry1 = nn.Linear(
            input_dim, h_dims[0] * int(seq_len / sampling_factor)
        )
        self.decode_entry2 = nn.Linear(
            input_dim, h_dims[0] * int(seq_len / sampling_factor)
        )

        self.decoder_traj1 = nn.Sequential(
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

        self.decoder_traj2 = nn.Sequential(
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

        self.decode_delta_t = nn.Sequential(
                        nn.Linear(input_dim,input_dim*2),
                        nn.ReLU(),
                        nn.BatchNorm1d(input_dim*2),
                        # nn.Linear(input_dim*2, input_dim*2 + 1),
                        # nn.BatchNorm1d(input_dim*2 + 1),
                        nn.Linear(input_dim*2, input_dim),
                        nn.BatchNorm1d(input_dim),
                        nn.Tanh(),
                        )

    def forward(self, x):
        # delta_t_hat = self.decode_delta_t(x)
        # x = self.decode_delta_t(x)  
        y1 = self.decode_entry1(x)
        y2 = self.decode_entry2(x)
        b, _ = y1.size()
        y1 = y1.view(b, -1, int(self.seq_len / self.sampling_factor))
        y2 = y2.view(b, -1, int(self.seq_len / self.sampling_factor))
        x1_hat = self.decoder_traj1(y1)
        x2_hat = self.decoder_traj2(y2)
        return x1_hat, x2_hat#, delta_t_hat


class TCVAE_Pairs(VAEPairs):
    """Temporal Convolutional Variational Autoencoder for pairs of Trajectories

    Pairs of trajectories and delta_t are encoded within the same latent space through 
    2 different branches

    Usage example:

        .. code-block:: console

            python tcvae_pairs.py --encoding_dim 32 --h_dims 16 16 16 --features \
track groundspeed altitude timedelta --info_features latitude \
longitude
    """

    _required_hparams = VAEPairs._required_hparams + [
        "sampling_factor",
        "kernel_size",
        "dilation_base",
        "n_components",
    ]

    def __init__(
        self,
        dataset_params: DatasetParams,
        config: Union[Dict, Namespace],
    ) -> None:
        super().__init__(dataset_params, config)

        self.example_input_array = tuple((torch.rand(
            (
                1,
                self.dataset_params["input_dim"],
                self.dataset_params["seq_len"],
            )),
            torch.rand(
            (
                1,
                self.dataset_params["input_dim"],
                self.dataset_params["seq_len"],
            ))))
            # torch.rand(1)))

        h_dim = self.hparams.h_dims[-1] * (
            int(self.dataset_params["seq_len"] / self.hparams.sampling_factor)
        )

        #Encode first traj of the pair
        self.encoder_traj1 = nn.Sequential(
            TCN(
                input_dim=self.dataset_params["input_dim"],
                out_dim=self.hparams.h_dims[-1],
                h_dims=self.hparams.h_dims[:-1],
                kernel_size=self.hparams.kernel_size,
                dilation_base=self.hparams.dilation_base,
                dropout=self.hparams.dropout,
                h_activ=nn.ReLU(),
            ),
            nn.AvgPool1d(self.hparams.sampling_factor),
            nn.Flatten(),
            # nn.BatchNorm1d(h_dim),
        )

        #Encode second traj of the pair
        self.encoder_traj2 = nn.Sequential(
            TCN(
                input_dim=self.dataset_params["input_dim"],
                out_dim=self.hparams.h_dims[-1],
                h_dims=self.hparams.h_dims[:-1],
                kernel_size=self.hparams.kernel_size,
                dilation_base=self.hparams.dilation_base,
                dropout=self.hparams.dropout,
                h_activ=nn.ReLU(),
            ),
            nn.AvgPool1d(self.hparams.sampling_factor),
            nn.Flatten(),
            # nn.BatchNorm1d(h_dim),
        )

        #Neural Net that encodes trajs encoded with TCN concatenated with delta_t
        # self.encoder_delta_t = nn.Sequential(
        #                 # nn.Linear(h_dim*2 + 1,h_dim*2),
        #                 nn.Linear(h_dim*2, h_dim*2),
        #                 nn.ReLU(),
        #                 nn.BatchNorm1d(h_dim*2),
        #                 nn.Linear(h_dim*2, h_dim),
        #                 nn.BatchNorm1d(h_dim),
        #                 nn.Tanh(),
        #                 )
        
        # self.encoder_delta_t = nn.Sequential(
        #                 nn.Linear(h_dim*2, h_dim),
        #                 nn.ReLU(),
        #                 nn.BatchNorm1d(h_dim),
        #                 nn.Linear(h_dim, h_dim),
        #                 nn.BatchNorm1d(h_dim),
        #                 # nn.Tanh(),
        #                 )

        #delta_t is also considered for the pseudo_inputs creation
        if self.hparams.prior == "vampprior":
            self.lsr = VampPriorLSR(
                original_dim=self.dataset_params["input_dim"],
                original_seq_len=self.dataset_params["seq_len"],
                input_dim=2*h_dim,
                out_dim=self.hparams.encoding_dim,
                encoder=self.encoder_traj1,
                n_components=self.hparams.n_components,
                encoder_traj2 = self.encoder_traj2,
                # encoder_delta_t=self.encoder_delta_t,
            )

        elif self.hparams.prior == "standard":
            self.lsr = NormalLSR(
                input_dim=h_dim,
                out_dim=self.hparams.encoding_dim,
            )

        elif self.hparams.prior == "exemplar":
            if self.hparams.exemplar_path is None:
                raise Exception(
                    "--exemplar_path have to be specified for --prior exemplar"
                )

            # load trajectories used for prior with the same parameters as the training dataset
            # The scaler is the one trained on the training dataset
            self.prior_trajs = TrafficDataset.from_file(
                self.hparams.exemplar_path,
                features=self.dataset_params["features"],
                shape=self.dataset_params["shape"],
                scaler=self.dataset_params["scaler"],
                info_params=self.dataset_params["info_params"],
            )

            self.lsr = ExemplarLSR(
                original_dim=self.dataset_params["input_dim"],
                original_seq_len=self.dataset_params["seq_len"],
                input_dim=h_dim,
                out_dim=self.hparams.encoding_dim,
                encoder=self.encoder,
                prior_trajs=self.prior_trajs.data,
            )

        else:
            raise Exception("Wrong name of the prior!")

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
        )

        # non-linear activation after decoder
        self.out_activ = nn.Identity()


    def test_step(self, batch, batch_idx):
        # x1, x2, delta_t = batch
        x1, x2 = batch
        # _, _, x1_hat, x2_hat, delta_t_hat = self.forward(x1, x2, delta_t)
        _, _, x1_hat, x2_hat = self.forward(x1, x2)
        # loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2) + F.mse_loss(delta_t_hat, delta_t)
        loss = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
        self.log("hp/test_loss", loss)
        # return torch.transpose(x1, 1, 2), torch.transpose(x1_hat, 1, 2),torch.transpose(x2, 1, 2), torch.transpose(x2_hat, 1, 2), delta_t, delta_t_hat
        return torch.transpose(x1, 1, 2), torch.transpose(x1_hat, 1, 2),torch.transpose(x2, 1, 2), torch.transpose(x2_hat, 1, 2)


    @classmethod
    def network_name(cls) -> str:
        return "tcvae_pairs"

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
            choices=["vampprior", "standard", "exemplar"],
            default="vampprior",
        )

        parser.add_argument(
            "--exemplar_path", dest="exemplar_path", type=Path, default=None
        )

        return parent_parser, parser


if __name__ == "__main__":
    # cli_main(TCVAE_Pairs, TrafficDatasetPairs, "image", seed=42) #for Orly
    cli_main(TCVAE_Pairs, TrafficDatasetPairsRandom, "image", seed=42) #for Zurich
