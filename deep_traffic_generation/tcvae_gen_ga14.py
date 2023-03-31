"""Generated trajectories with a trained TCVAE Go arounds 14 in LSZH.
    
    Arguments:
        dataset_path: path of training dataset of TCVAE
        version: version of trained TCVAE
        name: name of output traffic object
        lat: reference latitude
        lon: reference longitude
        n-gen: Number of generated trajectories
    
    Returns:
        traffic object: generated synthetic trajectories
    """


from deep_traffic_generation.tcvae import TCVAE
from deep_traffic_generation.VAE_Generation import SingleStageVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDataset
from traffic.core import Traffic
from shapely.geometry import LineString
from shapely.ops import nearest_points

from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from os import walk
import click
from pathlib import Path
import time

from torch.distributions import (
    Distribution, Independent, MixtureSameFamily, MultivariateNormal, Normal
)
from torch.distributions.categorical import Categorical

def load_TCVAE(dataset_path : Path, version:  str)-> tuple:

    dataset = TrafficDataset.from_file(
        dataset_path,
        features=["track", "groundspeed", "altitude", "timedelta"],
        scaler=MinMaxScaler(feature_range=(-1,1)),
        shape="image",
        info_params={"features": ["latitude", "longitude"], "index": -1},
    )

    tcvae_path = "../deep_traffic_generation/lightning_logs/tcvae/"+ version + "/"

    t = SingleStageVAE(X = dataset, sim_type = "generation")
    t.load(tcvae_path, dataset.parameters)
    g = Generation(generation=t, features = t.VAE.hparams.features, scaler=dataset.scaler) 

    return t, g

def pseudo_inputs(t: SingleStageVAE, g: Generation, lat:float, lon:float, forward:bool) -> tuple:
    #Vampprior
    pseudo_X = t.VAE.lsr.pseudo_inputs_NN(t.VAE.lsr.idle_input) 
    pseudo_X = pseudo_X.view((pseudo_X.shape[0], 4, 100))

    pseudo_h = t.VAE.encoder(pseudo_X)
    pseudo_means = t.VAE.lsr.z_loc(pseudo_h)
    pseudo_scales = (t.VAE.lsr.z_log_var(pseudo_h) / 2).exp()

    #Reconstructed pseudo-inputs
    out = t.decode(pseudo_means)
    #Neural net don't predict exaclty timedelta = 0 for the first observation
    out[:,3] = 0
    out_traf = g.build_traffic(out,coordinates = dict(latitude = lat, longitude = lon), forward=forward)

    return out_traf, pseudo_means, pseudo_scales

#Based on the analysis of the pseudo-inputs in a notebook
def selecting_PI_14(pi:Traffic)-> Traffic:

    selected_PI = pi.query(
        "flight_id not in ['TRAJ_66', 'TRAJ_131', 'TRAJ_107', 'TRAJ_2', 'TRAJ_75', 'TRAJ_129', 'TRAJ_81', 'TRAJ_118', 'TRAJ_85', 'TRAJ_67', 'TRAJ_4', 'TRAJ_84']"
    )

    id_PI = [int(i.split("_",1)[1]) for i in selected_PI.flight_ids]
    return id_PI

def generate_traffic(selected_pseudo_means:torch.Tensor, selected_pseudo_scales:torch.Tensor, t: SingleStageVAE, g: Generation, lat:float, lon:float, forward:bool, n:int)->Traffic:

    dist_gen = MixtureSameFamily(
        Categorical(torch.ones((len(selected_pseudo_means),))),
            Independent(
                Normal(
                    selected_pseudo_means,
                    selected_pseudo_scales,
                ),
                1,
            ),
    )

    gen_latent = dist_gen.sample(torch.Size([n]))
    decode = t.decode(gen_latent)
    decode[:, 3] = 0

    traf_gen_tcas = g.build_traffic(
    decode,
    coordinates = dict(latitude = lat, longitude = lon),
    forward=forward,
    )

    return traf_gen_tcas


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.argument("version", type=str)
@click.argument("name", type=str)
@click.argument("lat", type=float)
@click.argument("lon", type=float)
@click.option("-n", "--n-gen", default=10000, help="Number of generated trajectories")

def main(
    dataset_path:  Path,
    version:  str,
    name: str,
    lat:  float,
    lon: float,
    n_gen:int,
):

    start_time = time.time()
    click.echo("Loading TCVAE...")
    t, g = load_TCVAE(dataset_path, version)
    click.echo("--- %s seconds ---" % (time.time() - start_time))

    click.echo("Selecting suitable pseudo-inputs...")
    if dataset_path.split("/")[-1] == "ga_LSZH_14_100.pkl": 
        out_traf, pseudo_means, pseudo_scales = pseudo_inputs(t, g, lat, lon, forward = True)
        id_PI = selecting_PI_14(out_traf)
        click.echo("--- %s seconds ---" % (time.time() - start_time))
        click.echo("Generating traffic...")
        gen_traf = generate_traffic(pseudo_means[id_PI], pseudo_scales[id_PI], t, g, lat, lon, True, n_gen)
    else:
        raise ValueError("Selection for those pseudo-inputs hasn't been implemented")
    click.echo("--- %s seconds ---" % (time.time() - start_time))

    
    gen_traf.to_pickle("../deep_traffic_generation/data/generated_datasets/"+ name)
    click.echo("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()