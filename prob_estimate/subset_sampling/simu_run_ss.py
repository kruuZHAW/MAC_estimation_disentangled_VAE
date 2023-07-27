""" Script running several simulations of subset sampling
"""

from deep_traffic_generation.tcvae_pairs_disent import TCVAE_Pairs_disent
from deep_traffic_generation.VAE_Generation import PairsVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDatasetPairsRandom
from traffic.core import Traffic

from subset_simulation import subsim

import openturns as ot
from scipy.stats import multivariate_normal, norm
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from tqdm import tqdm

import pickle as pkl
import os
import click
import glob


def load_TCVAE()-> tuple:

    dataset = TrafficDatasetPairsRandom.from_file(
        ("../../deep_traffic_generation/data/training_datasets/to_LSZH_16_50_bb.pkl", "../../deep_traffic_generation/data/training_datasets/ga_LSZH_14_50_bb.pkl"),
        features=["track", "groundspeed", "altitude", "timedelta"],
        n_samples = 10000,
        scaler=MinMaxScaler(feature_range=(-1,1)),
        shape="image",
        info_params={"features": ["latitude", "longitude"], "index": -1},
    )

    path = "../../deep_traffic_generation/lightning_logs/tcvae_pairs_disent/version_22/"
    
    global t, g

    t = PairsVAE(X = dataset, vae_type="TCVAEPairs_disent", sim_type = "generation")
    t.load(path, dataset.parameters)
    g = Generation(generation=t, features = t.VAE.hparams.features, scaler=dataset.scaler)  


def isoProbaTrans() :
    """ 
    Iso probabilistic transformation from N(0,1) to the prior distribution of the VAE 
    """
    
    #VAE prior in pytorch.distribution format
    Z = t.latent_space(0)
    p_z = t.VAE.lsr.get_prior()
    
    #VAE prior in OpenTurns format
    marginals = []
    for i in range(Z.shape[1]):
        collDist = [ot.Normal(mu.item(), sigma.item()) for mu, sigma in zip(p_z.base_dist.component_distribution.base_dist.loc.squeeze(2)[i], p_z.base_dist.component_distribution.base_dist.scale.squeeze(2)[i])]
        weights = p_z.base_dist.mixture_distribution.probs[i].detach().numpy()
        mixt = ot.Mixture(collDist, weights)
        marginals.append(mixt)
    prior = ot.ComposedDistribution(marginals)
    
    global isoTrans
    isoTrans = prior.getInverseIsoProbabilisticTransformation()

def limit_state(z):
    """limit_state is the black box function for the sumbset samling. 

    inputs:
        - z = a latent space vector from N(0,1)
        (t, g, and isoTrans cannot be defined as inputs, but should be calculated before calling limit_state: they are global variables)
        
    outputs:
        - limit_state(X) = closest distance between the two trajectories of the pair
    """
    diam = 55
    
    #From N(0,1) to prior
    z = np.array(z).reshape(1,-1)
    z = isoTrans(ot.Sample(z))
    z = torch.Tensor(z)
    
    #Decode latent representation into a pair of trajectories
    decoded = t.decode(z)
    to = g.build_traffic(decoded[:,:200], coordinates = dict(latitude =  47.44464, longitude = 8.55732), forward=True).iterate_lazy().resample("1s").eval()
    to = to.assign(flight_id=lambda x: x.flight_id + "_to", inplace=True)
    ga = g.build_traffic(decoded[:,200:], coordinates = dict(latitude = 47.500086, longitude = 8.51149), forward=True).iterate_lazy().resample("1s").eval() 
    ga = ga.assign(flight_id=lambda x: x.flight_id + "_ga", inplace=True)
    
    # Calulate distance between the two trajectories
    dist = to[0].distance(ga[0])
    dist["3d_distance"] = dist.apply(lambda x: ((x.lateral*1852)**2 + (x.vertical*0.3048)**2)**0.5 - diam, axis=1) #distance between two spheres in m
    return dist["3d_distance"].min() 

@click.command()
@click.argument("n_sim", type=int)
@click.option('--dim', default=10, type=int)
@click.option('--p0', default=0.2, type=float)
@click.option('--n', default=10000, type=int)
@click.option('--spread', default=0.01, type=float)


def main(
    n_sim:  int,
    dim:  int,
    p0:  float,
    n: int,
    spread: float,
):

    click.echo("Loading VAE...")
    load_TCVAE()
    
    click.echo("Isoprobabilistic transformation...")
    isoProbaTrans()
    
    click.echo("Subset Sampling runs...")
    results = []
    pi = multivariate_normal(mean = np.zeros(dim), cov = np.identity(dim))
    pi_marginal = dim*[norm(loc = 0, scale = 1)]
    for i in tqdm(range(n_sim)):
        res = {}
        theta, g, b, ar, pf = subsim(pi, pi_marginal, limit_state, p0, N = n, spread = spread, g_failure='<=0')
        res["theta"] = theta
        res["g"] = g
        res["b"] = b
        res["ar"] = ar
        res["pf"] = pf
        results.append(pf)
        with open("results/subset_sampling_"+ str(i) +".pkl", 'wb') as f:
            pkl.dump(res, f)
            
    with open("results/sim_estimates", "wb") as fp:
        pkl.dump(results, fp)

if __name__ == "__main__":
    main()