""" Script to calculate Sobol indices for the minimal distance within a pair of aircraft
"""

from deep_traffic_generation.tcvae_pairs_disent import TCVAE_Pairs_disent
from deep_traffic_generation.VAE_Generation import PairsVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDatasetPairsRandom
from traffic.core import Traffic

import openturns as ot
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


def otPrior(t):
    """Translate prior distribution from torch format to OpenTurns format"""
    
    Z = t.latent_space(0)
    p_z = t.VAE.lsr.get_prior()

    marginals = []
    for i in range(Z.shape[1]):
        collDist = [ot.Normal(mu.item(), sigma.item()) for mu, sigma in zip(p_z.base_dist.component_distribution.base_dist.loc.squeeze(2)[i], p_z.base_dist.component_distribution.base_dist.scale.squeeze(2)[i])]
        weights = p_z.base_dist.mixture_distribution.probs[i].detach().numpy()
        mixt = ot.Mixture(collDist, weights)
        marginals.append(mixt)
    prior = ot.ComposedDistribution(marginals)
    return prior

def limit_state(z):
    """limit_state is the black box function for the sumbset samling. 

    inputs:
        - z = a latent space vector from N(0,1)
        (t, g, and isoTrans cannot be defined as inputs, but should be calculated before calling limit_state: they are global variables)
        
    outputs:
        - limit_state(X) = closest distance between the two trajectories of the pair
        - keep inputs if the failure domain
    """
    diam = 55
    
    z = np.array(z).reshape(1,-1)
    z = torch.Tensor(z)
    
    #Decode latent representation into a pair of trajectories
    decoded = t.decode(z)
    to = g.build_traffic(decoded[:,:200], coordinates = dict(latitude =  47.44464, longitude = 8.55732), forward=True).iterate_lazy().resample("1s").eval()
    # to = to.assign(flight_id=lambda x: x.flight_id + "_to", inplace=True)
    ga = g.build_traffic(decoded[:,200:], coordinates = dict(latitude = 47.500086, longitude = 8.51149), forward=True).iterate_lazy().resample("1s").eval() 
    # ga = ga.assign(flight_id=lambda x: x.flight_id + "_ga", inplace=True)
    
    # Calulate distance between the two trajectories
    dist = to[0].distance(ga[0])
    dist["3d_distance"] = dist.apply(lambda x: ((x.lateral*1852)**2 + (x.vertical*0.3048)**2)**0.5 - diam, axis=1) #distance between two spheres in m
    min_dist = dist["3d_distance"].min()
    
    return [min_dist]

@click.command()
@click.argument("n", type=int)
@click.option('--dim', default=10, type=int)

def main(
    n: int,
    dim:  int,
):

    click.echo("Loading VAE...")
    load_TCVAE()
    
    click.echo("OpenTurns prior...")
    prior = otPrior(t)
    
    click.echo("Sobol' Indicies computation...")
        
    #Run Sobol Indices
    inputDesign = ot.SobolIndicesExperiment(prior, n).generate()
    lsf = ot.PythonFunction(dim, 1, limit_state)
    outputDesign = lsf(inputDesign)
    
    #Get Results
    results = {}
    sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, n)
    results["first order"] = np.array(sensitivityAnalysis.getFirstOrderIndices())
    results["ic first order"] = np.array(sensitivityAnalysis.getFirstOrderIndicesInterval())
    results["total order"] = np.array(sensitivityAnalysis.getTotalOrderIndices())
    results["ic total order"] = np.array(sensitivityAnalysis.getTotalOrderIndicesInterval())
    
    with open("results.pkl", 'wb') as f:
        pkl.dump(results, f)

if __name__ == "__main__":
    main()