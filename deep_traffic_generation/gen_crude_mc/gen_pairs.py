"""Generated traffic of pairs take-off/go-around

Returns:
    traffic object 
"""

from deep_traffic_generation.tcvae_pairs_disent import TCVAE_Pairs_disent
from deep_traffic_generation.VAE_Generation import PairsVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDatasetPairsRandom
from traffic.core import Traffic

from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import numpy as np
from os import walk

from multiprocessing import Pool
import os
from pathlib import Path
import time
import click
import glob

def load_TCVAE()-> tuple:

    dataset = TrafficDatasetPairsRandom.from_file(
        ("../data/training_datasets/to_LSZH_16_50_bb.pkl", "../data/training_datasets/ga_LSZH_14_50_bb.pkl"),
        features=["track", "groundspeed", "altitude", "timedelta"],
        n_samples = 10000,
        scaler=MinMaxScaler(feature_range=(-1,1)),
        shape="image",
        info_params={"features": ["latitude", "longitude"], "index": -1},
    )

    path = "../lightning_logs/tcvae_pairs_disent/version_22/"

    t = PairsVAE(X = dataset, vae_type="TCVAEPairs_disent", sim_type = "generation")
    t.load(path, dataset.parameters)
    g = Generation(generation=t, features = t.VAE.hparams.features, scaler=dataset.scaler)  

    return t, g 

def gen_pairs(t: PairsVAE, g: Generation, n_gen: int) ->Traffic:
    
    Z = t.latent_space(0)
    p_z = t.VAE.lsr.get_prior()
    z_gen = p_z.sample(torch.Size([n_gen])).squeeze(1)
    if len(z_gen.shape) == 3:
        z_gen = z_gen.squeeze(2)
    gen = t.decode(z_gen)

    gen_traf_to = g.build_traffic(gen[:,:200], coordinates = dict(latitude =  47.44464, longitude = 8.55732), forward=True)
    gen_traf_to = gen_traf_to.assign(flight_id=lambda x: x.flight_id + "_to", inplace=True)
    gen_traf_to = gen_traf_to.resample("1s").eval()

    gen_traf_ga = g.build_traffic(gen[:,200:], coordinates = dict(latitude = 47.500086, longitude = 8.51149), forward=True)
    gen_traf_ga = gen_traf_ga.assign(flight_id=lambda x: x.flight_id + "_ga", inplace=True)
    gen_traf_ga = gen_traf_ga.resample("1s").eval()
    
    return gen_traf_to, gen_traf_ga

def reassign_timestamp(iter):
    t, ga = iter
    nb = int(ga.flight_id.split("_")[1])
    
    diff = ga.start - t.start

    ga.data = ga.data.assign(
        timestamp = ga.data.timestamp - diff + pd.to_timedelta(nb*15, unit="Min")  
        )
    
    t.data = t.data.assign(
        timestamp = t.data.timestamp + pd.to_timedelta(nb*15, unit="Min")  
        )
    
    return t+ga



@click.command()
@click.argument("n_gen", type=int)


def main(
    n_gen: int,
):

    start_time = time.time()
    click.echo("Loading VAE...")
    t, g = load_TCVAE()
    click.echo("--- %s seconds ---" % (time.time() - start_time))
    
    click.echo("Generation of trajectories...")
    gen_traf_to, gen_traf_ga = gen_pairs(t, g, n_gen)
    click.echo("--- %s seconds ---" % (time.time() - start_time))

    # click.echo("Adjusting timestamps...")
    # with Pool(processes=os.cpu_count()) as p: 
    #     result = p.map(reassign_timestamp, zip(gen_traf_to, gen_traf_ga))
    #     p.close()
    #     p.join()
    # results = sum(result)   
    # click.echo("--- %s seconds ---" % (time.time() - start_time))
    
    click.echo("Saving results...")
    gen_traf = gen_traf_to + gen_traf_ga
    gen_traf.to_parquet("GATO_MC_" + ".parquet")
    click.echo("--- %s seconds ---" % (time.time() - start_time))

    # click.echo("Saving results...")
    # num = len(glob.glob1(os.getcwd(),"*.parquet"))
    # results = results.assign(flight_id = lambda df: df.flight_id + "_MC_"+ str(num)) #Unique flight_ids over different MC traffics
    # results.to_parquet("GATO_MC_" + str(num) + ".parquet")
    # click.echo("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()