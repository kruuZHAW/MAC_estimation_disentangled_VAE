# %%
from deep_traffic_generation.tcvae_pairs_disent import TCVAE_Pairs_disent
from deep_traffic_generation.VAE_Generation import PairsVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDatasetPairsRandom
from traffic.core import Traffic

import openturns as ot
from sklearn.preprocessing import MinMaxScaler

import torch
import numpy as np
import pickle as pkl

# %%
dataset = TrafficDatasetPairsRandom.from_file(
    ("../deep_traffic_generation/data/training_datasets/to_LSZH_16_50_bb.pkl", "../deep_traffic_generation/data/training_datasets/ga_LSZH_14_50_bb.pkl"),
    features=["track", "groundspeed", "altitude", "timedelta"],
    n_samples = 10000,
    scaler=MinMaxScaler(feature_range=(-1,1)),
    shape="image",
    info_params={"features": ["latitude", "longitude"], "index": -1},
)

path = "../deep_traffic_generation/lightning_logs/tcvae_pairs_disent/version_22/"

t = PairsVAE(X = dataset, vae_type="TCVAEPairs_disent", sim_type = "generation")
t.load(path, dataset.parameters)
gen_vae = Generation(generation=t, features = t.VAE.hparams.features, scaler=dataset.scaler)

# %%
# Get isoprobabilist transformation as SS samples within a standard Gaussian
Z = t.latent_space(0)
p_z = t.VAE.lsr.get_prior()

#Create the multivariate distrib with 1d gmm on each dimension
marginals = []
for i in range(Z.shape[1]):
    collDist = [ot.Normal(mu.item(), sigma.item()) for mu, sigma in zip(p_z.base_dist.component_distribution.base_dist.loc.squeeze(2)[i], p_z.base_dist.component_distribution.base_dist.scale.squeeze(2)[i])]
    weights = p_z.base_dist.mixture_distribution.probs[i].detach().numpy()
    mixt = ot.Mixture(collDist, weights)
    marginals.append(mixt)
prior = ot.ComposedDistribution(marginals)

# isoproba transformation from N(0,1) to prior
isoTrans = prior.getInverseIsoProbabilisticTransformation()

# %%
from traffic.core.projection import EuroPP
from traffic.data import airports
from traffic.drawing import countries
import matplotlib.pyplot as plt

with open("../prob_estimate/subset_sampling/results/subset_sampling_0.pkl", "rb") as f:
        ss_res = pkl.load(f)
        theta = ss_res["theta"]
        g = ss_res["g"]

#j = 1828 east
#j = 1717 west
#bad = 9127

j = 1717
# j = np.random.randint(len(theta[-1]))
print(j)

#Reconstruction
z = isoTrans(ot.Point(theta[-1][j]))
decoded = t.decode(torch.Tensor(z).unsqueeze(0))
reconstructed_to = gen_vae.build_traffic(decoded[:,:200], coordinates = dict(latitude =  47.44464, longitude = 8.55732), forward=True)
reconstructed_ga = gen_vae.build_traffic(decoded[:,200:], coordinates = dict(latitude = 47.500086, longitude = 8.51149), forward=True)

with plt.style.context("traffic"):
    fig, ax = plt.subplots(1, subplot_kw=dict(projection=EuroPP()))
    reconstructed_to.plot(ax, c="#4c78a8", lw=2, linestyle="--", label = "Take-off")
    reconstructed_to[0].at_ratio(0.35).plot(
        ax,
        color="#4c78a8",
        zorder=2,
        s=600,
        text_kw=dict(s=""))
    
    reconstructed_ga.plot(ax, c="#f58518", lw=2, linestyle="--", label = "Go-around")
    reconstructed_ga[0].at_ratio(0.35).plot(
        ax,
        color="#f58518",
        zorder=2,
        s=600,
        text_kw=dict(s=""))
    
    plt.legend(prop={'size': 18})

    airports["LSZH"].plot(ax, footprint=False, runways=dict(lw=1), labels=False)

    plt.show()
    
    fig.savefig("failure_traj_west.png", transparent=False, dpi=300)
# %%
