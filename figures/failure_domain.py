# %%
from deep_traffic_generation.tcvae_pairs_disent import TCVAE_Pairs_disent
from deep_traffic_generation.VAE_Generation import PairsVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDatasetPairsRandom
from traffic.core import Traffic

import openturns as ot
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
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

j = 97

with open("../prob_estimate/subset_sampling/results/subset_sampling_"+str(j)+".pkl", "rb") as f:
        ss_res = pkl.load(f)
        theta_ss = ss_res["theta"]
        g = ss_res["g"]
        
with open("../prob_estimate/monte_carlo/results/monte_carlo_"+str(j)+".pkl", "rb") as f:
        mc_res = pkl.load(f)
        failure_mc = torch.cat(mc_res["event_inputs"]).numpy()

#inputs from the failure domain
failure_ss = theta_ss[-1][g[-1]<0]
#from N(0,1) to prior
failure_ss = isoTrans(failure_ss)

Z_emb = Z[:,[4,9]]
Z_ss_emb = failure_ss[:,[4,9]]
Z_mc_emb = failure_mc[:,[4,9]]

with plt.style.context("traffic"):
    fig, ax = plt.subplots(1, figsize=(15, 10))
    ax.scatter(Z_emb[:, 0], Z_emb[:, 1], s=8, c ='#aaaaaa', alpha = 0.3, label = "VAE Latent Space")
    ax.scatter(Z_ss_emb[:, 0], Z_ss_emb[:, 1], s=8, c="#4c78a8", label = "Failure domain of SS")
    ax.scatter(Z_mc_emb[:, 0], Z_mc_emb[:, 1], s=20, c="#f58518", label = "Failure domain of MC")
    
    ax.set_xlabel('Dimension 5', fontsize=20)
    ax.set_ylabel('Dimension 9', fontsize = 20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    ax.title.set_text("Failure Domains of run "+str(j))
    ax.title.set_fontsize(22)
    ax.legend(loc='upper left', fontsize=16)

    plt.show()

    fig.savefig("failure_domain_run"+str(j)+".png", dpi=300)
# %%
