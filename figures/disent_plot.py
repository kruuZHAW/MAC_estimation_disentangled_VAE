# %%
from deep_traffic_generation.tcvae_pairs_disent import TCVAE_Pairs_disent
from deep_traffic_generation.VAE_Generation import PairsVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDatasetPairsRandom
from traffic.core import Traffic

from sklearn.preprocessing import MinMaxScaler
import torch
from torch.distributions import (
    Distribution, MixtureSameFamily, Normal
)
from torch.distributions.categorical import Categorical
import numpy as np

# %%
dataset = TrafficDatasetPairsRandom.from_file(
    ("../deep_traffic_generation/data/training_datasets/to_LSZH_16_50_bb.pkl", "../deep_traffic_generation/data/training_datasets/ga_LSZH_14_50_bb.pkl"),
    features=["track", "groundspeed", "altitude", "timedelta"],
    n_samples = 30000,
    scaler=MinMaxScaler(feature_range=(-1,1)),
    # scaler=None,
    shape="image",
    info_params={"features": ["latitude", "longitude"], "index": -1},
)

# %%
path = "../deep_traffic_generation/lightning_logs/tcvae_pairs_disent/version_22/"

t = PairsVAE(X = dataset, vae_type="TCVAEPairs_disent", sim_type = "generation")
t.load(path, dataset.parameters)
g = Generation(generation=t, features = t.VAE.hparams.features, scaler=dataset.scaler) 

# %% 
n_gen = 50
Z = t.latent_space(0)
p_z = t.VAE.lsr.get_prior()

# Storage of generated points
tos = []
gas = []
latents = []

# Get marginal parameters
probs, means, scales = p_z.base_dist.mixture_distribution.probs.detach(), p_z.base_dist.component_distribution.base_dist.loc.squeeze(2).detach(), p_z.base_dist.component_distribution.base_dist.scale.squeeze(2).detach()


# We sample a reference point in the prior
ref = p_z.sample(torch.Size([1])).squeeze(-1)

# We modify each dimension separately
for i in range(ref.shape[1]):
    z0 = ref.repeat(n_gen+1,1)
    
    #build marginal distribution
    mix = Categorical(probs=probs[i,:])
    comp = Normal(means[i,:], scales[i,:])
    marginal = MixtureSameFamily(mix, comp)
    
    z0[1:,i] = marginal.sample(torch.Size([n_gen])) # the first is the reference point
    latents.append(z0)
    
    #VAE decoder
    gen0 = t.decode(torch.Tensor(z0))
    to = g.build_traffic(gen0[:,:200], coordinates = dict(latitude =  47.44464, longitude = 8.55732), forward=True).iterate_lazy().resample("1s").cumulative_distance().eval()
    tos.append(to)
    ga = g.build_traffic(gen0[:,200:], coordinates = dict(latitude = 47.500086, longitude = 8.51149), forward=True).iterate_lazy().resample("1s").cumulative_distance().eval()
    gas.append(ga)
    
# %%
import matplotlib.pyplot as plt
from traffic.core.projection import EuroPP
from traffic.data import airports

with plt.style.context("traffic"):
    fig, ax = plt.subplots(2, 5, figsize=(20, 6), subplot_kw=dict(projection=EuroPP()), sharey=True, sharex=True)
    
    for i in range(ref.shape[1]):
        tos[i].plot(ax[i//5,i%5], color="#9ecae9", alpha = 0.5)
        tos[i]["TRAJ_0"].plot(ax[i//5,i%5], color="#4c78a8", alpha = 1)
        gas[i].plot(ax[i//5,i%5], color="#ffbf79", alpha = 0.5)
        gas[i]["TRAJ_0"].plot(ax[i//5,i%5], color="#f58518", alpha = 1)
        airports["LSZH"].plot(ax[i//5,i%5], footprint=False, runways=dict(lw=1), labels=False)
        ax[i//5,i%5].set_title('Marginal distribution {}'.format(i+1), fontsize = 16, y=0, pad=-15, verticalalignment="top")
    
    fig.suptitle('Variations induced by the prior marginals', fontsize=24, fontweight = "bold", y = 0.95)
    plt.show()
    
fig.savefig("disent_trajs.png", bbox_inches='tight', dpi = 200)

# %%
import altair as alt

#Number of the dimension displayed above
j = 10

chart1 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "timestamp:T",
                axis=alt.Axis(tickCount= 10),
                title="Time (CET)",
            ),
            y=alt.Y("altitude", title=None),
            opacity=alt.condition(
                alt.datum.flight_id == "TRAJ_0",
                alt.value(1),
                alt.value(0.5),
            ),
            color=alt.condition(
                alt.datum.flight_id == "TRAJ_0",
                alt.value("#4c78a8"),
                alt.value("#9ecae9"),
            ),
        )
        for flight in tos[j-1]
    )
).properties(title="Altitude (in ft)", width=500, height=250)

chart2 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "timestamp:T",
                axis=alt.Axis(tickCount= 10),
                title="Time (CET)",
            ),
            y=alt.Y("groundspeed", title=None),
            opacity=alt.condition(
                alt.datum.flight_id == "TRAJ_0",
                alt.value(1),
                alt.value(0.5),
            ),
            color=alt.condition(
                alt.datum.flight_id == "TRAJ_0",
                alt.value("#4c78a8"),
                alt.value("#9ecae9"),
            ),
        )
        for flight in tos[j-1]
    )
).properties(title="Ground speed (in kts)", width=500, height=250)

chart3 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "timestamp:T",
                axis=alt.Axis(tickCount= 10),
                title="Time (CET)",
            ),
            y=alt.Y("altitude", title=None),
            opacity=alt.condition(
                alt.datum.flight_id == "TRAJ_0",
                alt.value(1),
                alt.value(0.5),
            ),
            color=alt.condition(
                alt.datum.flight_id == "TRAJ_0",
                alt.value("#f58518"),
                alt.value("#ffbf79"),
            ),
        )
        for flight in gas[j-1]
    )
).properties(title="Altitude (in ft)", width=500, height=250)

chart4 = alt.layer(
    *(
        flight.chart().encode(
            x=alt.X(
                "timestamp:T",
                axis=alt.Axis(tickCount= 10),
                title="Time (CET)",
            ),
            y=alt.Y("groundspeed", title=None),
            opacity=alt.condition(
                alt.datum.flight_id == "TRAJ_0",
                alt.value(1),
                alt.value(0.5),
            ),
            color=alt.condition(
                alt.datum.flight_id == "TRAJ_0",
                alt.value("#f58518"),
                alt.value("#ffbf79"),
            ),
        )
        for flight in gas[j-1]
    )
).properties(title="Ground speed (in kts)", width=500, height=250)

plots = (chart1 + chart3) | (chart2 + chart4)

plots
# %%
