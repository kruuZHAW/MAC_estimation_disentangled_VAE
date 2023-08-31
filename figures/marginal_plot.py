# %%
from deep_traffic_generation.tcvae_pairs_disent import TCVAE_Pairs_disent
from deep_traffic_generation.VAE_Generation import PairsVAE
from traffic.algorithms.generation import Generation
from deep_traffic_generation.core.datasets import TrafficDatasetPairsRandom
from traffic.core import Traffic

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

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

    
#%%
Z = t.latent_space(0)
df_Z = pd.DataFrame(Z, columns = ["marginal {}".format(i+1) for i in range(Z.shape[1])])
data = df_Z.melt().rename(columns=dict(variable="step"))
marginal_std_serie = data.groupby('step')['value'].std()
data['std_marginal'] = data['step'].map(marginal_std_serie)

marginal_dict = {1: 'Marginal 1',
              2: 'Marginal 2',
              3: 'Marginal 3',
              4: 'Marginal 4',
              5: 'Marginal 5',
              6: 'Marginal 6',
              7: 'Marginal 7',
              8: 'Marginal 8',
              9: 'Marginal 9',
              10: 'Marginal 10'}


#%%
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# we generate a color palette with Seaborn.color_palette()
pal = sns.color_palette(palette='viridis_r', n_colors=10)
# pal = sns.diverging_palette(30, 250, l=70, n=10)

# in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
g = sns.FacetGrid(data, row='step',hue='std_marginal', aspect=9, height=1.2, palette=pal)
g.fig.set_figwidth(6)
g.fig.set_figheight(15)

# then we add the densities kdeplots for each month
g.map(sns.kdeplot, 'value',
      bw_adjust=1, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)

# here we add a white line that represents the contour of each kdeplot
g.map(sns.kdeplot, 'value', 
      bw_adjust=1, clip_on=False, 
      color="w", lw=2)

# here we add a horizontal line for each plot
g.map(plt.axhline, y=0,
      lw=2, clip_on=False)

# we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
# notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
for i, ax in enumerate(g.axes.flat):
    ax.text(-8, 0.1, marginal_dict[i+1],
            fontweight='bold', fontsize=12,
            color=ax.lines[-1].get_color())
    # ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1)
    
# we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
g.fig.subplots_adjust(hspace=0)

# eventually we remove axes titles, yticks and spines
g.set_titles("")
g.set(yticks=[])
g.set_ylabels("")
g.despine(bottom=True, left=True)

# plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
plt.xlabel('', fontweight='bold', fontsize=15)
# g.fig.suptitle('Prior Distribution Marginals',
#                ha='center',
#                fontsize=20,
#                fontweight="bold")

plt.show()
g.fig.savefig("marginals_ridgeline_2.png")


# %%
