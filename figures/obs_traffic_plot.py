# %%
from traffic.core import Traffic
import matplotlib.pyplot as plt
from traffic.core.projection import EuroPP, PlateCarree
from traffic.drawing import countries
from traffic.data import navaids
from traffic.data import airports
import matplotlib.patches as mpatches
import numpy as np

# %%
to = Traffic.from_file(
    "../deep_traffic_generation/data/training_datasets/to_LSZH_16_100.pkl"
)

ga = Traffic.from_file(
    "../deep_traffic_generation/data/training_datasets/ga_LSZH_14_100.pkl"
)

# %%
x = np.array([8.5, 8.5, 8.84, 8.84])
y = np.array([47.36, 47.52, 47.52, 47.36])

poly_corners = np.zeros((len(y), 2), np.float64)
poly_corners[:,0] = x
poly_corners[:,1] = y
poly1 = mpatches.Polygon(poly_corners, 
                        closed=True, 
                        alpha = 0.1,
                        ec='#b22222', 
                        fill=True, 
                        fc='#b22222', 
                        transform = PlateCarree())

poly2 = mpatches.Polygon(poly_corners, 
                        closed=True, 
                        alpha = 1,
                        ec='#b22222', 
                        fill=False, 
                        lw=5,
                        linestyle = "--",
                        transform = PlateCarree())

# %%
with plt.style.context("traffic"):
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 10), subplot_kw=dict(projection=EuroPP()), dpi=500
    )

    to[:2000].plot(ax, alpha=0.2, color="#ffbf79", zorder = 1)
    ga.plot(ax, alpha=0.2, color="#9ecae9", zorder = 2)

    k1 = 0
    to[k1].plot(ax, color="#f58518", lw=1.5, zorder = 1)
    to[k1].at_ratio(0.15).plot(
        ax,
        color="#f58518",
        zorder=1,
        s=600,
        text_kw=dict(s=""))
    
    k2 = 0
    ga[k2].plot(ax, color="#4c78a8", lw=1.5, zorder = 2)
    ga[k2].at_ratio(0.25).plot(
        ax,
        color="#4c78a8",
        zorder=2,
        s=600,
        text_kw=dict(s=""))

    airports["LSZH"].plot(ax, footprint=False, runways=dict(lw=1), labels=False)
    
    # ax.set_title("Observed Traffic",
    #              loc = "left",
    #              y = 0.95,
    #              fontsize=22, 
    #              fontweight=570, 
    #              fontstretch = 0)
    
    ax.add_patch(poly1)
    ax.add_patch(poly2)
    
    plt.margins(0,0)
    
    plt.show()

    fig.savefig("observed_traffic.png", transparent=False, dpi=500)
# %%
