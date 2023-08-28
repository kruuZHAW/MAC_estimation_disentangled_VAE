#%% 
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

import traffic

with open("../prob_estimate/sensitivity_analysis/results.pkl", "rb") as f:
        sobo_res = pkl.load(f)


# %%
from matplotlib.patches import ConnectionPatch
import seaborn as sns

dic = {}
for i, v in enumerate(sobo_res["first order"]):
    dic["Marginal " + str(i+1)] = np.abs(v)

# group together all elements in the dictionary whose value is less than 2
# name this group 'All the rest'
import itertools
newdic={}
for key, group in itertools.groupby(dic, lambda k: 'All the rest' if (dic[k]<0.01) else k):
     newdic[key] = sum([dic[k] for k in list(group)])   


with plt.style.context("traffic"):
    # make figure and assign axis objects
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(wspace=0)

    # pie chart parameters
    ratios = list(newdic.values()) / sum(newdic.values())
    labels = newdic.keys()
    explode = [0.1, 0, 0]
    # rotate so that first wedge is split by the x-axis
    angle = -180 * ratios[0]
    ax1.pie(ratios, autopct='%1.1f%%', startangle=angle,
            labels=labels, explode=explode)
    ax1.set_title("Sobol' indices for the minimal distance",
                loc = "left",
                y = 1.08,
                x = -0.2,
                fontsize=18, 
                fontweight=570, 
                fontstretch = 0)

    # bar chart parameters

    xpos = 0
    bottom = 0
    ratios = [dic[i] for i in list(set(dic) - set(newdic))]
    ratios = ratios / sum(ratios)
    width = .2
    pal = sns.color_palette("Blues", len(ratios))

    for j in range(len(ratios)):
        height = ratios[j]
        ax2.bar(xpos, height, width, bottom=bottom, color=pal[j])
        ypos = bottom + ax2.patches[j].get_height() / 2
        bottom += height
        ax2.text(xpos, ypos, "%d%%" % (ax2.patches[j].get_height() * 100),
                ha='center')

    ax2.set_title('Non-significant marginals')
    ax2.axis('off')
    ax2.legend(list(set(dic) - set(newdic)))
    ax2.set_xlim(- 2.5 * width, 2.5 * width)

    # use ConnectionPatch to draw lines between the two plots
    # get the wedge data
    theta1, theta2 = ax1.patches[0].theta1, ax1.patches[0].theta2
    center, r = ax1.patches[0].center, ax1.patches[0].r
    bar_height = sum([item.get_height() for item in ax2.patches])

    # draw top connecting line
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = center[1]
    con = ConnectionPatch(xyA=(- width / 2, bar_height), 
                        xyB=(x, y),
                        linestyle='dotted',
                        coordsA="data", 
                        coordsB="data", 
                        axesA=ax2, 
                        axesB=ax1)
    con.set_color([0, 0, 0])
    con.set_linewidth(3)
    ax2.add_artist(con)

plt.show()
fig.savefig("sobol.png", bbox_inches='tight', dpi = 300)
# %%
