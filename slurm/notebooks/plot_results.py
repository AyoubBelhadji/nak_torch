# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: nak-torch
#     language: python
#     name: python3
# ---

# %%
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import Any
which_exp = None
sns.set_style('dark')

# %%
which_exp = "/path/to/data/problem_TIMESTAMP"

# %%
which_exp

# %%
# Define `which_exp` if there's a folder path that you want
if which_exp is None or not os.path.isdir(which_exp):
    files = os.listdir("../data")
    jokers = [d for d in files if "joker" in d]
    which_exp = os.path.join("..", "data", jokers[0])
assert which_exp is not None and os.path.isdir(which_exp)
experiments = os.listdir(which_exp)
res: dict[tuple[str, int], list[tuple[Any,...]]] = {}
for exp in experiments:
    with open(os.path.join(which_exp, exp), "rb") as f:
        joker_data = pickle.load(f)
    config = joker_data["config"]
    output = joker_data["output"]
    alg = config["algorithm"]
    key = (alg, config["n_particles"])
    curr_vals = res.get(key, None)
    if curr_vals is None:
        res[key] = [(config, output)]
    else:
        res[key].append((config, output))

# %%
cols = ["algorithm", "n_particles", "ksd", "rmse", "foerstner", "sample_sq_mmd", "spectral"]
results = {k: [] for k in cols}
for (k,v) in res.items():
    for (_,output) in v:
        results["algorithm"].append(k[0])
        results["n_particles"].append(k[1])
        for v_k in cols[2:]:
            results[v_k].append(np.float64(output[v_k]))

# %%
df = pd.DataFrame(results)
df['mmd'] = np.sqrt(df.sample_sq_mmd)

# %%
# Choose which algorithms you want to plot
algs = ["msip_fredholm", "msip_gradientfree"]
# Choose which metric you want to plot
metric = "foerstner"
metric_col = df.get(metric)
assert metric_col is not None
limits = [m(metric_col) for m in (np.min, np.max)]
limits = (limits[0]/2, limits[1]*2)
fig, axs = plt.subplots(1, len(algs), figsize=(3*len(algs),2*np.sqrt(len(algs))))
for (ax,algorithm) in zip(axs, algs):
    ax = df[df.algorithm == algorithm].boxplot(column=metric, by="n_particles", ax=ax)
    ax.set_title(algorithm)
    ax.set_yscale("log")
    ax.set_ylim(*limits)
fig.suptitle(metric)
plt.show()

# %%
# If you want to plot a bunch of metrics for one algorithm
# Choose algorithm
algorithm = "msip_fredholm"
# Choose which metrics to plot
metrics = ['ksd', 'rmse', 'foerstner', 'spectral', 'mmd']
fig, axs = plt.subplots(1, len(metrics), figsize=(3*len(metrics),2*np.sqrt(len(metrics))))
for (ax,metric) in zip(axs, metrics):
    ax = df[
        (df.algorithm == algorithm)
    ].boxplot(column=metric, by="n_particles", ax=ax)
    ax.set_title(metric)
    ax.set_yscale("log")
fig.suptitle(algorithm)
plt.show()
