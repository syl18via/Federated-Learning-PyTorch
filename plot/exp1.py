import os
import sys
import re

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from typing import List

def smooth(scalars: List[float], weight: float) -> List[float]:
    # One of the easiest implementations I found was to use that Exponential Moving Average the Tensorboard uses, https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
    # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return np.array(smoothed)

exp_dir = sys.argv[1]

ALL_DF = {}


def key_to_config(key: str):
    '''convert a serialized key to a dict of configuration'''
    cfg = re.search(f"(?P<dataset>\w+)-(?P<target_label>\w+)_label-(?P<model>\w+)-(?P<policy>\w+)_policy", key)
    return cfg

def policy_legend_name(policy):
    if policy.lower() == "nmfli":
        return "NmFLI"
    elif policy.lower() == "size":
        return "Accord2Size"
    elif policy.lower() in ["greedy", "random", "simple"]:
        return policy[0].upper() + policy[1:].lower()
    elif policy.lower() in ["afl"]:
        return policy.upper()
    else:
        raise ValueError(policy)

fig = plt.figure(figsize=(12, 8))
ax0 = fig.add_subplot(221); ax0.grid()
ax1 = fig.add_subplot(222); ax1.grid()
ax2 = fig.add_subplot(223); ax2.grid()

for _file in sorted(os.listdir(exp_dir)):
    if not _file.endswith(".csv"):
        continue
    rst = key_to_config(_file)
    print(rst["dataset"], rst["target_label"], rst["model"], rst["policy"])
    path = os.path.join(exp_dir, _file)
    df = pd.read_csv(path)
    # ax0.plot(df["Step"], df["Task 0 test accu."], alpha=0.3)
    # smooth
    ax0.plot(df["Step"], smooth(df["Task 0 test accu."], .9), label=policy_legend_name(rst["policy"]))
    ax1.plot(df["Step"], smooth(df["Task 1 test accu."], .9), label=policy_legend_name(rst["policy"]))
    
    ax2.plot(df["Step"], (smooth(df["Task 0 test accu."], .9) + smooth(df["Task 1 test accu."], .9)) / 2, label=policy_legend_name(rst["policy"]))

ax0.set_xlabel("Step")
ax0.set_ylabel("Task 0 test accu. (smoothed)")
ax0.legend()

ax1.set_xlabel("Step")
ax1.set_ylabel("Task 1 test accu. (smoothed)")
ax1.legend()

ax2.set_xlabel("Step")
ax2.set_ylabel("Averaged test accu. (smoothed)")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(exp_dir, "fig.pdf"))
    
    







