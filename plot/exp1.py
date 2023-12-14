import os
import sys
import re

import pandas as pd
import numpy as np
import math

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

def fig_base(fig_num, row_first=True):
    if row_first:
        row_num = math.ceil(math.sqrt(fig_num))
        col_num = math.ceil(fig_num / row_num)
    else:
        col_num = math.ceil(math.sqrt(fig_num))
        row_num = math.ceil(fig_num / col_num)
    return row_num * 100 + col_num * 10

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

IGNORE_POLICY = ["simple"]
target_label_to_fig_dict = {}
sub_fig_num = 3

for _file in sorted(os.listdir(exp_dir)):
    if not _file.endswith(".csv"):
        continue
    rst = key_to_config(_file)
    if rst["policy"] in IGNORE_POLICY:
        continue
    print(rst["dataset"], rst["target_label"], rst["model"], rst["policy"])
    if rst["target_label"] not in target_label_to_fig_dict:
        target_label_to_fig_dict[rst["target_label"]] = [[] for _ in range(sub_fig_num)]
        
    path = os.path.join(exp_dir, _file)
    df = pd.read_csv(path)
    sub_fig_list = target_label_to_fig_dict[rst["target_label"]]
    # ax0.plot(df["Step"], df["Task 0 test accu."], alpha=0.3)
    # smooth
    kwargs = {
        "label": policy_legend_name(rst["policy"])
    }
    if rst["policy"] == "nmfli":
        kwargs["color"] = "red"
        kwargs["linewidth"] = 2
        # kwargs["markersize"] = 2
    sub_fig_list[0].append((df["Step"], smooth(df["Task 0 test accu."], .9), kwargs))
    sub_fig_list[1].append((df["Step"], smooth(df["Task 1 test accu."], .9), kwargs))
    sub_fig_list[2].append((df["Step"], (smooth(df["Task 0 test accu."], .9) + smooth(df["Task 1 test accu."], .9)) / 2, kwargs))

for target_label in target_label_to_fig_dict.keys():
    sub_fig_list = target_label_to_fig_dict[target_label]
    fig = plt.figure(figsize=(12, 8))
    _fig_base = fig_base(sub_fig_num)
    all_subfig = []
    for sub_fig_id in range(sub_fig_num):
        ax = fig.add_subplot(_fig_base + sub_fig_id + 1); ax.grid()
        all_subfig.append(ax)

    for sub_fig_id, sub_fig_data in enumerate(sub_fig_list):
        for x, y, kwargs in sub_fig_data:
            all_subfig[sub_fig_id].plot(x, y, **kwargs)
            
    for sub_fig_id in range(sub_fig_num):
        all_subfig[sub_fig_id].set_xlabel("Step")
        all_subfig[sub_fig_id].set_ylabel("Task 0 test accu. (smoothed)")
        all_subfig[sub_fig_id].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, f"{target_label}.pdf"))
    plt.close()
        
    







