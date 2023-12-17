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

POLICY_TO_INFO = {
    "nmfli": ("NmFLI", "red", "-"),
    "size": ("Data size-based", "orange", ":"),
    "greedy": ("Greedy", "purple", (0, (3, 5, 1, 5, 1, 5))),
    "random": ("Random", "green", "--"),
    "afl": ("AFL", "blue", "-."),
    "simple": ("Simple", "black", (0, (3, 1, 1, 1, 1, 1)))
}

def policy_to_plot_kwargs(policy):
    ''' generate legend name, color
    '''
    legend_name, color, linestyle = POLICY_TO_INFO[policy.lower()]
    kwargs = {
        "label": legend_name,
        "color": color,
        "linestyle": linestyle
    }
    if policy == "nmfli":
        kwargs["linewidth"] = 2
        # kwargs["markersize"] = 2
    return kwargs

IGNORE_POLICY = ["simple"]
target_label_to_fig_dict = {}
sub_fig_num = 3

for _file in sorted(os.listdir(exp_dir)):
    if not _file.endswith(".csv"):
        continue
    rst = key_to_config(_file)
    if rst["policy"] in IGNORE_POLICY:
        continue
    # print(rst["dataset"], rst["target_label"], rst["model"], rst["policy"])
    if rst["target_label"] not in target_label_to_fig_dict:
        target_label_to_fig_dict[rst["target_label"]] = [[] for _ in range(sub_fig_num)]
        
    path = os.path.join(exp_dir, _file)
    df = pd.read_csv(path)
    sub_fig_list = target_label_to_fig_dict[rst["target_label"]]
    # ax0.plot(df["Step"], df["Task 0 test accu."], alpha=0.3)
    # smooth
    kwargs = policy_to_plot_kwargs(rst["policy"])
    sub_fig_list[0].append((df["Step"], smooth(df["Task 0 test accu."], .9), kwargs))
    sub_fig_list[1].append((df["Step"], smooth(df["Task 1 test accu."], .9), kwargs))
    sub_fig_list[2].append((df["Step"], (smooth(df["Task 0 test accu."], .9) + smooth(df["Task 1 test accu."], .9)) / 2, kwargs))

is_one_figure = False

sub_figure_y_labels = [
    "Task 0 test accu. (smoothed)",
    "Task 1 test accu. (smoothed)",
    "Average test accu. (smoothed)"
]
if is_one_figure:
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
            all_subfig[sub_fig_id].set_ylabel(sub_figure_y_labels[sub_fig_id])
            all_subfig[sub_fig_id].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"{os.path.basename(exp_dir)}-{target_label}.pdf"))
        plt.close()
else:
    for target_label in target_label_to_fig_dict.keys():
        sub_fig_list = target_label_to_fig_dict[target_label]
        for sub_fig_id, sub_fig_data in enumerate(sub_fig_list):
            fig = plt.figure(figsize=(6, 4))
            for x, y, kwargs in sub_fig_data:
                plt.plot(x, y, **kwargs)
            plt.grid()
            plt.xlabel("Step")
            plt.ylabel(sub_figure_y_labels[sub_fig_id])
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, f"{os.path.basename(exp_dir)}-{target_label}_{sub_fig_id}.pdf"))
            plt.close()
        
    







