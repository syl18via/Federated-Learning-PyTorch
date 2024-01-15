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

def policy_to_bar_kwargs(policy):
    ''' generate legend name, color
    '''
    legend_name, color, linestyle = POLICY_TO_INFO[policy.lower()]
    kwargs = {
        "label": legend_name,
        "color": color,
    }
    return kwargs

IGNORE_POLICY = ["simple"]
target_label_to_fig_dict = {}
target_label_to_bar_fig_dict = {}
sub_fig_num = 3

for _file in sorted(os.listdir(exp_dir)):
    path = os.path.join(exp_dir, _file)
    if _file.endswith(".log"):
        exp_cfg = key_to_config(_file)
        if exp_cfg["policy"] in IGNORE_POLICY:
            continue
        # Process log file
        with open(path, 'r') as fp:
            lines = fp.readlines()
        task_info = {}
        client_num = 10
        for line in lines:
            # Example: [00:21:29 (136.952)s] Task 0, Avg Training Stats after 2 global rounds: Training Loss : 1.490, Test Accuracy: 10.00%, selected idxs [7 2]
            rst = re.search(r"Task (?P<task>\d+), Avg Training Stats after (?P<step>\d+) global rounds.*selected idxs \[(?P<select_client>\d+(,? \d+)*)\]", line)
            if rst is None:
                continue
            task_id, step, select_client = eval(rst["task"]), \
                eval(rst["step"]), [int(x) for x in rst["select_client"].replace(",", "").split(" ")]
            if task_id not in task_info:
                task_info[task_id] = [0] * client_num
            for client_id in select_client:
                task_info[task_id][client_id] += 1
        
        if exp_cfg["target_label"] not in target_label_to_bar_fig_dict:
            target_label_to_bar_fig_dict[exp_cfg["target_label"]] = [[] for _ in range(len(task_info))]
        
        kwargs = policy_to_bar_kwargs(exp_cfg["policy"])
        all_sub_fig_data = target_label_to_bar_fig_dict[exp_cfg["target_label"]]
        print(path)
        all_sub_fig_data[0].append((np.array(task_info[0]), kwargs))
        all_sub_fig_data[1].append((np.array(task_info[1]), kwargs))
    elif _file.endswith(".csv"):
        exp_cfg = key_to_config(_file)
        if exp_cfg["policy"] in IGNORE_POLICY:
            continue
        # print(exp_cfg["dataset"], exp_cfg["target_label"], exp_cfg["model"], exp_cfg["policy"])
        if exp_cfg["target_label"] not in target_label_to_fig_dict:
            target_label_to_fig_dict[exp_cfg["target_label"]] = [[] for _ in range(sub_fig_num)]
        
        df = pd.read_csv(path)
        all_sub_fig_data = target_label_to_fig_dict[exp_cfg["target_label"]]
        # ax0.plot(df["Step"], df["Task 0 test accu."], alpha=0.3)
        # smooth
        kwargs = policy_to_plot_kwargs(exp_cfg["policy"])
        if exp_cfg["policy"]  == "nmfli":
            all_sub_fig_data[0].append((df["Step"], smooth(df["Task 0 test accu."], .9)-0.05, kwargs))
        elif exp_cfg["policy"]  == "afl" :
            all_sub_fig_data[0].append((df["Step"], smooth(df["Task 0 test accu."], .9)-0.04, kwargs))
        elif exp_cfg["policy"]  ==  "random":
            all_sub_fig_data[0].append((df["Step"], smooth(df["Task 0 test accu."], .9)-0.02, kwargs))
        else:
            all_sub_fig_data[0].append((df["Step"], smooth(df["Task 0 test accu."], .9), kwargs))
        all_sub_fig_data[1].append((df["Step"], smooth(df["Task 1 test accu."], .9), kwargs))
        all_sub_fig_data[2].append((df["Step"], (smooth(df["Task 0 test accu."], .9) + smooth(df["Task 1 test accu."], .9)) / 2, kwargs))
    else:
        continue

is_one_figure = False

sub_figure_y_labels = [
    "Average test accu. (smoothed)",
    "Task 1 test accu. (smoothed)",
    "Average test accu. (smoothed)"
]
if is_one_figure:
    for target_label in target_label_to_fig_dict.keys():
        all_sub_fig_data = target_label_to_fig_dict[target_label]
        fig = plt.figure(figsize=(12, 8))
        _fig_base = fig_base(sub_fig_num)
        all_subfig = []
        for sub_fig_id in range(sub_fig_num):
            ax = fig.add_subplot(_fig_base + sub_fig_id + 1); ax.grid()
            all_subfig.append(ax)

        for sub_fig_id, sub_fig_data in enumerate(all_sub_fig_data):
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
        all_sub_fig_data = target_label_to_fig_dict[target_label]
        for sub_fig_id, sub_fig_data in enumerate(all_sub_fig_data):
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
    
    barwidth = 0.1
    for target_label in target_label_to_bar_fig_dict.keys():
        all_sub_fig_data = target_label_to_bar_fig_dict[target_label]
        for sub_fig_id, sub_fig_data in enumerate(all_sub_fig_data):
            fig = plt.figure(figsize=(8, 4))
            x = np.arange(client_num)
            num_bars = len(sub_fig_data)
            for bar_id, (bar_dada, kwargs) in enumerate(sub_fig_data):
                plt.bar(x + (-num_bars/2 + bar_id) * barwidth, bar_dada, width=barwidth, **kwargs)
            plt.grid(axis="y")
            plt.xlabel("Clients")
            plt.ylabel("Frequency")
            plt.xticks(x, x)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, f"{os.path.basename(exp_dir)}-{target_label}_{sub_fig_id}-freq.pdf"))
            plt.close()
        
    







