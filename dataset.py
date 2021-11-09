import numpy as np
from numpy.compat.py3k import sixu
import tensorflow as tf
from itertools import combinations, count, product
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(
    context="paper", style="ticks", palette="colorblind", font="serif"
)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{lmodern}")
plt.rc("font", family="serif")
plt.rc("font", size=8)
plt.rc("pgf", texsystem="pdflatex")
plt.rc("pgf", rcfonts=False)
plt.rc("pgf", preamble=r"\usepackage{lmodern}")
plt.rc("axes", labelsize=8)
plt.rc("legend", fontsize=8)
plt.rc("legend", title_fontsize=8)
plt.rc("xtick", labelsize=8)
plt.rc("ytick", labelsize=8)


def bars_and_stripes(n=100, s=2, no_fills_or_empties=False):
    samples = []
    valid_values = valid_bars_and_stripes(s, no_fills_or_empties)
    for i in np.random.choice(valid_values.shape[0], n):
        samples.append(valid_values[i])
    return np.array(samples)


def valid_bars_and_stripes(s=2, no_fills_or_empties=False):
    samples = []
    all_combinations = []
    for i in range(
        1 if no_fills_or_empties else 0, s if no_fills_or_empties else s + 1
    ):
        all_combinations = [
            *all_combinations,
            *list(combinations(range(s), r=i)),
        ]
    for indecis in all_combinations:
        sample = np.zeros([s, s], dtype=int)
        for index in indecis:
            sample[index] = 1
        samples.append(sample)
        sample_t = np.transpose(sample)
        if not np.all(sample == sample_t):
            samples.append(sample_t)
    return np.array(samples)


def invalid_bars_and_stripes(s=2, no_fills_or_empties=False):
    all_values = np.array(list(product([0, 1], repeat=s ** 2)))
    all_values = np.reshape(all_values, [-1, s, s])
    valid_values = valid_bars_and_stripes(s, no_fills_or_empties)
    invalid_values = []

    for value in all_values:
        match = False
        for valid_value in valid_values:
            if np.all(valid_value == value):
                match = True
        if not match:
            invalid_values.append(value)
    return np.array(invalid_values)


def bars_and_stripes_probability(values):
    s = values.shape[-1]
    values = values.reshape([-1, s ** 2])
    values = np.array([binary_array_to_int(i) for i in values])
    counts = np.bincount(values, minlength=2 ** (s ** 2))
    return counts / np.sum(counts)


def binary_array_to_int(a, n=None):
    if n is None:
        n = a.shape[0]
    return a.dot(2 ** np.arange(n)[::-1])


def int_to_binary_array(b, n=None):
    binary = bin(b)[2:]
    if n is not None and len(binary) < n:
        binary = "".join(["0" for _ in range(n - len(binary))]) + binary
    return [int(i) for i in binary]


def samples_from_distribution(distribution, n=16, size=4):
    samples = []
    for i in np.random.choice(distribution.shape[0], size=n, p=distribution):
        samples.append(
            np.array(int_to_binary_array(i, n=size)).reshape(
                int(np.sqrt(size)), int(np.sqrt(size))
            )
        )
    return np.array(samples)


def plot_dataset(
    data, file_format="pdf", size=5, cmap=None, desc=None, shape=None, sort=True, save=True
):
    if sort:
        data = sort_dataset(data)
    n = data.shape[0]
    n_cols = 4 if shape is None else shape[1]
    n_rows = int(np.ceil(n / n_cols)) if shape is None else shape[0]

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=[n_cols * size, n_rows * size]
    )

    for i, ax in enumerate(axes.flatten()):
        if i >= n:
            fig.delaxes(ax)
            continue
        plot = sns.heatmap(
            data[i],
            cmap=cmap,
            square=True,
            vmin=0,
            vmax=1,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )
        for _, spine in plot.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.2)

        plot.set_title(
            fr"$|{''.join(data[i].flatten().astype(str).tolist())} \rangle$",
            size=7,
            y=0.95,
        )
    fig.subplots_adjust(wspace=0.1, hspace=0.4)

    filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}{'_' + desc if desc is not None else ''}"

    if save:
        args = {"format": file_format, "bbox_inches": "tight"}
        if file_format == "pgf":
            args["backend"] = "pgf"
        plt.savefig(f"{filename}.{file_format}", **args)
    else:
        plt.show()
    plt.close()


def sort_dataset(data):
    data_flatten = data.reshape(-1, data.shape[-1] ** 2)
    idx = np.argsort(np.sum(data_flatten, 1))
    return data[idx]

