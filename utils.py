import json
import string
from typing import Any, Dict, List, Optional, Union
import cirq
from cirq import value
from cirq.contrib.svg.svg import circuit_to_svg
from cirq.contrib.svg import SVGCircuit
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import datetime
from matplotlib import ticker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.autonotebook import tqdm, trange

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


METRIC_NAME_TO_PLOT_NAME = {
    "energy": "Energy",
    "norm": "Norm",
    "fidelity": "Fidelity",
    "quantum_relative_entropy": "Quantum relative entropy",
    "trace_distance": "Trace distance",
}


def get_ancillary_qubits(
    problem_qubits: List[cirq.GridQubit],
) -> List[cirq.GridQubit]:
    rows = problem_qubits[-1].row + 1
    cols = problem_qubits[-1].col + 1
    top = problem_qubits[-1].row + 1
    return cirq.GridQubit.rect(rows=rows, cols=cols, top=top)


def save_circuit_to_svg(circuit: cirq.Circuit, show: bool = False):
    circuit_svg = circuit_to_svg(circuit)
    if show:
        return SVGCircuit(circuit)
    with open(
        f"circuit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.svg", "x"
    ) as file:
        file.write(circuit_svg)


def evaluate_exact_state(
    hamiltonian: cirq.PauliSum, evolution_time: float
) -> tf.Tensor:
    op = tf.cast(hamiltonian.matrix(), tf.complex64)
    exp_op = tf.linalg.expm(-evolution_time * op)
    return exp_op / tf.linalg.trace(exp_op)


def circuit_to_state(
    circuit: cirq.Circuit,
    symbol_names: List[str],
    symbol_values: tf.Tensor,
    indecis: List[int],
    backend=None,
) -> tf.Tensor:
    op = tfq.layers.State(backend)
    state = op(
        circuit,
        symbol_names=symbol_names,
        symbol_values=tf.expand_dims(symbol_values, 0),
    )
    return tf.convert_to_tensor(
        cirq.density_matrix_from_state_vector(state.numpy(), indecis)
    )


def fidelity(state1: tf.Tensor, state2: tf.Tensor) -> tf.Tensor:
    if tf.math.reduce_any(
        tf.math.logical_not(tf.math.is_finite(tf.math.real(state1)))
    ):
        raise RuntimeError(f"Invalid state1")
    if tf.math.reduce_any(
        tf.math.logical_not(tf.math.is_finite(tf.math.real(state2)))
    ):
        raise RuntimeError(f"Invalid state2")
    sqrtm1 = tf.linalg.sqrtm(state1)
    value = tf.math.real(
        tf.square(tf.linalg.trace(tf.linalg.sqrtm(sqrtm1 @ state2 @ sqrtm1)))
    )
    if not tf.math.is_finite(value):
        raise RuntimeError(f"Infinite fidelity")
    return value


def quantum_relative_entropy(state1: tf.Tensor, state2: tf.Tensor) -> tf.Tensor:
    if tf.math.reduce_any(
        tf.math.logical_not(tf.math.is_finite(tf.math.real(state1)))
    ):
        raise RuntimeError(f"Invalid state1")
    if tf.math.reduce_any(
        tf.math.logical_not(tf.math.is_finite(tf.math.real(state2)))
    ):
        raise RuntimeError(f"Invalid state2")
    value = tf.math.real(
        tf.linalg.trace(
            state1 @ (tf.linalg.logm(state1) - tf.linalg.logm(state2))
        )
    )
    if not tf.math.is_finite(value):
        raise RuntimeError(f"Infinite quantum relative entropy")
    return value


def trace_distance(state1: tf.Tensor, state2: tf.Tensor) -> tf.Tensor:
    if tf.math.reduce_any(
        tf.math.logical_not(tf.math.is_finite(tf.math.real(state1)))
    ):
        raise RuntimeError(f"Invalid state1")
    if tf.math.reduce_any(
        tf.math.logical_not(tf.math.is_finite(tf.math.real(state2)))
    ):
        raise RuntimeError(f"Invalid state2")
    diff = state1 - state2
    value = tf.math.real(
        tf.linalg.trace(tf.linalg.sqrtm(tf.transpose(diff) @ diff)) / 2
    )
    if not tf.math.is_finite(value):
        raise RuntimeError(f"Infinite trace distance")
    return value


# https://arxiv.org/pdf/2008.06517.pdf
def get_two_variable_shift_matrix(size, shift=0.5):
    index = tf.range(0, size)
    diff = tf.reshape(tf.constant([1, -1], dtype=tf.float32), [1, 2, 1])
    pair_indecis_same = tf.one_hot(
        tf.reshape(
            tf.stack(tf.meshgrid(index, index, indexing="ij"), axis=-1), [-1, 2]
        ),
        size,
    )
    pair_indecis_diff = pair_indecis_same * diff
    pair_indecis = tf.concat(
        [
            pair_indecis_same,
            -pair_indecis_same,
            pair_indecis_diff,
            -pair_indecis_diff,
        ],
        axis=0,
    )
    return tf.reduce_sum(pair_indecis, axis=1) * (np.pi / (4 * shift))


def plot_density_matrix_heatmap(
    densities: List[tf.Tensor],
    size: int = 5,
    annot: bool = False,
    titles: List[str] = None,
    file_format: str = "pdf",
    show: bool = False,
    orient: str = "horizontal",
    vmin: float = None,
    vmax: float = None,
    cmap=None,
    desc=None,
):
    n = len(densities)

    if orient == "horizontal":
        fig, axes = plt.subplots(1, n, figsize=[size, size])
    elif orient == "verical":
        fig, axes = plt.subplots(n, 1)
    else:
        raise NotImplementedError()

    if vmin is None:
        vmin = get_vmin(densities)

    if vmax is None:
        vmax = get_vmax(densities)

    for i, d in enumerate(densities):
        plot = sns.heatmap(
            d.numpy().real,
            cmap=cmap,
            square=True,
            annot=annot,
            annot_kws={"size": "xx-small"},
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            ax=axes[i] if n > 1 else axes,
            fmt=".2f",
        )
        if titles and titles[i] is not None:
            plot.set_title(
                fr"\textbf{{({list(string.ascii_lowercase)[i]})}} {titles[i]}",
                y=-0.2,
                size=8,
            )

    fig.tight_layout()

    if show:
        plt.show()
    else:
        filename = f"density_{desc + '_' if desc is not None else ''}{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        args = {"format": file_format, "bbox_inches": "tight"}
        if file_format == "pgf":
            args["backend"] = "pgf"

        plt.savefig(f"{filename}.{file_format}", **args)


def get_vmin(densities: List[tf.Tensor]) -> float:
    return min([tf.reduce_min(tf.math.real(d)) for d in densities])


def get_vmax(densities: List[tf.Tensor]) -> float:
    return max([tf.reduce_max(tf.math.real(d)) for d in densities])


def save_statistics(statistics: Any, desc=None) -> str:
    filename = f"{desc + '_' if desc is not None else ''}statistics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "x") as file:
        json.dump(dict(statistics), file, ensure_ascii=False, indent=4)
    return filename


def read_statistics(filename: str) -> pd.DataFrame:
    return pd.read_json(filename)


def plot_statistics(
    statistics: Union[Dict, pd.DataFrame],
    file_format: str = "pdf",
    size: int = 4,
    exclude: List[str] = ["energy"],
):
    # Prepare data
    if type(statistics) is not pd.DataFrame:
        statistics = pd.DataFrame.from_dict(statistics)

    df = statistics.drop(exclude, axis=1)
    df = df.rename(columns=METRIC_NAME_TO_PLOT_NAME)
    df = df.melt(id_vars=["time_step"])
    df.columns = ["Time step", "Metric", "Value"]

    vmin = 0
    vmax = df["Time step"].max()
    interval_options = [1, 2, 5, 10, 20]
    interval = interval_options[
        np.abs(vmax / np.array(interval_options) - 10).argmin()
    ]

    g = sns.FacetGrid(df, col="Metric", sharey=False, hue="Metric", height=size)
    g.map(sns.lineplot, "Time step", "Value")
    g.set_titles("{col_name}")
    g.set(xticks=range(vmin, vmax + 2, interval))

    filename = f"statistics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    args = {"format": file_format, "bbox_inches": "tight"}
    if file_format == "pgf":
        args["backend"] = "pgf"

    plt.savefig(f"{filename}.{file_format}", **args)


def flatten_multirun_index(multi_run_metrics: Dict) -> Dict:
    return {
        (run, metric): values
        for run in multi_run_metrics.keys()
        for metric, values in multi_run_metrics[run].items()
    }


def normalize_tuple_key_dict(d: Dict) -> Dict:
    return {
        "values": [
            {"key": list(key), "value": value} for key, value in d.items()
        ]
    }


def plot_multi_run_statistics(
    statistics: Union[Dict, pd.DataFrame],
    file_format: str = "pdf",
    size: int = 4,
    exclude: List[str] = ["energy"],
):
    # Prepare data
    if type(statistics) is not pd.DataFrame:
        statistics = pd.DataFrame.from_dict(statistics)

    df = statistics.drop(exclude, axis=1, level=1)
    df = df.rename(columns=METRIC_NAME_TO_PLOT_NAME)
    df["time_step"] = df.index
    df = df.drop(columns=["time_step"], level=1)
    df = df.melt(id_vars=["time_step"])
    df.columns = ["Time step", "Run", "Metric", "Value"]

    vmax = df["Time step"].max()
    interval_options = [1, 2, 5, 10, 20]
    interval = interval_options[
        np.abs(vmax / np.array(interval_options) - 10).argmin()
    ]
    vmin = interval

    g = sns.FacetGrid(df, col="Metric", sharey=False, hue="Metric", height=size)
    g.map(sns.lineplot, "Time step", "Value")
    g.set_titles("{col_name}")

    filename = f"multi_run_statistics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    args = {"format": file_format, "bbox_inches": "tight"}
    if file_format == "pgf":
        args["backend"] = "pgf"

    plt.savefig(f"{filename}.{file_format}", **args)

