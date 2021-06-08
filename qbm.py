from collections import defaultdict

from utils import evaluate_exact_state, fidelity, plot_density_matrix_heatmap
import numpy as np
import cirq
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import optimize as scipyoptimize
from hamiltonian import Hamiltonian
from qite import QITE
from typing import Callable, Dict, Optional
from tqdm import trange, tqdm

# import pybobyqa


class QBM:
    def __init__(
        self,
        circuit: cirq.Circuit,
        symbol_names: tf.Tensor,
        initial_symbol_values: tf.Tensor,
        hamiltonian: Hamiltonian,
        evolution_time: float = 1 / 2,
        n_timesteps: int = 20,
        regularization: float = 1e-4,
        use_grad: Optional[bool] = True,
        spsa: Optional[int] = None,
        finite_difference: float = 0.1,
        qite_spsa: Optional[int] = None,
        qite_spsa_repeats: Optional[int] = 10,
        qite_spsa_perturbation: Optional[float] = np.pi / 80,
        verbose: int = 1,
        verbose_qite: int = 0,
        calculate_qite_solution: bool = True,
        backend=None,
        differentiator=None,
    ) -> None:
        self.qite = QITE(
            n_extra_timesteps=0,
            regularization=regularization,
            spsa=qite_spsa,
            spsa_perturbation=qite_spsa_perturbation,
            verbose=verbose_qite,
            backend=backend,
            differentiator=differentiator,
        )
        self.use_grad = use_grad
        self.spsa = spsa
        self.finite_difference = finite_difference
        self.qite_spsa = qite_spsa
        self.qite_spsa_repeats = qite_spsa_repeats
        self.verbose = verbose
        self.circuit = circuit
        self.symbol_names = symbol_names
        self.initial_symbol_values = initial_symbol_values
        self.hamiltonian = hamiltonian
        self.evolution_time = evolution_time
        self.n_timesteps = n_timesteps
        self.calculate_qite_solution = calculate_qite_solution

    def train(
        self,
        p_data: tf.Tensor,
        loss_fn: Callable[
            [tf.Tensor, tf.Tensor], tf.Tensor
        ] = tf.losses.categorical_crossentropy,
        epochs: int = 100,
        optimizer: tf.optimizers.Optimizer = tf.optimizers.SGD(learning_rate=1),
        density_to_p: Callable[[tf.Tensor], tf.Tensor] = lambda d: tf.math.real(
            tf.linalg.diag_part(d)
        ),
        spsa_args: Dict = {
            "a": 0.5,
            "c": 0.1,
            "alpha": 0.602,
            "gamma": 0.101,
            "A": 0,
            "adapt": 0,
        },
        normalization: bool = False,
        patience: int = 5,
    ):
        self.metrics = defaultdict(list)
        self.epochs = epochs
        self.symbol_values = None
        self.state = None

        if (
            self.use_grad
            and self.spsa is not None
            and spsa_args["adapt"] is not None
            and spsa_args["adapt"] > 0
        ):
            spsa_args = self.adapt_spsa_parameters(
                loss_fn, p_data, density_to_p, epochs, spsa_args
            )

        try:

            if self.use_grad:
                # Use tqdm if verbose
                iterator = (
                    trange(0, epochs, desc="QBM Train", leave=False)
                    if self.verbose > 0
                    else range(epochs)
                )
                for epoch in iterator:
                    self.metrics["epoch"].append(epoch)
                    self.metrics["coefficients"].append(
                        self.hamiltonian.coefficients.numpy().tolist()
                    )

                    symbol_values, state = self.run_qite(self.hamiltonian)

                    self.symbol_values = symbol_values
                    self.metrics["symbol_values"].append(
                        self.symbol_values.numpy().flatten().tolist()
                    )
                    self.state = state
                    self.metrics["state"].append(
                        self.state.numpy().real.flatten().tolist()
                    )

                    loss = loss_fn(p_data, density_to_p(state))
                    tqdm.write(f"Loss:{loss.numpy().item()}")
                    self.metrics["loss"].append(loss.numpy().item())
                    tqdm.write(
                        f"Coefficients: {self.hamiltonian.coefficients.numpy()}"
                    )

                    self.optimize(
                        loss_fn, p_data, density_to_p, optimizer, spsa_args
                    )

                    if normalization:
                        self.hamiltonian.normalize_coefficients()
            else:
                iterator = tqdm(total=epochs, desc="QBM Train", leave=False)

                def scipy_optimize(params):
                    if len(self.metrics["epoch"]) == 0:
                        self.metrics["epoch"].append(0)
                    else:
                        self.metrics["epoch"].append(
                            self.metrics["epoch"][-1] + 1
                        )
                    epoch = self.metrics["epoch"][-1]

                    self.hamiltonian.coefficients = tf.Variable(params)
                    if normalization:
                        self.hamiltonian.normalize_coefficients()
                    self.metrics["coefficients"].append(
                        self.hamiltonian.coefficients.numpy().tolist()
                    )
                    symbol_values, state = self.run_qite(self.hamiltonian)
                    self.symbol_values = symbol_values
                    self.metrics["symbol_values"].append(
                        self.symbol_values.numpy().flatten().tolist()
                    )
                    self.state = state
                    self.metrics["state"].append(
                        self.state.numpy().real.flatten().tolist()
                    )
                    loss = loss_fn(p_data, density_to_p(state))
                    tqdm.write(f"Loss:{loss.numpy().item()}")
                    self.metrics["loss"].append(loss.numpy().item())
                    iterator.update(1)
                    return loss.numpy()

                scipyoptimize.minimize(
                    scipy_optimize,
                    tf.random.uniform(
                        [len(self.hamiltonian.problem)], minval=-1, maxval=1
                    ).numpy(),
                    method="COBYLA",
                    options={"rhoberg": 0.5, "maxiter": epochs, "disp": True},
                )
                iterator.close()
        except Exception as error:
            print(error)
            pass

        return (
            self.hamiltonian,
            self.state,
            self.symbol_values,
            self.metrics,
        )

    def optimize(self, loss_fn, p_data, density_to_p, optimizer, spsa_args):
        if self.spsa is not None:
            self.optimize_spsa(loss_fn, p_data, density_to_p, spsa_args)
        else:
            self.optimize_finite_difference(
                loss_fn, p_data, density_to_p, optimizer
            )
        return

    def optimize_spsa(self, loss_fn, p_data, density_to_p, spsa_args):
        a = spsa_args["a"]
        c = spsa_args["c"]
        alpha = spsa_args["alpha"]
        gamma = spsa_args["gamma"]
        A = spsa_args["A"]
        k = self.metrics["epoch"][-1]
        ak = a / ((k + 1 + A) ** alpha)
        ck = c / ((k + 1) ** gamma)
        tqdm.write(
            f"a = {a}, c = {c}, alpha = {alpha}, gamma = {gamma}, A = {A}, ak = {ak}, ck = {ck}"
        )
        theta = self.hamiltonian.coefficients
        gradients = []

        for _ in trange(0, self.spsa, desc="SPSA Average", leave=False):
            bernoulli = tfp.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
            delta = (bernoulli.sample([theta.shape[0]]) * 2 - 1) * ck
            theta_pos = theta + delta
            theta_neg = theta - delta
            hamiltonian_pos = Hamiltonian(self.hamiltonian.problem, theta_pos)
            hamiltonian_neg = Hamiltonian(self.hamiltonian.problem, theta_neg)
            _, pos_state = self.run_qite(hamiltonian_pos)
            _, neg_state = self.run_qite(hamiltonian_neg)
            loss_pos = loss_fn(p_data, density_to_p(pos_state))
            loss_neg = loss_fn(p_data, density_to_p(neg_state))
            gradients.append((loss_pos - loss_neg) / (2 * delta))

        gradient = tf.reduce_mean(tf.convert_to_tensor(gradients), 0)
        tqdm.write(f"Gradient:{gradient.numpy()}")
        self.metrics["gradient"].append(gradient.numpy().tolist())
        theta = theta - ak * gradient
        self.hamiltonian.coefficients = theta

    def adapt_spsa_parameters(
        self, loss_fn, p_data, density_to_p, epochs, spsa_args,
    ):
        a = spsa_args["a"]
        c = spsa_args["c"]
        alpha = spsa_args["alpha"]
        A = 0.01 * epochs
        theta = self.hamiltonian.coefficients
        gradients = []

        for _ in trange(0, spsa_args["adapt"], desc="SPSA Adapt", leave=False):
            bernoulli = tfp.distributions.Bernoulli(probs=0.5, dtype=tf.float32)
            delta = (bernoulli.sample([theta.shape[0]]) * 2 - 1) * c
            theta_pos = theta + delta
            theta_neg = theta - delta
            hamiltonian_pos = Hamiltonian(self.hamiltonian.problem, theta_pos)
            hamiltonian_neg = Hamiltonian(self.hamiltonian.problem, theta_neg)
            _, pos_state = self.run_qite(hamiltonian_pos, skip_metrics=True)
            _, neg_state = self.run_qite(hamiltonian_neg, skip_metrics=True)
            loss_pos = loss_fn(p_data, density_to_p(pos_state))
            loss_neg = loss_fn(p_data, density_to_p(neg_state))
            gradients.append((loss_pos - loss_neg) / (2 * delta))
        estimate = tf.reduce_mean(tf.abs(tf.convert_to_tensor(gradients)))

        spsa_args["a"] = a * ((1 + A) ** alpha) / estimate
        spsa_args["A"] = A
        return spsa_args

    def optimize_finite_difference(
        self, loss_fn, p_data, density_to_p, optimizer
    ):
        diff_losses = []
        for finite_difference_hamiltonian in tqdm(
            self.hamiltonian.finite_difference_models(self.finite_difference),
            desc="Finite difference",
            leave=False,
        ):
            _, diff_state = self.run_qite(finite_difference_hamiltonian)
            diff_loss = loss_fn(p_data, density_to_p(diff_state))
            tqdm.write(f"Diff loss:{diff_loss.numpy().item()}")
            diff_losses.append(diff_loss)
        gradient = (
            tf.convert_to_tensor(diff_losses) - self.metrics["loss"][-1]
        ) / self.finite_difference
        tqdm.write(f"Gradient:{gradient.numpy()}")
        optimizer.apply_gradients(
            zip([gradient], [self.hamiltonian.coefficients])
        )

    def run_qite(self, hamiltonian, skip_metrics=False):

        if self.calculate_qite_solution and not skip_metrics:
            exact_state = evaluate_exact_state(
                hamiltonian.to_pauli_sum(), self.evolution_time,
            )
        else:
            exact_state = None
        try:
            if self.qite_spsa is not None:
                symbol_values, state = self.run_spsa_qite(
                    hamiltonian, exact_state, skip_metrics
                )
            else:
                symbol_values, state = self.run_exact_qite(
                    hamiltonian, exact_state, skip_metrics
                )
        except RuntimeError as e:
            return (
                tf.fill(self.symbol_values.shape, np.nan),
                tf.fill(self.state.shape, np.nan),
            )
        return symbol_values, state

    def run_spsa_qite(self, hamiltonian, exact_state=None, skip_metrics=False):

        state_list = []
        symbol_values_list = []

        for _ in trange(
            0, self.qite_spsa_repeats, desc="QITE SPSA Repeats", leave=False
        ):

            symbol_values, state, metrics = self.qite.run(
                self.circuit,
                hamiltonian.to_pauli_sum(),
                self.symbol_names,
                self.initial_symbol_values,
                self.evolution_time,
                exact_state,
                n_timesteps=self.n_timesteps,
            )
            state_list.append(state)
            symbol_values_list.append(symbol_values)
            if not skip_metrics:
                self.append_qite_metrics(metrics)

        state = tf.reduce_mean(tf.convert_to_tensor(state_list), 0)
        joint_fidelity = fidelity(exact_state, state)
        self.metrics["qite_joint_fidelity"].append(
            joint_fidelity.numpy().item()
        )
        tqdm.write(f"Joint fidelity:{joint_fidelity.numpy().item()}")

        return (
            tf.reduce_mean(tf.convert_to_tensor(symbol_values_list), 0),
            state,
        )

    def run_exact_qite(self, hamiltonian, exact_state=None, skip_metrics=False):

        symbol_values, state, metrics = self.qite.run(
            self.circuit,
            hamiltonian.to_pauli_sum(),
            self.symbol_names,
            self.initial_symbol_values,
            self.evolution_time,
            exact_state,
            n_timesteps=self.n_timesteps,
        )
        if not skip_metrics:
            self.append_qite_metrics(metrics)

        return symbol_values, state

    def init_qite_metrics(self, metrics: Dict):
        for metric, _ in metrics.items():
            self.metrics["qite_" + metric].append([])

    def append_qite_metrics(self, metrics: Dict):
        if len(self.metrics["epoch"]) > min(
            [len(self.metrics["qite_" + key]) for key in metrics.keys()]
        ):
            self.init_qite_metrics(metrics)

        for metric, values in metrics.items():
            self.metrics["qite_" + metric][-1].append(values[-1])

    def print_epoch_metrics(self):
        current_epoch = self.metrics["epoch"][-1]
        list_metrics = [
            f"{key} = {values[-1]:.3f}"
            for key, values in filter(
                lambda k: k[0] != "epoch", self.metrics.items()
            )
        ]
        string_metrics = ", ".join(list_metrics)
        entry = f"QBM train epoch {current_epoch:3}: [{string_metrics}]"
        if self.verbose == 1 and current_epoch == self.epochs - 1:
            tqdm.write(entry)
        elif self.verbose == 2 and current_epoch % 5 == 0:
            tqdm.write(entry)
        elif self.verbose == 3:
            tqdm.write(entry)

