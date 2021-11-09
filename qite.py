import cirq
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from tqdm.autonotebook import tqdm, trange
from collections import defaultdict
from utils import (
    circuit_to_state,
    quantum_relative_entropy,
    fidelity,
    trace_distance,
    get_two_variable_shift_matrix,
)

tf.get_logger().setLevel("ERROR")


class QITE:
    def __init__(
        self,
        n_extra_timesteps: int = 0,
        regularization: float = 1e-4,
        spsa: Optional[int] = None,
        spsa_perturbation: Optional[float] = 2 * np.pi / 100,
        verbose: int = 1,
        backend=None,
        differentiator=None,
    ) -> None:
        self.n_extra_timesteps = n_extra_timesteps
        self.regularization = regularization
        self.param_shift = 0.5
        self.spsa = spsa
        self.spsa_perturbation = spsa_perturbation
        self.verbose = verbose
        self.backend = backend
        self.differentiator = differentiator
        if spsa is not None and spsa > 0:
            self.bernoulli = tfp.distributions.Bernoulli(
                probs=0.5, dtype=tf.float32
            )

    def run(
        self,
        circuit: cirq.Circuit,
        hamiltonian: cirq.PauliSum,
        symbol_names: tf.Tensor,
        initial_symbol_values: tf.Tensor,
        evolution_time: float,
        solution: Optional[tf.Tensor],
        n_timesteps: int = 20,
    ) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, List[Any]]]:
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        self.metrics = defaultdict(list)
        self.qubits = sorted(circuit.all_qubits())
        self.problem_qubits = hamiltonian.qubits
        self.n_timesteps = n_timesteps

        # Calculate the step size from evolution time and number of time steps
        # In order to avoid going over the correct value +1 is used for mimicing as if there would be an extra timestep that is not run
        step_size = tf.constant(evolution_time / (self.n_timesteps + 1))

        # Initialize values
        symbol_values = initial_symbol_values
        state = circuit_to_state(
            circuit,
            symbol_names,
            symbol_values,
            list(range(len(self.problem_qubits))),
            backend=self.backend,
        )

        # Avoid calculating shifts for each time step in exact mode
        if self.spsa is None:
            self.two_variable_shifts = get_two_variable_shift_matrix(
                symbol_values.shape[0]
            )

        # Use tqdm if verbose
        iterator = (
            trange(
                1,
                self.n_timesteps + self.n_extra_timesteps + 1,
                desc="QITE",
                leave=False,
            )
            if self.verbose > 0
            else range(1, self.n_timesteps + self.n_extra_timesteps + 1)
        )

        # Calculate initial metrics
        self.metrics["time_step"].append(0)
        self.calculate_initial_energy(symbol_names, symbol_values)

        # If true solution is given calculate the initial metrics
        if solution is not None:
            if tf.math.reduce_any(
                tf.math.logical_not(tf.math.is_finite(tf.math.real(solution)))
            ):
                raise RuntimeError(f"Invalid solution")
            self.calculate_metrics(state, solution)

        # Print metrics if verbose
        if self.verbose > 0:
            self.print_step_metrics()

        # Iterate through the time stpes
        for time_step in iterator:

            self.metrics["time_step"].append(time_step)

            # Evolve symbol values
            symbol_values -= step_size * self.step(symbol_names, symbol_values)

            # Build density matrix from circuit with current values
            state = circuit_to_state(
                circuit,
                symbol_names,
                symbol_values,
                list(range(len(self.problem_qubits))),
                backend=self.backend,
            )

            # If true solution is given calculate the metrics
            if solution is not None:
                self.calculate_metrics(state, solution)

            # Print metrics if verbose
            if self.verbose > 0:
                self.print_step_metrics()

        # Return trained symbol values, final density matrix and metrics
        return symbol_values, state, self.metrics

    def step(
        self,
        symbol_names: tf.Tensor,
        symbol_values: tf.Tensor,
        tries: int = 10,
    ) -> tf.Tensor:
        gradient = self.evaluate_circuit_gradient(symbol_names, symbol_values)

        # In case information matrix is not invertable, looping and trying again
        attempt = 0
        while attempt < tries:
            attempt += 1
            if self.spsa is not None:
                information_matrix = self.evaluate_information_matrix_spsa(
                    symbol_names, symbol_values
                )
            else:
                information_matrix = self.evaluate_information_matrix_exact(
                    symbol_names, symbol_values
                )

            im_T = tf.transpose(information_matrix)
            im_im = im_T @ information_matrix
            im_g = im_T @ tf.expand_dims(gradient, 1)

            regularizer = (
                tf.eye(information_matrix.shape[0]) * self.regularization
            )
            try:
                result = tf.linalg.solve(
                    (im_im + regularizer) / (1 + self.regularization), im_g
                )
                break
            except:
                tqdm.write("Failed to solve matrix.")
                continue
        else:
            raise RuntimeError(f"Could not solve matrix in {tries} tries.")
        return tf.squeeze(result)

    def evaluate_circuit_gradient(
        self, symbol_names: tf.Tensor, symbol_values: tf.Tensor
    ) -> tf.Tensor:

        op = tfq.layers.Expectation(
            backend=self.backend, differentiator=self.differentiator
        )
        with tf.GradientTape() as g:
            g.watch(symbol_values)
            energy = op(
                self.circuit,
                operators=self.hamiltonian,
                symbol_names=symbol_names,
                symbol_values=tf.expand_dims(symbol_values, 0),
            )
        gradient = g.gradient(energy, symbol_values)
        self.metrics["energy"].append(tf.squeeze(energy).numpy().item())
        return gradient

    # https://arxiv.org/pdf/2008.06517.pdf
    def evaluate_information_matrix_exact(
        self, symbol_names: tf.Tensor, symbol_values: tf.Tensor
    ) -> tf.Tensor:

        # Calculate inner products
        product = self.calculate_overlap(
            symbol_names,
            symbol_values,
            symbol_values + self.two_variable_shifts,
        )

        # Reshape according to parameter shifts
        product = tf.reshape(
            product, [4, symbol_values.shape[0], symbol_values.shape[0]]
        )

        # Sum over the differences
        information_matrix = (
            -tf.cast(
                (product[0, :] + product[1, :] - product[2, :] - product[3, :]),
                tf.float32,
            )
            * self.param_shift
        )
        # self.metrics["norm"].append(tf.norm(information_matrix).numpy().item())
        return information_matrix

    # https://arxiv.org/pdf/2103.09232.pdf
    def evaluate_information_matrix_spsa(
        self, symbol_names: tf.Tensor, symbol_values: tf.Tensor
    ) -> tf.Tensor:

        # Sample two random directions [-1, 1]^s
        directions = (
            self.bernoulli.sample([self.spsa, 2, symbol_values.shape[0]]) * 2
            - 1
        )

        # Build basis from the directions
        basis = (
            tf.matmul(
                tf.transpose(directions, [0, 2, 1]),
                tf.transpose(directions[:, ::-1], [0, 1, 2]),
            )
            / 2
        )
        # Perturb the symbol values according to 2-spsa
        symbol_values_perturbations = (
            tf.concat(
                [
                    directions[:, 0, :] + directions[:, 1, :],
                    -directions[:, 0, :],
                    -directions[:, 0, :] + directions[:, 1, :],
                    directions[:, 0, :],
                ],
                0,
            )
            * self.spsa_perturbation
        )

        # Calculate inner products
        product = self.calculate_overlap(
            symbol_names,
            symbol_values,
            symbol_values + symbol_values_perturbations,
        )

        # Reshape according to perturbations shifts
        product = tf.reshape(product, [4, self.spsa, -1])

        # Sum over the differences
        difference = (
            -2
            * tf.cast(
                (product[0, :] - product[1, :] - product[2, :] + product[3, :]),
                tf.float32,
            )
            / (self.spsa_perturbation ** 2)
        )
        information_matrix = basis * tf.expand_dims(difference, 1)

        # Reduce by mean over all SPSA estimates
        information_matrix = tf.reduce_mean(information_matrix, 0)

        information_matrix = tf.matrix_square_root(
            tf.matmul(information_matrix, information_matrix)
        )
        # self.metrics["norm"].append(tf.norm(information_matrix).numpy().item())
        return information_matrix

    def calculate_overlap(
        self,
        symbol_names: tf.Tensor,
        symbol_values: tf.Tensor,
        symbol_values_overlap: tf.Tensor,
    ) -> tf.Tensor:
        op = tfq.layers.State(backend=self.backend)
        fixed_state = op(
            self.circuit,
            symbol_names=symbol_names,
            symbol_values=tf.expand_dims(symbol_values, 0),
        )
        diff_states = op(
            self.circuit,
            symbol_names=symbol_names,
            symbol_values=symbol_values_overlap,
        )
        # Calculate inner products
        product = diff_states.to_tensor() @ tf.transpose(
            fixed_state.to_tensor()
        )
        return tf.square(tf.abs(product))

    def calculate_metrics(self, state: tf.Tensor, solution: tf.Tensor):
        fidel = fidelity(solution, state)
        relative_entropy = quantum_relative_entropy(solution, state)
        trace_dist = trace_distance(solution, state)
        self.metrics["fidelity"].append(fidel.numpy().item())
        self.metrics["quantum_relative_entropy"].append(
            relative_entropy.numpy().item()
        )
        self.metrics["trace_distance"].append(trace_dist.numpy().item())

    def calculate_initial_energy(
        self, symbol_names: tf.Tensor, symbol_values: tf.Tensor
    ):
        op = tfq.layers.Expectation(
            backend=self.backend, differentiator=self.differentiator
        )
        energy = op(
            self.circuit,
            operators=self.hamiltonian,
            symbol_names=symbol_names,
            symbol_values=tf.expand_dims(symbol_values, 0),
        )
        self.metrics["energy"].append(tf.squeeze(energy).numpy().item())

    def print_step_metrics(self):
        current_time_step = self.metrics["time_step"][-1]
        list_metrics = [
            f"{key} = {values[-1]:.3f}"
            for key, values in filter(
                lambda k: k[0] != "time_step", self.metrics.items()
            )
        ]
        string_metrics = ", ".join(list_metrics)
        entry = f"QITE timestep {current_time_step:3}: [{string_metrics}]"
        if self.verbose == 1 and current_time_step == self.n_timesteps:
            tqdm.write(entry)
        elif self.verbose == 2 and current_time_step % 5 == 0:
            tqdm.write(entry)
        elif self.verbose == 3:
            tqdm.write(entry)
