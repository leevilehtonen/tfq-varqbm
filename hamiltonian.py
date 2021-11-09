from typing import Optional
import cirq
import tensorflow as tf


class Hamiltonian:
    def __init__(
        self, problem: cirq.PauliSum, coefficients: Optional[tf.Tensor] = None,
    ) -> None:
        self._problem = problem
        self.pauli_terms = list(problem)
        if coefficients is not None:
            self._coefficients = tf.Variable(coefficients)
        else:
            self._coefficients = tf.Variable(
                [term.coefficient.real for term in self.pauli_terms]
            )

    def to_pauli_sum(self) -> cirq.PauliSum:
        return cirq.PauliSum.from_pauli_strings(
            [
                pauli_term.with_coefficient(self._coefficients[i])
                for i, pauli_term in enumerate(self.pauli_terms)
            ]
        )

    @property
    def coefficients(self) -> tf.Tensor:
        return self._coefficients

    @property
    def problem(self) -> tf.Tensor:
        return self._problem

    @coefficients.setter
    def coefficients(self, new_coefficients: tf.Tensor):
        self._coefficients = new_coefficients

    def normalize_coefficients(self, clip=False):
        if clip:
            clipped_coefficients = tf.clip_by_value(
                self._coefficients, clip_value_min=-10, clip_value_max=10
            )
            self._coefficients = tf.Variable(clipped_coefficients)
        else:
            normalized_coefficients, _ = tf.linalg.normalize(self._coefficients)
            self._coefficients = tf.Variable(normalized_coefficients)

    def finite_difference_models(self, difference=0.1):
        n = self.coefficients.shape[0]
        finite_difference_coefficients = (
            tf.tile(tf.expand_dims(self.coefficients, 0), [n, 1])
            + tf.eye(n) * difference
        )
        return [
            Hamiltonian(self._problem, finite_difference_coefficients[i])
            for i in range(n)
        ]

    def __str__(self) -> str:
        return self.to_pauli_sum().__str__()
