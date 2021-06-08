from typing import Callable, List, Tuple
import cirq
import numpy as np
import sympy
import tensorflow as tf


def rot_layer(
    qubits: List[cirq.GridQubit],
    theta: List[sympy.Symbol],
    rotations: List[Callable] = [cirq.rx, cirq.ry, cirq.rz],
):
    for i, rot in enumerate(rotations):
        yield cirq.Moment([rot(theta[j, i])(q) for j, q in enumerate(qubits)])


def cnot_entangler_layer(
    qubits: List[cirq.GridQubit],
    phi: List[sympy.Symbol],
    pattern: str = "chain",
):
    entanglement_map = [((i + 1) % len(qubits), i) for i in range(len(qubits))]
    if pattern == "ring":
        for i, j in entanglement_map:
            yield cirq.CNOT(qubits[i], qubits[j]) ** phi[j]
    elif pattern == "chain":
        for i, j in entanglement_map[:-1]:
            yield cirq.CNOT(qubits[i], qubits[j]) ** phi[j]
    elif pattern == "pairs":
        for i, j in entanglement_map[::2] + entanglement_map[1:-1:2]:
            yield cirq.CNOT(qubits[i], qubits[j]) ** phi[j]


def rot_entangler_layer(
    qubits: List[cirq.GridQubit],
    phi: List[sympy.Symbol],
    pattern: str = "chain",
    rotations: List[Callable] = [cirq.XX, cirq.YY, cirq.ZZ],
):
    entanglement_map = [((i + 1) % len(qubits), i) for i in range(len(qubits))]
    for r, rot in enumerate(rotations):
        if pattern == "ring":
            for i, j in entanglement_map:
                yield rot(qubits[i], qubits[j]) ** phi[j, r]
        elif pattern == "chain":
            for i, j in entanglement_map[:-1]:
                yield rot(qubits[i], qubits[j]) ** phi[j, r]
        elif pattern == "pairs":
            for i, j in entanglement_map[::2] + entanglement_map[1::2]:
                yield rot(qubits[i], qubits[j]) ** phi[j, r]


# Used for preparing inital mixed state
def cnot_half_entangler_layer(qubits: List[cirq.GridQubit]):
    half = int(len(qubits) / 2)
    entanglement_map = [(i, i + half) for i in range(half)]
    for i, j in entanglement_map:
        yield cirq.CNOT(qubits[i], qubits[j])


def build_ansatz(
    qubits: List[cirq.GridQubit],
    n_layers: int = 2,
    rotations: List[Callable] = [cirq.ry, cirq.rz],
    entanglers: List[cirq.Gate] = [cirq.CNOT],
    pattern: str = "chain",
) -> Tuple[cirq.Circuit, tf.Tensor]:
    circuit = cirq.Circuit()
    n_qubits = int(len(qubits) / 2)

    theta = sympy.symbols(
        [f"theta_(0:{n_layers+1})(0:{n_qubits*2})(0:{len(rotations)})"]
    )
    theta = np.array(theta).reshape(n_layers + 1, n_qubits * 2, len(rotations))

    n_phi = n_qubits * 2 - 1 if pattern == "chain" else n_qubits * 2
    phi = sympy.symbols([f"phi_(0:{n_layers})(0:{n_phi})(0:{len(entanglers)})"])
    phi = np.array(phi).reshape(n_layers, n_phi, len(entanglers))

    strategy = cirq.circuits.InsertStrategy.NEW_THEN_INLINE
    for i in range(n_layers):
        circuit.append(rot_layer(qubits, theta[i], rotations))
        circuit.append(
            rot_entangler_layer(qubits, phi[i], pattern, entanglers), strategy
        )

    circuit.append(rot_layer(qubits, theta[n_layers], rotations))
    circuit.append(cnot_half_entangler_layer(qubits), strategy)

    symbol_names = tf.convert_to_tensor(
        [t.name for t in theta.flatten()] + [p.name for p in phi.flatten()]
    )
    return circuit, symbol_names


def initialize_ansatz_symbols(
    n_qubits: int,
    n_layers: int = 2,
    n_rotations: int = 2,
    rot_i: int = 0,
    rot_angle: float = np.pi / 2,
    n_entanglers: int = 1,
    pattern: str = "chain",
):
    rot_values = np.zeros((n_layers + 1, n_qubits, n_rotations))
    rot_values[n_layers, : (n_qubits // 2), rot_i] = rot_angle
    rot_values = tf.constant(rot_values.flatten(), dtype=tf.float32)

    n_phi = n_qubits - 1 if pattern == "chain" else n_qubits
    entangler_values = np.ones((n_layers, n_phi, n_entanglers))
    entangler_values = tf.constant(entangler_values.flatten(), dtype=tf.float32)
    return tf.concat([rot_values, entangler_values], 0)
