from typing import List, Optional
import networkx as nx
import cirq
import numpy as np


def build_ising_model_hamiltonian(
    connectivity: List[int],
    coupling: int = 1,
    external: int = 1,
    transverse: Optional[int] = 1,
    random: bool = False,
    random_high: float = 1,
    random_low: float = -1,
):

    # Interaction either as complete graph (between all) or bipartite
    if len(connectivity) not in [1, 2]:
        raise NotImplementedError

    interaction_graph = None
    if len(connectivity) == 1:
        interaction_graph = nx.complete_graph(connectivity[0])
    else:
        interaction_graph = nx.complete_bipartite_graph(
            connectivity[0], connectivity[1]
        )

    # Build the hamiltonian
    hamiltonian = 0

    # Two qubit couplings
    for node1, node2 in interaction_graph.edges:
        c = (
            np.random.uniform(low=random_low, high=random_high)
            if random
            else coupling
        )
        hamiltonian += (
            c
            * cirq.Z(cirq.GridQubit(node1, 0))
            * cirq.Z(cirq.GridQubit(node2, 0))
        )

    # One qubit interaction
    for node in interaction_graph.nodes:
        e = (
            np.random.uniform(low=random_low, high=random_high)
            if random
            else external
        )
        hamiltonian += e * cirq.Z(cirq.GridQubit(node, 0))
        t = (
            np.random.uniform(low=random_low, high=random_high)
            if random
            else transverse
        )
        if transverse is not None:
            hamiltonian += t * cirq.X(cirq.GridQubit(node, 0))

    all_qubits = [cirq.GridQubit(node, 0) for node in interaction_graph.nodes]
    return hamiltonian, sorted(all_qubits)


def build_heisenberg_model_hamiltonian(
    connectivity: List[int],
    variant: str = "XYZ",
    coupling: List[int] = [1, 1, 1],
    external: int = 1,
    random: bool = False,
):

    # Interaction either as chain or 2d lattice
    if len(connectivity) not in [1, 2]:
        raise NotImplementedError

    interaction_graph = None
    if len(connectivity) == 1:
        interaction_graph = nx.path_graph(connectivity[0])
    else:
        interaction_graph = nx.grid_2d_graph(connectivity[0], connectivity[1])

    supported_variants = ["XYZ", "XXZ", "XXX"]
    if variant not in supported_variants:
        raise NotImplementedError

    # Initialize from given couplings or random
    J_x, J_y, J_z = coupling
    if random:
        J_x, J_y, J_z = np.random.uniform(size=[3], low=-1, high=1).tolist()

    # Depending on the variant restrict coupling
    if variant == "XXZ":
        J_y = J_x
    elif variant == "XXX":
        J_y = J_x
        J_z = J_x

    # Build the hamiltonian
    hamiltonian = 0

    # Two qubit couplings
    for node1, node2 in interaction_graph.edges:
        qubit1, qubit2 = cirq.GridQubit(*node1), cirq.GridQubit(*node2)
        hamiltonian += J_x * cirq.X(qubit1) * cirq.X(qubit2)
        hamiltonian += J_y * cirq.Y(qubit1) * cirq.Y(qubit2)
        hamiltonian += J_z * cirq.Z(qubit1) * cirq.Z(qubit2)

    # One qubit interaction
    if external != 0 or random == True:
        if random == True:
            external = np.random.uniform(low=-1, high=1)
        for node in interaction_graph.nodes:
            qubit = cirq.GridQubit(*node)
            hamiltonian += external * cirq.Z(qubit)

    all_qubits = [cirq.GridQubit(*node) for node in interaction_graph.nodes]
    return hamiltonian, sorted(all_qubits)
