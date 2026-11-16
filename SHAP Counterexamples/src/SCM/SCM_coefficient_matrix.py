### ---------------------------------------------------------------------------------------
### Compute the SCM coefficient matrix for a given graph, features, and target node.
### Uses frozenset() inorder to make the coalitions hashable and usable as dictionary keys.
### ---------------------------------------------------------------------------------------

from itertools import combinations
from math import comb

import numpy as np


def coalition_basis(graph, coalition, target):
    """Graph Exploration to find the basis of a coalition with respect to a target node"""
    coalition = set(coalition)
    basis = set()
    visited = set()
    stack = [target]

    while stack:
        name = stack.pop()
        if name in visited:
            continue
        visited.add(name)

        if name in coalition:
            basis.add(name)
        else:
            stack.extend(graph.nodes[name].parents)

    return frozenset(basis)


def scm_coefficient_matrix(graph, features, target):
    """
    Compute the SCM coefficient matrix for a given graph, features, and target node.
    """
    features = tuple(features)
    n = len(features)

    ### Compute every coalition as a frozenset in order to hash it in the "reduced" dictionary.
    coalitions = [
        frozenset(S)
        for size in range(n + 1)
        for S in combinations(features, size)
    ]

    ### Reduced dictionary for efficient lookup of coalition basis with respect to the target node.
    reduced = {S: coalition_basis(graph, S, target) for S in coalitions}
    bases = list(dict.fromkeys(reduced.values()))

    columns = {basis: i for i, basis in enumerate(bases)}
    W = np.zeros((n, len(bases)))

    for i, feature in enumerate(features):
        for S in coalitions:
            if feature in S:
                continue
            weight = 1 / (n * comb(n - 1, len(S)))
            ### Weight Compuattion based on the Shapley value formula. The weight is distributed to the coalition basis with and without the feature.
            W[i, columns[reduced[S | {feature}]]] += weight
            W[i, columns[reduced[S]]] -= weight

    labels = [tuple(name for name in features if name in basis) for basis in bases]
    return W, labels
