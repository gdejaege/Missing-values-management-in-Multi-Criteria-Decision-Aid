"""Implementation of the analysis of layer-wise regression.

The purpose of this code is, given a dataset, to check whether there is always a
dominance layer that will, using a regression, estimate the correct missing
value.

"""

import regression as reg
import numpy as np
import copy
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from helpers import NULL
import helpers
from scipy import stats
from sklearn.metrics import mean_squared_error as sk_mean_squared_error


def layer_regression_all(A):
    """Implementation of the analysis of layer-wise regression.

    Returns a list of estimation for each of the layers.
    """
    incomplete = [alt for alt in A if NULL in alt][0]
    complete_alts = [alt for alt in A if NULL not in alt]
    c = incomplete.index(NULL)

    layers = compute_layers(complete_alts)

    res = []
    for layer in layers:
        dataset = layer + [incomplete]
        estimation = reg.get_regression(dataset)
        # helpers.printmatrix(dataset)
        # print(":: ", estimation, "\n")
        # helpers.printmatrix(dataset)
        res.append(float(estimation))
    return res


def layer_regression_guess_layer(A):
    """Implementation of the analysis of layer-wise regression.

    Returns a list of estimation for each of the layers.
    """
    incomplete = [alt for alt in A if NULL in alt][0]
    complete_alts = [alt for alt in A if NULL not in alt]
    c = incomplete.index(NULL)
    # helpers.printmatrix(complete_alts)

    layers = compute_layers(complete_alts[:])

    res = None
    for layer in layers:
        # helpers.printmatrix(layer)
        if res is None and pareto_equivalent(incomplete, layer):
            # print("here")
            # helpers.printmatrix(incomplete)
            res = float(reg.get_regression(layer + [incomplete]))

    # print(complete_alts)
    if res is None:
        res = min([alt[c] for alt in complete_alts])
    return res


def pareto_equivalent(a, A):
    """Check if alternative a is pareto equivalent to all alternatives in A."""
    res = True
    for dom in A:
        if dominates(a, dom):
            res = False
    return res


def compute_layers(A):
    """Compute the layers."""
    A.sort(reverse=True)
    layers = []
    layer = []
    while (len(A) > 0):
        i = 0
        while (i < len(A)):
            a = A[i]
            dominated = False
            for dominant in layer:
                if dominates(a, dominant):
                    dominated = True

            if not dominated:
                layer.append(A.pop(i))
            else:
                i += 1
        layers.append(layer)
        layer = []
    return layers


def dominates(dominated, dominant):
    """Check whether these alternatives are really dominated.

    If an evaluation is missing on one criterion, it is note taken into account.
    """
    res = True
    for ev1, ev2 in zip(dominated, dominant):
        # print(ev1, ev2)
        if res and ev1 != NULL and ev2 != NULL and ev1 > ev2:
            res = False
    return res
