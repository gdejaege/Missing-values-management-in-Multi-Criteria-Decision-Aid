"""Plot the missing values estimations for dataset X, 1/X."""

from helpers import NULL
import helpers
import regression as rg
import local_regression as lrg
import dominance_estimations as de
import knn
import missing_values as mv
import promethee as prom
import data_reader as dr

import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random
import copy
import time


def plot(A, points, ranges=None, normalised=True):
    """Plot the alternatives, the points in the given ranges."""
    symbols = ['r-', 'bv', 'g*', 'ks', 'mo', 'ro']
    symb = iter(symbols)
    A_1, A_2 = zip(*sorted(A))
    plt.plot(A_1, A_2, next(symb), label='evaluations')

    for meth in points:
        x, y = zip(*points[meth])
        plt.plot(x, y, next(symb), label=meth)

    plt.legend()
    if normalised:
        plt.axis([0, 1, -0.5, 1])
    plt.show()


def get_evaluations(A_miss, methods):
    """Compare strategies once.

    A = alternatives with one NULL.
    """
    evaluations = {}
    for method in methods:
        evaluation = methods[method](A_miss)
        evaluations[method] = evaluation

    return evaluations


def main(dataset='HDI_20', alt_num=100, iterations=30,
         methods=['reg', 'lrg']):
    """Compare strategies on a fake dataset.

    Plot evaluations of a 2D simulation (1/x^2).

    Inputs:
        methods = methods to compare and plot
    """
    # data_file = 'data/' + dataset + '/noise_raw.csv'
    # out_file = 'res/method_plot/' + dataset + '.csv'
    methods_functs = {'reg': rg.get_regression,
                      'lrg': lrg.get_estimation_by_local_regression,
                      'dom': de.get_estimations_by_dominance,
                      'diff': de.get_estimations_by_dominance_diff,
                      'knn': knn.get_knn,
                      'mean': mv.get_mean,
                      'med': mv.get_med}

    methods = {meth: methods_functs[meth] for meth in methods}

    t0 = time.time()

    normalised = True
    normalised = False
    A = helpers.get_dataset(dataset, alt_num, normalised=normalised)

    # Each method will have a list of points
    res = {meth: [] for meth in methods}

    step = alt_num // iterations
    # print(step)
    for it in range(0, alt_num, step):
        # print(it)
        res_it = []
        i = random.randint(0, len(A)-1)
        i = it

        # Forcé !!
        c = 1

        ev = A[i][c]
        target = A[i][0]
        A[i][c] = NULL
        evaluations = get_evaluations(A, methods)
        A[i][c] = ev

        for m in evaluations:
            res[m].append((target, evaluations[m]))

    plot(A, res, normalised=normalised)
