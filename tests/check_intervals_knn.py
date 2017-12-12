"""Tests on the interval for knn.

The idea is to check whether we can obtain a good but not to big interval by
selecting the min and max evaluations on c of the k nearest neighboors.
This interval could then be used for multiple imputations in PII.
"""

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

import data_reader as dr
from helpers import NULL
import helpers

import copy
import numpy as np
import random

PLOT_FORMAT = 320
RANGE = range(5, 16, 2)


def get_interval(A, a_miss, c, neigh_number):
    """Return the max interval on c of the neighboors evaluations.

    A is the set of all alternatives WITHOUT a_miss.
    """
    n = len(A)
    k = len(A[0])

    # Nearest but of course not on the missing criteria
    data = [[a[i] for i in range(k) if i != c] for a in A]
    data.append([a_miss[i] for i in range(k) if i != c])

    data = np.array(data)

    target = data[-1]
    data = data[:-1]

    nnb_model = NearestNeighbors(n_neighbors=neigh_number).fit(data)
    distances, indices = nnb_model.kneighbors(target)
    neighboors = [A[j] for j in indices[0]]
    neighboors_ev_c = [n[c] for n in neighboors]
    return (min(neighboors_ev_c), max(neighboors_ev_c))


def plot(neighboor_res_list, dataset, neigh_numbs, deltas):
    """Plot the deltas."""
    # helpers.printmatrix(neighboor_res_list)
    neigh_numbs = list(neigh_numbs)

    # fig = plt.figure(1, figsize=(16, 12), dpi=880)
    fig = plt.figure(1)
    # ax = fig.gca()
    # ax.set_autoscale_on(False)
    title = dataset
    save_title = "res/knn_deltas/" + dataset + '.pdf'
    fig.canvas.set_window_title(title)

    for i, neigh_numb in enumerate(neigh_numbs):
        plt.subplot(PLOT_FORMAT + i + 1)
        values = neighboor_res_list[i]

        n, bins, patches = plt.hist(values, normed=1, bins=50,
                                    facecolor='green', alpha=0.5)
        plt.title(str(neigh_numb) + ' :: ' + str(deltas[i]))
        plt.axis([-1.5, 2.5, 0, 2])     # [xmin,xmax,ymin,ymax]
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=None, hspace=0.8)
        # plt.savefig(save_title, bbox_inches='tight')
    plt.show()
    plt.close()


def main(MAXIT=1000):
    """Check the intervals."""
    datasets = ("SHA",)
    n = 100

    for dataset in datasets:
        A = helpers.get_dataset(dataset, n)
        k = len(A[0])

        # list contains on list with each res = (ev - mn)/(mx - mn)
        neighboor_res_list = []
        neighboor_deltas_list = []

        # for neigh_number in range(2, 10):
        neigh_numbs = RANGE
        for neigh_number in neigh_numbs:
            res_list = []
            deltas = []
            for it in range(MAXIT):
                i, c = random.randint(0, n - 1), random.randint(0, k - 1)
                a_miss = A[i]
                ev = a_miss[c]
                a_miss[c] = NULL
                del A[i]
                mn, mx = get_interval(A, a_miss, c, neigh_number)
                deltas.append(mx - mn)
                res = (ev - mn)/(mx - mn)
                res_list.append(res)
                a_miss[c] = ev
                A.insert(i, a_miss)

            neighboor_res_list.append(res_list)
            neighboor_deltas_list.append(np.mean(deltas))

        plot(neighboor_res_list, dataset, neigh_numbs, neighboor_deltas_list)
