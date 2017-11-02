"""K nearest neighboors techniques."""

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import data_reader as dr
from helpers import NULL
import helpers

import copy
import numpy as np
import random


def knn_mse(alts, c, k=1):
    """Compute MSE of the knn."""
    MSE = 0
    n = len(alts)
    for i in range(n):
        ev = alts[i][c]
        alts[i][c] = NULL
        alts = replace_by_knn(alts, k)
        new_ev = alts[i][c]
        alts[i][c] = ev
        MSE += (ev - new_ev)**2
    return MSE/n


def replace_by_knn(A, k=1):
    """Replace the missing value by the mean of the k-nearest neighboor."""
    incompletes = [alt for alt in A if NULL in alt]
    complete_alts = [alt for alt in A if NULL not in alt]

    completed_alts = copy.deepcopy(A)
    for incomplete in incompletes:
        i = A.index(incomplete)
        criteria = [k for k, x in enumerate(incomplete) if x == NULL]
        for c in criteria:
            evaluation = estimate_by_knn(complete_alts, c, incomplete, k)
            completed_alts[i][c] = evaluation
    return completed_alts


def get_knn(A, k=1):
    """Return the estimation comupted with the knn."""
    incomplete = [alt for alt in A if NULL in alt][0]
    complete_alts = [alt for alt in A if NULL not in alt]
    c = incomplete.index(NULL)

    evaluation = estimate_by_knn(complete_alts, c, incomplete, k)
    return evaluation


def estimate_by_knn(completes, c, incomplete, k=1):
    """Estimate the missing value by the mean of the k-nearest neighboor.

    old code :
    # for alt in alts_p:
    #     if alt == [NULL for i in range(len(alt))]:
    #         for crit in range(len(alt)):
    #             evs = [alts_p[j][crit] for j in range(len(alts_p))
    #                    if alts_p[j][crit] != NULL]
    #             mean = sum(evs)/len(evs)
    #             alt[crit] = mean

    alts = copy.deepcopy(alts_p)
    for pivot_crit in range(len(alts[0])):
        for pivot_alt in range(len(alts)):
            if alts[pivot_alt][pivot_crit] != NULL:
                continue
            pivot_indices = [i for i in range(len(alts[pivot_alt]))
                             if alts[pivot_alt][i] != NULL]
            pivot_values = [alts[pivot_alt][j] for j in pivot_indices]

            data = []
            data_indices = []
            for j in range(len(alts)):
                evals = [alts[j][i] for i in pivot_indices]
                if NULL not in evals and alts[j][pivot_crit] != NULL:
                    data.append(evals)
                    data_indices.append(j)

            data.append(pivot_values)
            data = np.array(data)
            data = normalize(data, axis=0, copy=True, norm='max')

            pivot_values = data[-1]
            data = data[:-1]

            nnbrg = NearestNeighbors(n_neighbors=k).fit(data)
            distances, n_indices = nnbrg.kneighbors(pivot_values)
            final_indices = [data_indices[i] for i in n_indices[0]]
            # print(final_indices)
            estimation = sum([alts[j][pivot_crit] for j in final_indices])/k
            # print('knn replacement:', estimation)
            alts[pivot_alt][pivot_crit] = estimation
    return alts
    """
    # helpers.printmatrix(completes)
    # print(incomplete)

    data = [[a[i] for i in range(len(completes[0])) if i != c]
            for a in completes]
    data.append([incomplete[i] for i in range(len(incomplete)) if i != c])

    data = np.array(data)
    data = normalize(data, axis=0, copy=True, norm='max')

    target = data[-1]
    data = data[:-1]

    nnbrg = NearestNeighbors(n_neighbors=k).fit(data)
    distances, indices = nnbrg.kneighbors(target)
    estimation = sum([completes[j][c] for j in indices])/k
    return estimation
