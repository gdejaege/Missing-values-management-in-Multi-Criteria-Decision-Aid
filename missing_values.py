#!/usr/bin/env python3
"""Implementation of technique for the replacement of missing values."""


import random
import data_reader as dr
import copy
from sklearn.neighbors import NearestNeighbors
import numpy as np


def replace_by_mean(alternatives):
    """Replace missing evaluations by the mean of the same criteria."""
    T_alternatives = list(map(list, zip(*alternatives)))
    T_filled_alternatives = []
    for crit_evalutations in T_alternatives:
        copy = [j for j in crit_evalutations if j is not -1]
        mean = np.mean(copy)
        new_crit_evaluations = [j if j != -1 else mean
                                for j in crit_evalutations]
        T_filled_alternatives.append(new_crit_evaluations)
    filled_alternatives = list(map(list, zip(*T_filled_alternatives)))
    return filled_alternatives


def replace_by_median(alternatives):
    """Replace missing evaluations by the mean of the same criteria."""
    T_alternatives = list(map(list, zip(*alternatives)))
    T_filled_alternatives = []
    for crit_evalutations in T_alternatives:
        copy = [j for j in crit_evalutations if j is not -1]
        mean = np.median(copy)
        new_crit_evaluations = [j if j != -1 else mean
                                for j in crit_evalutations]
        T_filled_alternatives.append(new_crit_evaluations)
    filled_alternatives = list(map(list, zip(*T_filled_alternatives)))
    return filled_alternatives


def replace_by_neighboors(alts_p, k=1):
    """Replace the missing value by the mean of the k-nearest neighboor."""
    alts = copy.deepcopy(alts_p)
    for pivot_crit in range(len(alts[0])):
        for pivot_alt in range(len(alts)):
            if alts[pivot_alt][pivot_crit] != -1:
                continue
            pivot_indices = [i for i in range(len(alts[pivot_alt]))
                             if alts[pivot_alt][i] != -1]
            pivot_values = [alts[pivot_alt][j] for j in pivot_indices]

            data = []
            data_indices = []
            for j in range(len(alts)):
                evals = [alts[j][i] for i in pivot_indices]
                if -1 not in evals and alts[j][pivot_crit] != -1:
                    data.append(evals)
                    data_indices.append(j)

            # print(data)
            data = np.array(data)
            # print(data)
            # print(pivot_values)
            pivot_values = np.array(pivot_values).reshape(1, -1)
            # print(pivot_values)
            nnbrg = NearestNeighbors(n_neighbors=k).fit(data)
            distances, n_indices = nnbrg.kneighbors(pivot_values)
            final_indices = [data_indices[i] for i in n_indices[0]]
            missing_eval = sum([alts[j][pivot_crit] for j in final_indices])/k
            alts[pivot_alt][pivot_crit] = missing_eval
    return alts


# [[alt[j] for j in indices_needed] for alt in alts]


def delete_evaluations(alternatives, proportion, seed=0):
    """Delete 'proportion' of the alternatives evaluations."""
    random.seed(seed)
    for alt in alternatives:
        for i in range(len(alt)):
            if random.randint(0, 100) < proportion*100:
                alt[i] = -1

if __name__ == '__main__':
    random.seed(5)
    data_set = 'data/SHA/raw.csv'
    alts = dr.open_raw(data_set)[0][:5]
    proportion = 0.001
    # delete_evaluations(alts, proportion)
    alts[1][1] = -1
    alts2 = copy.deepcopy(alts)
    for alt in alts:
        print(alt)
    replace_by_neighboors(alts2, 2)
    for alt in alts2:
        print(alt)
