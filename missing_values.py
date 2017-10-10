#!/usr/bin/env python3
"""Implementation of technique for the replacement of missing values."""


import numpy as np
import random
import data_reader as dr
import copy
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")


def replace_by_mean(alternatives):
    """Replace missing evaluations by the mean of the same criteria."""
    T_alternatives = list(map(list, zip(*alternatives)))
    T_filled_alternatives = []
    for crit_evalutations in T_alternatives:
        copy = [j for j in crit_evalutations if j is not '*']
        mean = np.mean(copy)
        new_crit_evaluations = [j if j != '*' else mean
                                for j in crit_evalutations]
        T_filled_alternatives.append(new_crit_evaluations)
    filled_alternatives = list(map(list, zip(*T_filled_alternatives)))
    return filled_alternatives


def replace_by_median(alternatives):
    """Replace missing evaluations by the mean of the same criteria."""
    T_alternatives = list(map(list, zip(*alternatives)))
    T_filled_alternatives = []
    for crit_evalutations in T_alternatives:
        copy = [j for j in crit_evalutations if j is not '*']
        mean = np.median(copy)
        new_crit_evaluations = [j if j != '*' else mean
                                for j in crit_evalutations]
        T_filled_alternatives.append(new_crit_evaluations)
    filled_alternatives = list(map(list, zip(*T_filled_alternatives)))
    return filled_alternatives


def replace_by_neighboors(alts_p, k=1):
    """Replace the missing value by the mean of the k-nearest neighboor."""
    # for alt in alts_p:
    #     print(alt)

    # if alternative missing replace all evaluations by mean or median!
    for alt in alts_p:
        if alt == ['*' for i in range(len(alt))]:
            for crit in range(len(alt)):
                evs = [alts_p[j][crit] for j in range(len(alts_p))
                       if alts_p[j][crit] != '*']
                mean = sum(evs)/len(evs)
                alt[crit] = mean

    # for alt in alts_p:
    #     print(alt)

    alts = copy.deepcopy(alts_p)
    for pivot_crit in range(len(alts[0])):
        for pivot_alt in range(len(alts)):
            if alts[pivot_alt][pivot_crit] != '*':
                continue
            pivot_indices = [i for i in range(len(alts[pivot_alt]))
                             if alts[pivot_alt][i] != '*']
            pivot_values = [alts[pivot_alt][j] for j in pivot_indices]

            data = []
            data_indices = []
            for j in range(len(alts)):
                evals = [alts[j][i] for i in pivot_indices]
                if '*' not in evals and alts[j][pivot_crit] != '*':
                    data.append(evals)
                    data_indices.append(j)

            data.append(pivot_values)
            data = np.array(data)
            data = normalize(data, axis=0, copy=True, norm='max')

            pivot_values = data[-1]
            data = data[:-1]
            # print(data)
            # print(pivot_values)

            nnbrg = NearestNeighbors(n_neighbors=k).fit(data)
            distances, n_indices = nnbrg.kneighbors(pivot_values)
            # print(distances)
            # print(n_indices)
            final_indices = [data_indices[i] for i in n_indices[0]]
            # print(final_indices)
            missing_eval = sum([alts[j][pivot_crit] for j in final_indices])/k
            alts[pivot_alt][pivot_crit] = missing_eval
    return alts


def delete_evaluations(alternatives, proportion, seed=0):
    """Delete 'proportion' of the alternatives evaluations."""
    random.seed(seed)
    for alt in alternatives:
        for i in range(len(alt)):
            if random.randint(0, 100) < proportion*100:
                alt[i] = '*'

if __name__ == '__main__':
    random.seed(5)
    data_set = 'data/HP/raw.csv'
    alts = dr.open_raw(data_set)[0][:5]
    alts = [[2, 2, 2, 2, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 10],
            [2, 2, 2, 2, 20], [4, 4, 4, 4, 40]]
    # proportion = 0.01
    # delete_evaluations(alts, proportion)
    alts2 = copy.deepcopy(alts)
    alts[0][1] = '*'
    print('init :')
    for alt in alts2:
        print(alt)

    print()
    print('gapped :')
    for alt in alts:
        print(alt)

    alts = replace_by_neighboors(alts, 2)

    print()
    print('modified:')
    for alt in alts:
        print(alt)
