#!/usr/bin/env python3
"""Implementation of technique for the replacement of missing values."""


import data_reader as dr
import helpers
import copy
import numpy as np
import random
import time
from sklearn.preprocessing import scale
import warnings
warnings.filterwarnings("ignore")

NULL = '*'


def replace_by_mean(alternatives):
    """Replace missing evaluations by the mean of the same criteria."""
    T_alternatives = list(map(list, zip(*alternatives)))
    T_filled_alternatives = []
    for crit_evaluations in T_alternatives:
        copy = [j for j in crit_evaluations if j is not NULL]
        estimation = np.mean(copy)
        # if NULL in crit_evaluations:
        #     print('mean replacement:', estimation)
        new_crit_evaluations = [j if j != NULL else estimation
                                for j in crit_evaluations]
        T_filled_alternatives.append(new_crit_evaluations)
    filled_alternatives = list(map(list, zip(*T_filled_alternatives)))
    return filled_alternatives


def get_mean(A):
    """Find the mean estimation."""
    incomplete = [alt for alt in A if NULL in alt][0]
    c = incomplete.index(NULL)
    evs_c = [a[c] for a in A if a[c] != NULL]
    return np.mean(evs_c)


def get_med(A):
    """Find the med estimation."""
    incomplete = [alt for alt in A if NULL in alt][0]
    c = incomplete.index(NULL)
    evs_c = [a[c] for a in A if a[c] != NULL]
    return np.median(evs_c)


def med_mse(evaluations):
    """Compute MSE of the med."""
    MSE = 0
    for i in range(len(evaluations)):
        ev = evaluations[i]
        other_evs = evaluations[:i] + evaluations[i+1:]
        MSE += (ev - np.median(other_evs))**2
    return MSE/len(evaluations)


def replace_by_med(alternatives):
    """Replace missing evaluations by the mean of the same criteria."""
    T_alternatives = list(map(list, zip(*alternatives)))
    T_filled_alternatives = []
    for crit_evaluations in T_alternatives:
        copy = [j for j in crit_evaluations if j is not NULL]
        estimation = np.median(copy)
        # if NULL in crit_evaluations:
        #     print('med replacement:', estimation)
        new_crit_evaluations = [j if j != NULL else estimation
                                for j in crit_evaluations]
        T_filled_alternatives.append(new_crit_evaluations)
    filled_alternatives = list(map(list, zip(*T_filled_alternatives)))
    return filled_alternatives


if __name__ == '__main__':
    V = True
    # V = False
    # random.seed(5)
    data_set = 'data/SHA/raw.csv'
    # alts = dr.open_raw(data_set)[0][:30]
    alts = [[4, NULL, 1, 3, 1],
            [2, 0.5, 1, 8, 16],
            [6, 2, 3, 5, 11],
            [5, 5, 5, 3, 7],
            [0, 1.5, 3, 4, 12],
            [5, 3.5, 2, 3, 11],
            [11, 7.5, 4, 1, 2],
            [4, 7, 10, 11, 10]]

    t0 = time.time()
    for i in range(100):
        alts = [[random.randrange(0, 100) for i in range(5)]
                for j in range(100)]

        for alt in alts:
            alt[1] = (alt[0] + alt[2])/2

        ind = train_dom(alts, 1)
        print(ind)
    print('time :', time.time() - t0)
    A_plus = alts[1:]
    est = estimate_by_dom_with_criteria(A_plus, 1, alts[0], [0, 2])
    print(est)

    # proportion = 0.01
    # seed = random.randint(0, 1000)
    # print('seed', seed)
    # incomplete_alts = delete_l_evaluations(alts, l=1, seed=seed)

    # print('init :')
    # helpers.printmatrix(alts)

    # print()
    # print('gapped :')

    # alts = replace_by_sreg(incomplete_alts)
    # helpers.printmatrix(alts)
