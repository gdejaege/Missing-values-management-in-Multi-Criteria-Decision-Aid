"""Tests on the dominance relation based method."""

import time
from helpers import NULL
import helpers
import dominance_estimations as de
from sklearn.preprocessing import normalize
import data_reader as dr
import numpy as np
import random


def check_dominance_assumption(iterations=10):
    """Test if dominance is still respected."""
    datasets = ("SHA", "EPI", "HR")
    header = ["", "MEAN", "STD"]
    n = 100
    res = []

    for dataset in datasets:
        print("\n"*2, "-"*35, dataset, "-"*35, "\n")
        filename = 'data/' + dataset + '/raw.csv'
        A, weights = dr.open_raw(filename)[0], dr.open_raw(filename)[1]
        A = random.sample(A, n)
        A = normalize(A, axis=0, copy=True, norm='max')
        A = [list(alt) for alt in A]
        k = len(A[0])

        res = [[] for i in range(9)]

        for it in range(iterations):
            i = random.randint(0, n - 1)
            c = random.randint(0, k - 1)
            a = A[i]
            del A[i]
            a_miss = a[:]
            a_miss[c] = NULL
            indices = de.train_dom(A, c, a_miss)

            dominant, dominated = de.count_dominant_alts(A, indices, a_miss)
            indices.append(c)
            dominant_c, dominated_c = de.count_dominant_alts(A, indices, a)

            res[0].append(dominant)
            res[1].append(dominant_c)

            res[2].append(dominant_c/dominant if dominant else 0)

            res[3].append(dominated)
            res[4].append(dominated_c)
            res[5].append(dominated_c/dominated if dominated else 0)

            res[6].append(dominated + dominant)
            res[7].append(dominated_c + dominant_c)
            res[8].append((dominated_c + dominant_c)/(dominated + dominant)
                          if (dominated + dominant) else 0)

            A.insert(i, a)

        final_res = [[" ", "   ", "MEAN", "STD"]]
        lines = ["Dom+", "Dc+", "ratio", "dom-", "dc-", "ratio",
                 "Tot", "tot_c", "ratio"]

        for i in range(9):
            final_res.append([lines[i], " ", np.mean(res[i]), np.std(res[i])])

        helpers.printmatrix(final_res, width=5)


def check_if_dominance_interval(iterations=100):
    """Check to see wheter the evaluation is inside a given interval found."""
    random.seed(0)
    datasets = ("SHA", "EPI", "HR")
    header = ["", "Neither", "OR", "AND"]
    n = 100
    percentiles = (0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100)

    res = []
    for dataset in datasets:
        print("\n"*2, "-"*35, dataset, "-"*35, "\n")
        filename = 'data/' + dataset + '/raw.csv'
        A, weights = dr.open_raw(filename)[0], dr.open_raw(filename)[1]
        A = random.sample(A, n)
        A = normalize(A, axis=0, copy=True, norm='max')
        A = [list(alt) for alt in A]
        k = len(A[0])

        res_dataset = [0 for o in range(3)]
        for it in range(iterations):
            iteration_res = check_if_interval_iteration(A, n, k)
            for col in iteration_res:
                res_dataset[col] += 1
        res.append([dataset] + [o/iterations for o in res_dataset])

    helpers.printmatrix([header] + res)


def check_if_interval_iteration(A, n, k):
    """Perfom one iteration."""
    i = random.randint(0, n - 1)
    c = random.randint(0, k - 1)
    a = A[i]
    del A[i]

    a_miss = a[:]
    a_miss[c] = NULL

    ev = a[c]
    res = []

    indices = de.train_dom(A, c, a_miss)   # Train dom? or train dom perc ?
    dominants_c, dominateds_c = de.get_dominant_evaluations(A, indices, a_miss)
    A.insert(i, a)
    if len(dominants_c) == 0 and len(dominateds_c) == 0:
        res.append(0)
    elif len(dominants_c) == 0 or len(dominateds_c) == 0:
        res.append(1)
    else:
        res.append(2)
    return res


def check_good_dominance_interval(iterations):
    """Check to see wheter the evaluation is inside a given interval found."""
    random.seed(0)
    datasets = ("SHA", "EPI", "HR")
    header = [""] + list(datasets)
    n = 100
    percentiles = (0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100)

    res_tot = [[p] for p in percentiles]
    res_tot.append(['av'])

    for dataset in datasets:
        print("\n"*2, "-"*35, dataset, "-"*35, "\n")
        filename = 'data/' + dataset + '/raw.csv'
        A, weights = dr.open_raw(filename)[0], dr.open_raw(filename)[1]
        A = random.sample(A, n)
        A = normalize(A, axis=0, copy=True, norm='max')
        A = [list(alt) for alt in A]
        k = len(A[0])

        res = [[p, 0] for p in percentiles]
        res.append(['av', 0])

        for it in range(iterations):
            iteration_res = check_good_interval_iteration(A, n, k, percentiles)
            for col in iteration_res:
                res[col][1] += 1

        for i in range(len(res)):
            res[i][1] /= iterations
            res_tot[i].append(res[i][1])

        helpers.printmatrix(res)
    helpers.printmatrix([header] + res_tot)


def check_good_interval_iteration(A, n, k, percentiles):
    """Perfom one iteration."""
    i = random.randint(0, n - 1)
    c = random.randint(0, k - 1)
    a = A[i]
    del A[i]

    ev = a[c]
    res = []

    a_miss = a[:]
    a_miss[c] = NULL
    indices = de.train_dom(A, c, a_miss)   # Train dom? or train dom perc ?
    dominants_c, dominateds_c = de.get_dominant_evaluations(A, indices, a_miss)
    while len(dominants_c) == 0 or len(dominateds_c) == 0:
        A.insert(i, a)
        i = random.randint(0, n - 1)
        c = random.randint(0, k - 1)
        a = A[i]
        del A[i]

        a_miss = a[:]
        a_miss[c] = NULL
        indices = de.train_dom(A, c, a_miss)
        dominants_c, dominateds_c = de.get_dominant_evaluations(A, indices,
                                                                a_miss)

    A.insert(i, a)
    for ind, p in enumerate(percentiles):
        ev_plus = np.percentile(dominants_c, p)
        ev_minus = np.percentile(dominateds_c, 100 - p)
        if ev_minus <= ev <= ev_plus:
            res.append(ind)

    ev_plus = np.mean(dominants_c)
    ev_minus = np.mean(dominateds_c)
    if ev_minus <= ev <= ev_plus:
        res.append(len(percentiles))
    return res
