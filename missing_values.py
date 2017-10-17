#!/usr/bin/env python3
"""Implementation of technique for the replacement of missing values."""


import data_reader as dr
import helpers
import copy
import numpy as np
from scipy import stats
import random
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

NULL = '*'


def replace_by_mean(alternatives):
    """Replace missing evaluations by the mean of the same criteria."""
    T_alternatives = list(map(list, zip(*alternatives)))
    T_filled_alternatives = []
    for crit_evalutations in T_alternatives:
        copy = [j for j in crit_evalutations if j is not NULL]
        mean = np.mean(copy)
        new_crit_evaluations = [j if j != NULL else mean
                                for j in crit_evalutations]
        T_filled_alternatives.append(new_crit_evaluations)
    filled_alternatives = list(map(list, zip(*T_filled_alternatives)))
    return filled_alternatives


def replace_by_median(alternatives):
    """Replace missing evaluations by the mean of the same criteria."""
    T_alternatives = list(map(list, zip(*alternatives)))
    T_filled_alternatives = []
    for crit_evalutations in T_alternatives:
        copy = [j for j in crit_evalutations if j is not NULL]
        mean = np.median(copy)
        new_crit_evaluations = [j if j != NULL else mean
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
        if alt == [NULL for i in range(len(alt))]:
            for crit in range(len(alt)):
                evs = [alts_p[j][crit] for j in range(len(alts_p))
                       if alts_p[j][crit] != NULL]
                mean = sum(evs)/len(evs)
                alt[crit] = mean

    # for alt in alts_p:
    #     print(alt)

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


def delete_evaluations(alternatives_p, proportion, seed=0):
    """Delete 'proportion' of the alternatives evaluations."""
    alternatives = copy.deepcopy(alternatives_p)
    random.seed(seed)
    for alt in alternatives:
        for i in range(len(alt)):
            if random.randint(0, 100) < proportion*100:
                alt[i] = NULL


def delete_l_evaluation(alternatives_p, l=1, seed=0):
    """Delete one random evaluation from one alternative."""
    alternatives = copy.deepcopy(alternatives_p)
    random.seed(seed)
    n = len(alternatives)
    k = len(alternatives[0])
    for removal in range(l):
        (i, c) = random.randrange(0, n), random.randrange(0, k)
        alternatives[i][c] = NULL
    return alternatives


def guess_value(alts):
    """Try to guess the missing[s] value[s]."""
    random.shuffle(alts)
    incompletes = [alt for alt in alts if NULL in alt]
    complete_alts = [alt for alt in alts if NULL not in alt]

    for incomplete in incompletes:
        criteria = [k for k, x in enumerate(incomplete) if x == NULL]
        # print(criteria, incomplete)
        for criterion in criteria:
            model = find_model(complete_alts, criterion, incomplete)


def find_model(alternatives, criterion, incomplete):
    """Try to find a model to guess the evaluations on the given criterion."""
    # matrix of criteria instead of alternatives:
    criteria = list(map(list, zip(*alternatives)))

    # Begin with the machine learning notation : goal = y, data = x
    y = criteria[criterion]
    x_tr = criteria[:criterion] + criteria[criterion+1:]
    x_test = incomplete

    # helpers.print_transpose(x)
    # helpers.print_transpose(y)
    # print(len(x[0]), len(y))

    # First try simple regressions
    correlations = [stats.pearsonr(y, xi)[0] for xi in x_tr]
    print(correlations)
    # consider 4 folds by default
    estimates = [best_regression(y, x_tr_i, x_test[i])
                 for i, x_tr_i in enumerate(x_tr)]


def best_regression(y, x_tr, x_test, folds=4):
    """Find the best simple regression evaluation."""
    # these lm need to fit an 1D array of the style [[a0], [a1], ...]
    n = len(y)
    part = n // folds
    lms = []
    for fold in range(folds):
        indices = [j for j in range(i*part, (i+1)*part)]
        x_fold = [[x_tr[i]] for i in indices]
        y_fold = [[y[i]] for i in indices]
        lm = linear_model.LinearRegression()
        lm.fit(x_fold, y_fold)
        lms.append(lm)


if __name__ == '__main__':
    random.seed(5)
    data_set = 'data/SHA/raw.csv'
    alts = dr.open_raw(data_set)[0][:30]
    # alts = [[2, 2, 2, 2, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 10],
    #        [2, 2, 2, 2, 20], [4, 4, 4, 4, 40]]
    # proportion = 0.01
    seed = 0
    incomplete_alts = delete_l_evaluation(alts, l=1, seed=seed)

    print('init :')
    helpers.printmatrix(alts)

    print()
    print('gapped :')
    helpers.printmatrix(incomplete_alts)

    guess_value(incomplete_alts)
