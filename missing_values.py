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
from sklearn.metrics import mean_squared_error
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


def replace_by_knn(alts_p, k=1):
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
            estimation = sum([alts[j][pivot_crit] for j in final_indices])/k
            # print('knn replacement:', estimation)
            alts[pivot_alt][pivot_crit] = estimation
    return alts


def delete_evaluations(alternatives_p, proportion, seed=0):
    """Delete 'proportion' of the alternatives evaluations."""
    alternatives = copy.deepcopy(alternatives_p)
    random.seed(seed)
    for alt in alternatives:
        for i in range(len(alt)):
            if random.randint(0, 100) < proportion*100:
                alt[i] = NULL


def delete_l_evaluations(alternatives_p, l=1, seed=0):
    """Delete one random evaluation from one alternative."""
    alternatives = copy.deepcopy(alternatives_p)
    random.seed(seed)
    n = len(alternatives)
    k = len(alternatives[0])
    for removal in range(l):
        (i, c) = random.randrange(0, n), random.randrange(0, k)
        # print('deleted evaluation', alternatives[i][c])
        alternatives[i][c] = NULL
    return alternatives


def replace_by_creg(alts):
    """Try to guess the missing[s] value[s] using correlated regressions."""
    return replace_by_reg(alts, 'correlation')


def replace_by_sreg(alts):
    """Try to guess the missing[s] value[s] using simple regressions."""
    return replace_by_reg(alts, 'simple')


def replace_by_ereg(alts):
    """Try to guess the missing[s] value[s] using error regressions."""
    return replace_by_reg(alts, 'error')


def replace_by_reg(alts, method):
    """Try to guess the missing[s] value[s] using the precised regression."""
    # random.shuffle(alts)
    # helpers.printmatrix(alts)
    incompletes = [alt for alt in alts if NULL in alt]
    complete_alts = [alt for alt in alts if NULL not in alt]

    completed_alts = copy.deepcopy(alts)
    for incomplete in incompletes:
        i = alts.index(incomplete)
        criteria = [k for k, x in enumerate(incomplete) if x == NULL]
        for c in criteria:
            evaluation = evaluate_with_regression(complete_alts, c, incomplete,
                                                  method)
            completed_alts[i][c] = evaluation
    return completed_alts


def evaluate_with_regression(alternatives, criterion, incomplete,
                             method='simple'):
    """Try to find a model to guess the evaluations on the given criterion."""
    # matrix of criteria instead of alternatives:
    criteria = list(map(list, zip(*alternatives)))

    # Begin with the machine learning notation : goal = y, data = x
    y = criteria[criterion]
    x_tr = criteria[:criterion] + criteria[criterion+1:]
    x_test = [ev for ev in incomplete if ev != NULL]

    correlations = [stats.pearsonr(y, xi)[0] for xi in x_tr]

    models = [(train_regression(y, x_i)) for x_i in x_tr]
    estimations = [float(model[0].predict(xi))
                   for model, xi in zip(models, x_test)]
    MSEs = [i[1] for i in models]
    MSEs = [i/sum(MSEs) for i in MSEs]

    if method == 'simple':
        best_ind = MSEs.index(min(MSEs))
        estimation = float(models[best_ind][0].predict(x_test[best_ind]))
        # print('Sreg replacement:', estimation)
    elif method == 'correlation':
        estimation = sum([e*r for e, r in zip(estimations, correlations)])
        estimation /= sum(correlations)
        # print('Correlation replacement:', estimation)
    elif method == 'error':
        estimation = sum([e*(1 - err) for e, err in zip(estimations, MSEs)])
        estimation /= sum([1 - i for i in MSEs])
        # print('Error replacement:', estimation)

    return estimation


def train_regression(y, x,  folds=10):
    """Find the best simple regression evaluation."""
    # these lm need to fit an 1D array of the style [[a0], [a1], ...]
    n = len(y)
    part = n // folds
    lms = []
    MSEs = []
    # print('x tr : \n', x)
    # print('y : \n', y)
    # print()

    # for fold in range(1):
    for fold in range(folds):
        lm, MSE = regression(y, x, fold, part)
        MSEs.append(MSE)
        lms.append(lm)

    lms = sorted(lms, key=lambda model: MSEs[lms.index(model)])
    MSEs.sort()
    return lms[0], MSEs[0]


def regression(y, x, fold, part):
    """Perform and test a linear regression."""
    n = len(y)
    i_test = [j for j in range(fold*part, (fold+1)*part)]
    i_tr = [j for j in range(n) if j not in i_test]

    x_tr = [[x[i]] for i in i_tr]
    x_tr2 = [x[i] for i in i_tr]
    x_test = [[x[i]] for i in i_test]
    x_test2 = [x[i] for i in i_test]

    y_tr = [[y[i]] for i in i_tr]
    y_tr2 = [y[i] for i in i_tr]
    y_test = [y[i] for i in i_test]
    y_test2 = [y[i] for i in i_test]

    # helpers.print_transpose([x_tr2, y_tr2])

    lm = linear_model.LinearRegression()
    lm.fit(x_tr, y_tr)
    y_pred = [pred[0] for pred in lm.predict(x_test)]
    # helpers.print_transpose([x_test2, y_test2, y_pred])
    MSE = mean_squared_error(y_pred, y_test)
    # print(mean_squared_error(y_pred, y_test))
    return lm, MSE


def guess_best_estimation(alts):
    """Try to find the best estimation by learning."""
    incompletes = [alt for alt in alts if NULL in alt]
    complete_alts = [alt for alt in alts if NULL not in alt]

    completed_alts = copy.deepcopy(alts)

    for incomplete in incompletes:
        i = alts.index(incomplete)
        missing_crits = [k for k, x in enumerate(incomplete) if x == NULL]
        for c in missing_crits:
            1


if __name__ == '__main__':
    V = True
    # V = False
    # random.seed(5)
    data_set = 'data/SHA/raw.csv'
    alts = dr.open_raw(data_set)[0][:30]
    # alts = [[2, 2, 2, 2, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 10],
    #        [2, 2, 2, 2, 20], [4, 4, 4, 4, 40]]
    # proportion = 0.01
    seed = random.randint(0, 1000)
    print('seed', seed)
    incomplete_alts = delete_l_evaluations(alts, l=1, seed=seed)

    print('init :')
    helpers.printmatrix(alts)

    print()
    print('gapped :')

    alts = replace_by_sreg(incomplete_alts)
    helpers.printmatrix(alts)
