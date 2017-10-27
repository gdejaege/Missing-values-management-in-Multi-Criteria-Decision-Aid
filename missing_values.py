#!/usr/bin/env python3
"""Implementation of technique for the replacement of missing values."""


import data_reader as dr
import helpers
import copy
import numpy as np
from scipy import stats
import random
import time
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


def get_mean(A):
    """Find the mean estimation."""
    incomplete = [alt for alt in A if NULL in alt]
    c = incomplete.index(NULL)
    evs_c = [a[c] for a in A if a[c] != NULL]
    return mean(evs_c)


def get_med(A):
    """Find the med estimation."""
    incomplete = [alt for alt in A if NULL in alt]
    c = incomplete.index(NULL)
    evs_c = [a[c] for a in A if a[c] != NULL]
    return med(evs_c)


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


def estimate_by_knn(complete, c, incomplete, k=1):
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
    data = [[a[i] for i in range(len(a[0])) if i != c] for a in complete]
    data.append([incomplete[i] for i in range(len(incomplete)) if i != c])

    data = np.array(data)
    data = normalize(data, axis=0, copy=True, norm='max')

    target = data[-1]
    data = data[:-1]

    nnbrg = NearestNeighbors(n_neighbors=k).fit(data)
    distances, indices = nnbrg.kneighbors(pivot_values)
    estimation = sum([complete[j][c] for j in indices])/k
    return estimation


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
    # helpers.printmatrix(alternatives)
    n = len(alternatives)
    k = len(alternatives[0])
    for removal in range(l):
        (i, c) = random.randrange(0, n), random.randrange(0, k)
        # print('deleted evaluation', i, c, ':', alternatives[i][c])
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
    x_tr_transposed = criteria[:criterion] + criteria[criterion+1:]
    # training set in a n*k form
    x_tr = list(map(list, zip(*x_tr_transposed)))
    x_test = [ev for ev in incomplete if ev != NULL]

    correlations = [stats.pearsonr(y, xi)[0] for xi in x_tr_transposed]

    models = []
    for c in range(len(x_tr[0])):
        x_tr_c = [[a[c]] for a in x_tr]
        models.append(train_regression(y, x_tr_c))

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


def train_regression(y, x,  folds=4):
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

    if type(x[0]) == float:
        x = [x]

    x_tr = [x[i] for i in i_tr]
    # x_tr2 = [x[i] for i in i_tr]
    x_test = [x[i] for i in i_test]
    # x_test2 = [x[i] for i in i_test]

    y_tr = [[y[i]] for i in i_tr]
    # y_tr2 = [y[i] for i in i_tr]
    y_test = [y[i] for i in i_test]
    # y_test2 = [y[i] for i in i_test]

    # helpers.print_transpose([x_tr2, y_tr2])

    lm = linear_model.LinearRegression()
    lm.fit(x_tr, y_tr)
    y_pred = [pred[0] for pred in lm.predict(x_test)]
    # helpers.print_transpose([x_test2, y_test2, y_pred])
    MSE = mean_squared_error(y_pred, y_test)
    # print(mean_squared_error(y_pred, y_test))
    return lm, MSE


# Not completed!
def guess_all_bests_estimations(alts):
    """Try to find the best estimation by learning."""
    incompletes = [alt for alt in alts if NULL in alt]
    complete_alts = [alt for alt in alts if NULL not in alt]

    completed_alts = copy.deepcopy(alts)

    for incomplete in incompletes:
        i = alts.index(incomplete)
        missing_crits = [k for k, x in enumerate(incomplete) if x == NULL]
        for c in missing_crits:
            # method = tuple with method name, and criteria needed.
            # ex ('reg', [1, 3, 4]) or ('mean', [])
            estimation = guess_best_estimation(complete_alts, incomplete, c)


# Not completed!
def guess_best_estimation(completes, incomplete, c):
    """Try to find the best estimation by learning."""
    # matrix of criteria instead of alternatives:
    criteria = list(map(list, zip(*completes)))

    # Begin with the machine learning notation : goal = y, data = x
    y = criteria[c]
    x = criteria[:c] + criteria[c+1:]

    methods = ['mean', 'med', 'knn']
    methods = [(m, []) for m in methods]
    features = [i for i in range(len(x))]
    for sub_features in helpers.powerset(features):
        methods.append(('reg', sub_features))

    # print(methods)

    MSE = []
    for meth in methods:
        if meth[0] == 'mean':
            MSE.append(np.std(x[c]))
        elif meth[0] == 'med':
            MSE.append(med_mse(x[c]))
        elif meth[0] == 'knn':
            MSE.append(knn_mse(completes, c))
        else:
            x_tr = [xi for i, xi in enumerate(x) if i in meth[1]]
            MSE.append(train_regression(y, x_tr))

    print('ok')
    for i, meth in enumerate(methods):
        print(meth[0], meth[1], '::',  MSE[i])


def replace_by_dominance(alts):
    """Try to estimate all missing evaluations with dominant/ated alternatives.

    Replace all missing values of an alternative x on a criterion k by the
    average of :
        - the maximum on k of the alternatvies dominated by x
        - the minimum on k of the alternatives dominating x

    The dominance relation is not a strict dominance relation but is considering
    only a subset of the criteria. This subset is the one leading to the least
    MSE during a training phase on all the complete alternatives.
    """
    incompletes = [alt for alt in alts if NULL in alt]
    complete_alts = [alt for alt in alts if NULL not in alt]

    completed_alts = copy.deepcopy(alts)
    for incomplete in incompletes:
        i = alts.index(incomplete)
        criteria = [k for k, x in enumerate(incomplete) if x == NULL]
        for c in criteria:
            estimation = estimate_by_dominance(complete_alts, c, incomplete)
            completed_alts[i][c] = estimation
    return completed_alts


def replace_by_dominance_smallest_diff(alts):
    """Try to estimate all missing evaluations with dominant/ated alternatives.

    Replace all missing values of an alternative x on a criterion k by the
    average of :
        - the maximum on k of the alternatvies dominated by x
        - the minimum on k of the alternatives dominating x

    The dominance relation is not a strict dominance relation but is considering
    only a subset of the criteria. This subset is the one leading to the least
    MSE during a training phase on all the complete alternatives.
    """
    incompletes = [alt for alt in alts if NULL in alt]
    complete_alts = [alt for alt in alts if NULL not in alt]

    completed_alts = copy.deepcopy(alts)
    for incomplete in incompletes:
        i = alts.index(incomplete)
        criteria = [k for k, x in enumerate(incomplete) if x == NULL]
        for c in criteria:
            estimation = estimate_by_dominance_smallest_diff(complete_alts,
                                                             c, incomplete)
            if type(estimation) == str:
                print('pb, estimation:', estimation)
            completed_alts[i][c] = estimation
    return completed_alts


def get_estimations_by_dominance(A):
    """Get the evaluation provided with dominance.

    Please use it with only one missing evaluation.
    """
    incomplete = [alt for alt in alts if NULL in alt]
    complete_alts = [alt for alt in alts if NULL not in alt]
    estimation = estimate_by_dominance_smallest_diff(complete_alts, c,
                                                     incomplete)
    return estimation


def estimate_by_dominance(A_plus, c, alt_miss):
    """Try to estimate one evaluation with dominance.

    Replace the missing value of alt_miss on a criterion c by the
    average of :
        - the maximum on c of the alternatvies dominated by alt_miss
        - the minimum on c of the alternatives dominating alt_miss

    The dominance relation is not a strict dominance relation but is considering
    only a subset of the criteria. This subset is the one leading to the least
    MSE during a training phase on all the complete alternatives. This subset of
    indices is found via the train_dom function.
    """
    indices = train_dom(A_plus, c, alt_miss)
    # print('best indices:', indices)
    estimation = estimate_by_dom_with_criteria(A_plus, c, alt_miss, indices)[0]
    return estimation


def estimate_by_dominance_smallest_diff(A_plus, c, alt_miss):
    """Try to estimate one evaluation with dominance.

    Replace the missing value of alt_miss on a criterion c by the
    average of :
        - the maximum on c of the alternatvies dominated by alt_miss
        - the minimum on c of the alternatives dominating alt_miss

    The dominance relation is not a strict dominance relation but is considering
    only a subset of the criteria. This subset is the one with the smallest diff
    between the min and maximum (see above)
    MSE during a training phase on all the complete alternatives. This subset of
    indices is found via the train_dom_diff function.
    """
    indices = train_dom_diff(A_plus, c, alt_miss)
    # print('best indices:', indices)
    estimation = estimate_by_dom_with_criteria(A_plus, c, alt_miss, indices)[0]
    return estimation


def train_dom_diff(A_plus, c, alt_miss):
    """Find the best subset of criteria to apply the dominance estimator."""
    indices = list(range(len(A_plus[0])))
    missing = [i for i in range(len(alt_miss)) if alt_miss[i] == NULL]

    k_ss = helpers.powerset(indices)
    for j in missing:
        k_ss = [ss for ss in k_ss
                if j not in ss]

    # print(k_ss)
    deltas = [0 for i in range(len(k_ss))]

    for l, ss in enumerate(k_ss):
        delta = estimate_by_dom_with_criteria(A_plus, c, alt_miss, ss)[1]
        deltas[l] = delta

    i = 0
    while i < len(deltas):
        if deltas[i] < 0:
            del deltas[i]
            del k_ss[i]
        else:
            i += 1

    try:
        best_ss_index = deltas.index(min(deltas))
    except:
        print('matrix')
        helpers.printmatrix(A_plus)
        print(c)
        print(alt_miss)
        exit()

    return k_ss[best_ss_index]


def train_dom(A_plus, c, alt_miss):
    """Find the best subset of criteria to apply the dominance estimator."""
    indices = list(range(len(A_plus[0])))
    missing = [i for i in range(len(alt_miss)) if alt_miss[i] == NULL]

    k_ss = helpers.powerset(indices)
    for j in missing:
        k_ss = [ss for ss in k_ss
                if j not in ss]

    # print(k_ss)
    MSE = [0 for i in range(len(k_ss))]
    samples = [0 for i in range(len(k_ss))]

    for l, ss in enumerate(k_ss):
        for i, ai in enumerate(A_plus):
            old_ev = ai[c]
            ai[c] = NULL
            del A_plus[i]
            new_ev = estimate_by_dom_with_criteria(A_plus, c, ai, ss)[0]

            if type(new_ev) != str:
                MSE[l] += (old_ev - new_ev)**2
                samples[l] += 1
            # Restore A_plus as it was initially
            ai[c] = old_ev
            A_plus.insert(i, ai)

    for i, s in enumerate(samples):
        if s == 0:
            MSE[i] = max(MSE)
        else:
            MSE[i] /= s

    best_ss_index = MSE.index(min(MSE))
    return k_ss[best_ss_index]


def estimate_by_dom_with_criteria(A_plus, c, a_miss, indices):
    """Estimate the evaluation of a_miss on c with dominace extrapolation."""
    # helpers.printmatrix(A_plus)
    # print(a_miss)
    # print()
    better = A_plus
    worse = A_plus

    for i in indices:
        if type(a_miss[i]) == str:
            helpers.printmatrix(A_plus)
            print()
            print(a_miss)

    for i in indices:
        better = [a for a in better if a[i] >= a_miss[i]]
        worse = [a for a in worse if a[i] <= a_miss[i]]

    better_c = [a[c] for a in better]
    worse_c = [a[c] for a in worse]

    if better_c == [] and worse_c == []:
        return ('no dominated neither dominant', -11)
    elif worse_c == []:
        return min(better_c), -10
    elif better_c == []:
        return max(worse_c), -1
    else:
        return ((max(worse_c) + min(better_c))/2,
                abs(max(worse_c) - min(better_c)))


def check_train_dom(A):
    """Check to see wheter the training provides the best subset of indices."""
    # A_copy = copy.deepcopy(A)
    n = len(A)
    k = len(A[0])
    av_pos = 0
    # for tests purposis
    # for i in range(1):
    # ai = A[i][:]
    error_total_best_indices = 0
    error_total_all_ss = 0
    total_ss = 0
    total_best_ind = 0
    for i, ai in enumerate(A):
        for c, aic in enumerate(ai):
            # for c in range(1, 2):
            aic = ai[c]
            del A[i]
            best_ind = train_dom(A, c)

            # print('ok')

            all_ind_ss = helpers.powerset(list(range(k)))
            all_ind_ss = [ss for ss in all_ind_ss
                          if c not in ss and len(ss) > 0]

            error = []
            estimations = []
            for r, ss in enumerate(all_ind_ss):
                # helpers.printmatrix(A)
                est = estimate_by_dom_with_criteria(A, c, ai, ss)[0]
                estimations.append(est)
                error.append(abs(est - aic))

            error_total_best_indices += error[all_ind_ss.index(best_ind)]
            total_best_ind += 1
            error_total_all_ss += sum(error)
            total_ss += len(error)
#            print('best indices', best_ind, 'value :', aic)
#
#            for ss, est, e in zip(all_ind_ss, estimations, error):
#                print(ss, '::', est, '::', e)
#
            all_ind_ss = sorted(all_ind_ss,
                                key=lambda ss: error[all_ind_ss.index(ss)])

            ss_num = len(all_ind_ss)
#            print(len(all_ind_ss))
            av_pos += all_ind_ss.index(best_ind)

            A.insert(i, ai)
    print('av_pos', av_pos/(n), '/', ss_num)
    print('MSE best ind:', error_total_best_indices/total_best_ind)
    print('MSE all ind:', error_total_all_ss/total_ss)

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
