"""Estimate and replace evaluations based on the dominance relations."""

import numpy as np
from helpers import NULL
import random
import helpers
import data_reader as dr

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


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
            if type(estimation) == str:
                print('we have a problem')
                exit()
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


def get_estimations_by_dominance_knn_4(A):
    """Temp."""
    return get_estimations_by_dominance_knn(A, 4)


def get_estimations_by_dominance_knn_3(A):
    """Temp."""
    return get_estimations_by_dominance_knn(A, 3)


def get_estimations_by_dominance_knn_2(A):
    """Temp."""
    return get_estimations_by_dominance_knn(A, 2)


def get_estimations_by_dominance(A):
    """Get the evaluation provided with dominance.

    Please use it with only one missing evaluation.
    """
    incomplete = [alt for alt in A if NULL in alt][0]
    complete_alts = [alt for alt in A if NULL not in alt]

    c = incomplete.index(NULL)
    estimation = estimate_by_dominance(complete_alts, c, incomplete)
    return estimation


def get_estimations_by_dominance_diff(A):
    """Get the evaluation provided with dominance.

    Please use it with only one missing evaluation.
    """
    incomplete = [alt for alt in A if NULL in alt][0]
    complete_alts = [alt for alt in A if NULL not in alt]

    c = incomplete.index(NULL)
    estimation = estimate_by_dominance_smallest_diff(complete_alts, c,
                                                     incomplete)
    return estimation


def get_estimations_by_dominance_knn(A, k=1):
    """Get the estimation on the missing criterion using k neighboors."""
    incomplete = [alt for alt in A if NULL in alt][0]
    complete_alts = [alt for alt in A if NULL not in alt]

    c = incomplete.index(NULL)
    estimation = estimate_by_dominance_knn(complete_alts, c, incomplete, k)
    return estimation


def estimate_by_dominance(A_plus, c, a_miss):
    """Try to estimate one evaluation with dominance.

    Replace the missing value of a_miss on a criterion c by the
    average of :
        - the maximum on c of the alternatvies dominated by a_miss
        - the minimum on c of the alternatives dominating a_miss

    The dominance relation is not a strict dominance relation but is considering
    only a subset of the criteria. This subset is the one leading to the least
    MSE during a training phase on all the complete alternatives. This subset of
    indices is found via the train_dom function.
    """
    indices = train_dom(A_plus, c, a_miss)
    # print('indices train_dom :', indices)
    # print('best indices:', indices)
    estimation = estimate_by_dom_with_criteria(A_plus, c, a_miss, indices)[0]
    return estimation


def estimate_by_dominance_knn(A_plus, c, a_miss, k):
    """Try to estimate one evaluation with dominance.

    Replace the missing value of a_miss on a criterion c by the
    average of :
        - evaluation on c of the k closest alternatvies dominated by a_miss
        - evaluation on c of the k closest alternatives dominating a_miss

    The dominance relation is not a strict dominance relation but is considering
    only a subset of the criteria. This subset is the one leading to the least
    MSE during a training phase on all the complete alternatives. This subset of
    indices is found via the train_dom function.
    """
    indices = train_dom_knn(A_plus, c, a_miss, k)
    # print('indices train_dom :', indices)
    # print('best indices:', indices)
    estimation = estimate_by_dom_with_criteria_knn(A_plus, c, a_miss, indices,
                                                   k)
    return estimation


def estimate_by_dominance_smallest_diff(A_plus, c, a_miss):
    """Try to estimate one evaluation with dominance.

    Replace the missing value of a_miss on a criterion c by the
    average of :
        - the maximum on c of the alternatvies dominated by a_miss
        - the minimum on c of the alternatives dominating a_miss

    The dominance relation is not a strict dominance relation but is considering
    only a subset of the criteria. This subset is the one with the smallest diff
    between the min and maximum (see above)
    MSE during a training phase on all the complete alternatives. This subset of
    indices is found via the train_dom_diff function.
    """
    indices = train_dom_diff(A_plus, c, a_miss)
    # print('indices train_dom_diff:', indices)
    # print('best indices:', indices)
    estimation = estimate_by_dom_with_criteria(A_plus, c, a_miss, indices)[0]
    return estimation


def train_dom_diff(A_plus, c, a_miss):
    """Find the best subset of criteria to apply the dominance estimator."""
    indices = list(range(len(A_plus[0])))
    missing = [i for i in range(len(a_miss)) if a_miss[i] == NULL]

    k_ss = helpers.powerset(indices)
    for j in missing:
        k_ss = [ss for ss in k_ss
                if j not in ss
                and len(ss) > 1]      # to perfom tests on intervals!

    # print(k_ss)
    deltas = [0 for i in range(len(k_ss))]

    for l, ss in enumerate(k_ss):
        delta = estimate_by_dom_with_criteria(A_plus, c, a_miss, ss)[1]
        deltas[l] = delta

    i = 0
    # delta_2 = deltas[:]
    while i < len(deltas):
        if deltas[i] < 0:
            del deltas[i]
            del k_ss[i]
        else:
            i += 1

    try:
        best_ss_index = deltas.index(min(deltas))
    except:
        print('no deltas ...: no dominant neither dominee ...')
        return [i for i in range(len(a_miss)) if a_miss[i] != NULL]

    return k_ss[best_ss_index]


def train_dom(A_plus, c, a_miss):
    """Find the best subset of criteria to apply the dominance estimator."""
    # print('start training')
    indices = list(range(len(A_plus[0])))
    missing = [i for i in range(len(a_miss)) if a_miss[i] == NULL]
    # missing.append(c)           # needed for when a_miss has no real NULL

    k_ss = helpers.powerset(indices)
    for j in missing:
        k_ss = [ss for ss in k_ss
                if j not in ss
                and len(ss) > 1]      # to perfom tests on intervals!

    i = 0
    while i < len(k_ss):
        ss = k_ss[i]
        if count_dominant_alts(A_plus, ss, a_miss) == 0:
            # print('ss deleted', ss)
            del k_ss[i]
        else:
            i += 1

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
            MSE[i] = max(MSE) + 1
        else:
            MSE[i] /= s

    best_ss_index = MSE.index(min(MSE))
    # print('stop training')
    return k_ss[best_ss_index]


def train_dom_knn(A_plus, c, a_miss, k):
    """Find the best subset of criteria to apply the dominance estimator."""
    # print('start training')
    indices = list(range(len(A_plus[0])))
    missing = [i for i in range(len(a_miss)) if a_miss[i] == NULL]
    # missing.append(c)           # needed for when a_miss has no real NULL

    k_ss = helpers.powerset(indices)
    for j in missing:
        k_ss = [ss for ss in k_ss
                if j not in ss
                and len(ss) > 1]      # to perfom tests on intervals!

    i = 0
    while i < len(k_ss):
        ss = k_ss[i]
        if count_dominant_alts(A_plus, ss, a_miss) == 0:
            del k_ss[i]
        else:
            i += 1

    # print(k_ss)
    MSE = [0 for i in range(len(k_ss))]
    samples = [0 for i in range(len(k_ss))]

    for l, ss in enumerate(k_ss):
        for i, ai in enumerate(A_plus):
            old_ev = ai[c]
            ai[c] = NULL
            del A_plus[i]
            new_ev = estimate_by_dom_with_criteria_knn(A_plus, c, ai, ss, k)

            if type(new_ev) != str:
                MSE[l] += (old_ev - new_ev)**2
                samples[l] += 1
            # Restore A_plus as it was initially
            ai[c] = old_ev
            A_plus.insert(i, ai)

    for i, s in enumerate(samples):
        if s == 0:
            MSE[i] = max(MSE) + 1
        else:
            MSE[i] /= s

    # print(MSE)
    best_ss_index = MSE.index(min(MSE))
    # print('stop training')
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

    # better.sort(key=better[c], reverse=True)
    better_c = [a[c] for a in better]
    worse_c = [a[c] for a in worse]

    if better_c == [] and worse_c == []:
        # print('no dominated neither dominant')
        all_c = [a[c] for a in A_plus]
        # return (np.mean(all_c), -11)
        return ('no dominated neither dominant', -11)

    elif worse_c == []:
        return min(better_c), -10
    elif better_c == []:
        return max(worse_c), -1
    else:
        estimation = (max(worse_c) + min(better_c))/2
        return (estimation, abs(max(worse_c) - min(better_c)))


def estimate_by_dom_with_criteria_knn(A_plus, c, a_miss, indices, k):
    """Estimate the evaluation of a_miss on c with dominace extrapolation."""
    # helpers.printmatrix(A_plus)
    # print(a_miss)
    # print()
    better = A_plus
    worse = A_plus

    for i in indices:
        if a_miss[i] == NULL:
            helpers.printmatrix(A_plus)
            print(a_miss)

    for i in indices:
        better = [a for a in better if a[i] >= a_miss[i]]
        worse = [a for a in worse if a[i] <= a_miss[i]]

    # better.sort(key=better[c], reverse=True)
    better_ind = [[a[i] for i in indices] for a in better]
    worse_ind = [[a[i] for i in indices] for a in worse]

    target = [a_miss[i] for i in indices]
    target = np.array(target)

    if len(better_ind) > 0:
        better_ind = np.array(better_ind)
        nnbrg = NearestNeighbors(n_neighbors=min(k, len(better_ind)))
        nnbrg = nnbrg.fit(better_ind)
        distances, best_k_indices = nnbrg.kneighbors(target)
        # print('best k ind:', best_k_indices, k)
        best_k_indices = best_k_indices[0]
        better_estimation = sum([better[j][c] for j in best_k_indices])/k

    if len(worse_ind) > 0:
        worse_ind = np.array(worse_ind)
        nnbrg = NearestNeighbors(n_neighbors=min(k, len(worse_ind)))
        nnbrg = nnbrg.fit(worse_ind)
        distances, worse_k_indices = nnbrg.kneighbors(target)
        worse_k_indices = worse_k_indices[0]
        worse_estimation = sum([worse[j][c] for j in worse_k_indices])/k

    if len(better_ind) == 0 and len(worse_ind) == 0:
        # print('no dominated neither dominant')
        # all_c = [a[c] for a in A_plus]
        # return (np.mean(all_c), -11)
        return 'no dominated neither dominant'

    elif len(better_ind) == 0:
        return worse_estimation
    elif len(worse_ind) == 0:
        return better_estimation
    else:
        estimation = (worse_estimation + better_estimation)/2
        return estimation
        # return (estimation, abs(worse_estimation - better_estimation))


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


def count_dominant_alts(A_plus, criteria, a_miss):
    """Count the number of dominant and dominated alts in A_plus."""
    better = A_plus
    worse = A_plus

    for c in criteria:
        if a_miss[c] == NULL:
            print('error, count dominant with a[c] == NULL.')
            exit()
        better = [a for a in better if a[c] >= a_miss[c]]
        worse = [a for a in worse if a[c] <= a_miss[c]]

    return len(better), len(worse)


def get_dominant_evaluations(A, criteria, a_miss):
    """Return the evaluations on c on dominant and dominated alternatives."""
    better = A
    worse = A

    for c in criteria:
        if a_miss[c] == NULL:
            print('error, count dominant with a[c] == NULL.')
            exit()
        better = [a for a in better if a[c] >= a_miss[c]]
        worse = [a for a in worse if a[c] <= a_miss[c]]

    c = a_miss.index(NULL)
    better_c = [b[c] for b in better]
    worse_c = [b[c] for b in worse]

    return better_c, worse_c


if __name__ == '__main__':
    datasets = ("SHA", "EPI", "HR")
    header = ["", "MEAN", "STD"]
    alt_num = 100
    percentiles = [12.5, 25, 37.5, 50, 62.5, 75, 87.5]
    res = []

    perc = 50
    dataset = "SHA"
    filename = 'data/' + dataset + '/raw.csv'
    alts, weights = dr.open_raw(filename)[0], dr.open_raw(filename)[1]
    alts = random.sample(alts, alt_num)
    good_ints, bad_ints, no_ints, int_mean, int_std = \
        check_dominance_interval(alts, perc)
    res.append([dataset, good_ints, bad_ints, no_ints, int_mean, int_std])

    print('finish')

    # for perc in percentiles:
    #     print(perc)
    #     res = []
    #     for dataset in datasets:
    #         filename = 'data/' + dataset + '/raw.csv'
    #         alts, weights = dr.open_raw(filename)[0], dr.open_raw(filename)[1]
    #         alts = random.sample(alts, alt_num)
    #         alts = normalize(alts, axis=0, copy=True, norm='max')
    #         alts = [list(alt) for alt in alts]

    #         good_ints, bad_ints, no_ints, int_mean, int_std = \
    #             check_dominance_interval(alts, perc)
    #         res.append([dataset, good_ints, bad_ints, no_ints,
    #                     int_mean, int_std])

    #     helpers.printmatrix(res)
    #     print()

#         prop, std = check_dominance_assumption(alts)
#         res.append([dataset, prop, std])
#
#     res.append(["TOTAL", np.mean([r[1] for r in res]),
#                 np.mean([r[2] for r in res])])
