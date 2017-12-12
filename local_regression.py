"""Estimate and replace evaluations with regression of similar alternatives."""

import numpy as np
from helpers import NULL
import random
import helpers
import data_reader as dr
import regression

from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def get_estimation_by_local_regression(A):
    """Get the ev of missing alternative with regression on similar alts.

    Please use it with only one missing evaluation.
    """
    a_miss = [alt for alt in A if NULL in alt][0]
    A_complete = [alt for alt in A if NULL not in alt]

    c = a_miss.index(NULL)
    estimation = estimate_by_local_regression(A_complete, c, a_miss)
    # print(estimation)
    return estimation


def estimate_by_local_regression(A_complete, c, a_miss):
    """Estimatie the evaluation provided with regression on similar alts.

    Please use it with only one missing evaluation.
    """
    # helpers.printmatrix(A_complete)
    # helpers.printmatrix(a_miss)
    # 1.compute the subset of criteria that will be used to judge similarity
    # (See with Jipe)
    ss_criteria = [i for i in range(len(A_complete[0])) if i != c]   # all but c
    # print(ss_criteria)
    a_miss_ss = [a_miss[i] for i in ss_criteria]

    # 2. Compute alternatives similar to a_miss on ss_criteria
    # (see with Jeanno for better cluster)
    A_sim = compute_similars(A_complete, ss_criteria, a_miss)

    # print('similars:')
    # helpers.printmatrix(A_sim)

    # 3. Compute the set of delta A_sim_ss / delta A_sim_c
    delta_sim_ss, delta_sim_c = compute_deltas_tr(A_sim, c, ss_criteria)
    delta_test = compute_deltas_test(A_sim, a_miss, ss_criteria)

    # print('deltas:')
    # helpers.printmatrix(delta_sim_ss)
    # helpers.printmatrix(delta_sim_c)
    # helpers.printmatrix(delta_test)

    # print(delta_sim_c)

    estimation = compute_missing_ev(delta_sim_ss, delta_sim_c, delta_test,
                                    a_miss_ss, A_sim, c)

    return estimation


def compute_criteria(A, c):
    """Get a subset of criteria to be used (avoid overfitting?).

    The method returns at least 3 criteria.
    """
    criteria_evaluations = list(map(list, zip(*A)))
    criteria = list(range(len(criteria_evaluations)))
    # print('criteria:', criteria)

    crit_c = criteria_evaluations[c]

    if len(criteria) < 3:
        return criteria

    correlations = [stats.pearsonr(crit_c, crit)[0]
                    for crit in criteria_evaluations]

    criteria = sorted(criteria, key=lambda x: correlations[criteria.index(x)],
                      reverse=True)
    criteria.remove(c)
    # print('criteria:', criteria)

    # Qu'est ce que je fais ?
    # i : end -> 0, j: 0 -> i, if we find j s.t.  cor(i,j) > 0.65 : del i
    # end of list are those with the lowest correlation with c
    i = len(criteria) - 1
    while i > 0 and len(criteria) > 3:
        j = 0
        eval_i = criteria_evaluations[criteria[i]]
        eval_j = criteria_evaluations[criteria[j]]
        while j < i and stats.pearsonr(eval_i, eval_j)[0] < 0.65:
            j += 1
            eval_j = criteria_evaluations[criteria[j]]

        if j < i and stats.pearsonr(eval_i, eval_j)[0] >= 0.65:
            # print(i, j, stats.pearsonr(eval_i, eval_j)[0])
            del criteria[i]
        i -= 1
    return criteria


def compute_similars(A_complete, ss_criteria, a_miss, k=5):
    """Get the alts similar to a_miss of a subset of criteria."""
    A_comp_ss = [[a[i] for i in ss_criteria] for a in A_complete]
    A_comp_ss.append([a_miss[i] for i in ss_criteria])

    data = np.array(A_comp_ss)
    data = normalize(data, axis=0, copy=True, norm='max')

    target = np.array([data[-1]])
    data = data[:-1]

    nnbrg = NearestNeighbors(n_neighbors=k).fit(data)
    target = target.reshape(1, -1)
    distances, indices = nnbrg.kneighbors(target)

    A_sim = [A_complete[i] for i in indices[0]]
    return A_sim


def compute_deltas_tr(A_sim, c, ss_criteria):
    """Compute vector of deltas between each couple of similar alternatives."""
    delta_ss, delta_c = [], []

    # This does not work cause points are combili from each other
    # for i, ai in enumerate(A_sim):
    #     for aj in A_sim[i+1:]:
    #         line = [ai[k] - aj[k] for k in ss_criteria]
    #         delta_ss.append(line)
    #         # print(ai[c], aj[c])
    #         delta_c.append(ai[c] - aj[c])

    # 2th try
    for i in range(len(A_sim) - 1):
        line = [A_sim[i][k] - A_sim[i+1][k] for k in ss_criteria]
        delta_ss.append(line)
        # print(ai[c], aj[c])
        delta_c.append(A_sim[i][c] - A_sim[i+1][c])

    return delta_ss, delta_c


def compute_deltas_test(A_sim, a_miss, ss_criteria):
    """Compute vector of deltas between a_miss and similar alternatives."""
    delta_test = []

    for a in A_sim:
        line = [a_miss[k] - a[k] for k in ss_criteria]
        delta_test.append(line)

    return delta_test


def compute_missing_ev(delta_sim_ss, delta_sim_c, delta_test, a_miss_ss,
                       A_sim, c):
    """Compute the missing evaluation."""
    # print()
    # print("computing missing value")
    # print()

    model = regression.train_regression(delta_sim_c, delta_sim_ss)[0]

    # print()
    # print('delta test')
    # for i in delta_test:
    #     print(i)
    target = delta_test

    estimated_delta = model.predict(target)
    # print('estimated delta')
    # print(estimated_delta)

    estimations = []
    for i, a in enumerate(A_sim):
        # print('estimation = ', a[c], ' + ', estimated_delta[i])
        estimation = a[c] + estimated_delta[i]
        estimations.append(estimation)

    # print(estimations)
    return np.mean(estimations)


if __name__ == '__main__':
    # A = [[1, 2, 3, 4],
    #      [4, 3, 2, 1],
    #      [0, 0, 0, 0]]
    # for a in A:
    #     print(a)
    # print(compute_deltas(A, 0, [1, 2, 3]))

    dataset = "SHA"
    dataset = "CPU"
    filename = 'data/' + dataset + '/raw.csv'

    n = 100
    iterations = 1

    A = dr.open_raw(filename)[0]
    A = random.sample(A, n)

    x = int(input())
    crits = compute_criteria(A, x)
    """
    for it in range(iterations):
        i, c = random.randint(0, len(A)-1), random.randint(0, len(A[0])-1)
        a_miss = A[i]
        ev = a_miss[c]
        a_miss[c] = NULL
        estimation = get_estimation_by_local_regression(A)
        print('evaluation: ', ev)
        print('error: ', ev - estimation)
        A[i][c] = ev
    """
