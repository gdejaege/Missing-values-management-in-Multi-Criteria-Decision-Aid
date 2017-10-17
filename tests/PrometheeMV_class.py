"""Test the PrometheeMV class."""

import promethee as prom
import missing_values as mv
import data_reader as dr
import copy


def test_ranking(dataset='HDI'):
    """Test that PIIMV computes same ranking as PII when no missing value."""
    data_set = 'data/' + dataset + '/raw.csv'
    alts, weights = dr.open_raw(data_set)[0][0:5], dr.open_raw(data_set)[1]
    # print(alts)
    # print(weights)
    if weights == []:
        weights = None
    if dataset == 'HDI':
        weights = [0.5, 0.5]
        ceils = [3, 3]
        promethee = prom.PrometheeII(alts, weights=weights, ceils=ceils)
        prometheeMV = prom.PrometheeMV(alts, weights=weights, ceils=ceils)
    else:
        seed = 1
        promethee = prom.PrometheeII(alts, weights=weights, seed=seed)
        prometheeMV = prom.PrometheeMV(alts, weights=weights, seed=seed)
        # print(promethee.ceils, promethee.weights)
    scores = promethee.scores
    scoresMV = prometheeMV.scores
    rank = promethee.ranking
    rankMV = prometheeMV.ranking
    for i in range(len(rank)):
        print(str(rank[i] + 1) + '::' + str(scores[rank[i]]) + " :::: " +
              str(rankMV[i] + 1) + '::' + str(scoresMV[rank[i]]))


def test_replacements():
    """Test that pij are correctly replaced."""
    # initialisation purpose only
    data_set = 'data/HDI/raw.csv'
    alts = dr.open_raw(data_set)[0]
    seed = 1
    method = 'mean'
    prometheeMV = prom.PrometheeMV(alts, seed=seed, method=method)
    alternatives = [[1],
                    [0],
                    ['*'],
                    [2]]
    f = [myf]
    pref = [[[0, 1, '*', 0],
             [0, 0, '*', 0],
             ['*', '*', 0, '*'],
             [1, 1, '*', 0]]]

    for i in pref[0]:
        print(i)

    P = prometheeMV.compute_pairwise_comparisons(alternatives, f)
    print("second round")
    for i in P[0]:
        print(i)


def myf(b):
    """Temp."""
    if b > 0:
        return 1
    else:
        return 0


def test_PMV(dataset="HDI"):
    """Test PMV with, this time, missing values."""
    data_set = 'data/' + dataset + '/raw.csv'
    alts = dr.open_raw(data_set)[0][:10]
    proportion = 0.2
    seed = 1
    print("complete :")
    prom.printmatrix(alts)
    original_alts = copy.deepcopy(alts)
    mv.delete_evaluations(alts, proportion, seed)
    print("incomplete :")
    prom.printmatrix(alts)

    print("Promethee:")
    promethee = prom.PrometheeII(original_alts, seed=seed)
    rank = promethee.ranking
    scores = promethee.scores

    print("PrometheeMV without missing:")
    prometheeMV1 = prom.PrometheeMV(original_alts, seed=seed)
    rankMV1 = prometheeMV1.ranking
    scoresMV1 = prometheeMV1.scores

    print("PrometheeMV:")
    prometheeMV = prom.PrometheeMV(alts, seed=seed)
    rankMV = prometheeMV.ranking
    scoresMV = prometheeMV.scores
    for i in range(len(rank)):
        print(str(rank[i] + 1) + '::' + str(scores[rank[i]]) + " :::: " +
              str(rankMV1[i] + 1) + '::' + str(scoresMV1[rank[i]]) + " :::: " +
              str(rankMV[i] + 1) + '::' + str(scoresMV[rank[i]]))
