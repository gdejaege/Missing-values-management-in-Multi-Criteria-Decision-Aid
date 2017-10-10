"""Comparison the missing values replacement strategies."""

import missing_values as mv
import promethee as prom
import data_reader as dr
import copy
from scipy import stats
import random


def compare_rankings_once(all_alts, alt_num, weights):
    """Compare strategies once."""
    seed = random.randint(0, 1000)
    # seed = 289
    # print(seed)
    alts = random.sample(all_alts, alt_num)
    proportion = 0.05
    alts_95pc = copy.deepcopy(alts)

    mv.delete_evaluations(alts_95pc, proportion, seed)
    alts_mean = mv.replace_by_mean(alts_95pc)
    alts_median = mv.replace_by_median(alts_95pc)
    alts_knn = mv.replace_by_neighboors(alts_95pc)

    PII = prom.PrometheeII(alts, weights=weights, seed=seed)
    PMV = prom.PrometheeMV(alts, weights=weights, seed=seed)
    ranking_init = PII.ranking

    score = PII.compute_netflow(alts_mean)
    ranking_mean = PII.compute_ranking(score)

    score = PII.compute_netflow(alts_median)
    ranking_med = PII.compute_ranking(score)

    score = PII.compute_netflow(alts_knn)
    ranking_knn = PII.compute_ranking(score)

#     for i in range(len(ranking_init)):
#         print(str(ranking_init[i]) + " :: " + str(ranking_knn[i]) +
#               " :: " + str(ranking_mean[i]) + " :: " + str(ranking_med[i]))
#
#    print(str(stats.kendalltau(ranking_init, ranking_knn)[0]) + " :: " +
#          str(stats.kendalltau(ranking_init, ranking_mean)[0]) + " :: " +
#          str(stats.kendalltau(ranking_init, ranking_med)[0]))
#
    kendall_taus = {'knn': stats.kendalltau(ranking_init, ranking_knn)[0],
                    'mean': stats.kendalltau(ranking_init, ranking_mean)[0],
                    'med': stats.kendalltau(ranking_init, ranking_med)[0]}
    return kendall_taus


def compare_rankings(alt_num=20, it=500):
    """Compare strategies."""
    random.seed()
    tau_knn_tot = 0
    tau_med_tot = 0
    tau_mean_tot = 0
    datasets = ('SHA', 'EPI', 'HP')
    for dataset in datasets:
        filename = 'data/' + dataset + '/raw.csv'
        all_alts, weights = dr.open_raw(filename)[0], dr.open_raw(filename)[1]
        print(len(all_alts))
        if weights == []:
            weights = None

        tau_knn, tau_mean, tau_med = 0, 0, 0
        for i in range(it):
            taus = compare_rankings_once(all_alts, alt_num, weights)
            tau_knn += taus['knn']
            tau_med += taus['med']
            tau_mean += taus['mean']

        tau_knn_tot += tau_knn
        tau_med_tot += tau_med
        tau_mean_tot += tau_mean

        print('data set:: ' + dataset)
        print('tau knn :: ' + str(tau_knn/it))
        print('tau med :: ' + str(tau_med/it))
        print('tau mean :: ' + str(tau_mean/it))

    print('total average tau:')
    print('tau knn :: ' + str(tau_knn_tot/(len(datasets)*it)))
    print('tau med :: ' + str(tau_med_tot/(len(datasets)*it)))
    print('tau mean :: ' + str(tau_mean_tot/(len(datasets)*it)))
