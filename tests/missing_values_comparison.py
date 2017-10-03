"""Comparison the missing values replacement strategies."""

import missing_values as mv
import promethee as prom
import data_reader as dr
import copy
from scipy import stats
import random


def compare_rankings(dataset):
    """Compare strategies."""
    filename = 'data/' + dataset + '/raw.csv'
    random.seed()
    seed = random.randint(0, 1000)
    alts = dr.open_raw(filename)[0]
    proportion = 0.05
    alts_95pc = copy.deepcopy(alts)
    for i in alts:
        print(i)
    mv.delete_evaluations(alts_95pc, proportion, seed)
    for i in alts_95pc:
        print(i)
    alts_mean = mv.replace_by_mean(alts_95pc)
    for i in alts_mean:
        print(i)
    alts_median = mv.replace_by_median(alts_95pc)
    for i in alts_median:
        print(i)
    alts_knn = mv.replace_by_neighboors(alts_95pc)
    for i in alts_knn:
        print(i)

    PII = prom.PrometheeII(alts, seed=seed)
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
    print()
    print(str(stats.kendalltau(ranking_init, ranking_knn)[0]) + " :: " +
          str(stats.kendalltau(ranking_init, ranking_mean)[0]) + " :: " +
          str(stats.kendalltau(ranking_init, ranking_med)[0]))
