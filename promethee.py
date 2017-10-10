#!/usr/bin/env python3
"""Implementation of PrometheeII.

other usefull functions are implemented such as:
    * classes for the preference functions
    * a function checking if the common parameters between two methods
      instanciation are equal.
"""

import numpy as np
from math import exp
from scipy import stats
import random
import data_reader as dr
import copy

NULL = '*'

"""Preference function classes."""


class PreferenceType2:

    """Quasi-criterion."""

    q = 0

    def __init__(self, q=0):
        """Constructor."""
        self.q = q

    def value(self, diff):
        """Value."""
        if (diff <= self.q):
            return 0
        return 1


class PreferenceType5:

    """Linear preferences and indifference zone."""

    q = 0
    p = 1

    def __init__(self, q=0, p=1):
        """Constructor."""
        self.q = q
        self.p = p

    def value(self, diff):
        """Value."""
        if (diff <= self.q):
            return 0
        if (diff <= self.p):
            return (diff - self.q) / (self.p - self.q)
        return 1


class GeneralizedType5:

    """Symetric type5 criterion."""

    q = 0
    p = 1

    def __init__(self, q, p):
        """Constructor."""
        self.q = q
        self.p = p

    def value(self, diff):
        """Value."""
        if (abs(diff) <= self.q):
            return 0
        res = 0
        if(abs(diff) <= self.p):
            res = (abs(diff) - self.q)/(self.p - self.q)
        else:
            res = 1
        if (diff > 0):
            return res
        else:
            return - res


def check_parameters(method1, method2):
    """Check if all the common parameters between the methods are equal."""
    res = True
    # Alternatives
    A1 = method1.alternatives
    A2 = method2.alternatives
    for i in range(len(A1)):
        if A1[i] != A2[i]:
            res = False

    # Weights
    W1 = method1.weights
    W2 = method2.weights
    if W1 != W2:
        res = False

    # Ceils
    C1 = method1.ceils
    C2 = method2.ceils
    if C1 != C2:
        res = False

    return res


class PrometheeII:

    """PrometheeII class."""

    def __init__(self, alternatives, seed=0, alt_num=-1, ceils=None,
                 weights=None, coefficients=None):
        """Constructor of the PrometheeII class.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            seed - seed provided to python pseudo random number generator. It
                   is used to create some random (w, F) for the method if these
                   are not provided as arguments.
            alt_num - quantity of alternatives from 'alternative' which must be
                      kept.
            ceils - list of the values of the strict preference thresholds for
                    all criteria (p).
            weights - list of the relative importance (or weigths) of all
                      criteria.
            coefficients - if 'ceils' is not provided, some new ceils will be
                           computed as these coefficents time the amplitude
                           between the highest and lowest evaluation of each
                           criterion.
        """
        # first, each Promethee parameter is set at random, this value is then
        # overwritten by the value of the parameters given as argument
        self.alternatives, self.weights, self.coefficients = \
            self.random_parameters(seed, alternatives, alt_num)

        # Transposition of the alternatives: one list of alternatives for each
        # criteria
        self.eval_per_crit = list(map(list, zip(*self.alternatives)))

        if(weights is not None):
            self.weights = weights
        if(coefficients is not None):
            self.coefficients = coefficients

        # Preference functions
        diffs = self.max_diff_per_criterion(self.alternatives)
        if (ceils is None):
            ceils = [diffs[i] * coeff
                     for i, coeff in enumerate(self.coefficients)]
        self.ceils = ceils
        self.pref_functions = [PreferenceType5(0, ceil) for ceil in ceils]

        self.pi = self.compute_pairwise_pref()

        self.scores = self.compute_netflow()
        self.ranking = self.compute_ranking(self.scores)

    def random_parameters(self, s, alternatives, qty=-1):
        """Compute random subset of alternatives and parameters using a seed.

        Inputs:
            s - seed.
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            qty - quantity of alternatives desired, if 'qty' is equal to -1
                  then the set the whole set of alternatives is returned
                  (in the original ordering!).
        """
        random.seed(s)
        if qty != -1:
            alternatives = random.sample(alternatives, qty)

        coefficients = [random.uniform(0.4, 1) for i in alternatives[0]]
        weights = [random.randint(30, 100) for i in alternatives[0]]
        weights = [w/sum(weights) for w in weights]
        return alternatives, weights, coefficients

    def max_diff_per_criterion(self, alternatives, crit_quantity=-1):
        """Retun a list of delta max for each criterion.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            crit_quantity - number of criteria for which the amplitude of the
                            maximal difference must be computed.
        """
        # quantity of criteria we are looking at
        if crit_quantity == -1:
            crit_quantity = len(alternatives[0])

        eval_per_criterion = list(map(list, zip(*alternatives)))

        diff = []
        for criterion in eval_per_criterion:
            criterion_cleaned = [a for a in criterion if a != NULL]
            diff.append(max(criterion_cleaned) - min(criterion_cleaned))

        return diff

    def compute_pairwise_pref(self, alternatives=None, weights=None,
                              pref_funcs=None):
        """Return the pairwise preference matrix Pi.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            weights - list of weights of the criteria.
            pref_funcs - list of the preference functions for all criteria.

        """
        if alternatives is None:
            alternatives = self.alternatives
        if weights is None:
            weights = self.weights
        if pref_funcs is None:
            pref_funcs = self.pref_functions

        pi = [[0 for alt in alternatives] for alt in alternatives]
        for i, alti in enumerate(alternatives):
            for j, altj in enumerate(alternatives):
                for k in range(len(weights)):
                    weight = weights[k]
                    pref_function = pref_funcs[k]
                    valAlti = alti[k]
                    valAltj = altj[k]
                    diff = valAlti - valAltj
                    # val2 = valAlt2 - valAlt1
                    pi[i][j] += weight * pref_function.value(diff)
                    pi[j][i] += weight * pref_function.value(-diff)
        return pi

    def compute_netflow(self, alternatives=None, weights=None, pref_funcs=None):
        """Compute the netflow of the alternatives.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            weights      - list of weights of the criteria.
            pref_functs  - list of the preference functions for all criteria.

        Output :
            netflows[i] = score of the ith alternative in alternatives(input)
        """
        if alternatives is None:
            alternatives = self.alternatives
        if weights is None:
            weights = self.weights
        if pref_funcs is None:
            pref_funcs = self.pref_functions

        netflows = []
        if (len(alternatives) == 1):
                return [1]

        for ai in alternatives:
            flow = 0
            for aj in alternatives:
                if ai == aj:
                    continue
                for k in range(len(weights)):
                    weight = weights[k]
                    pref_k = pref_funcs[k]
                    ai_k = ai[k]
                    aj_k = aj[k]
                    diff = ai_k - aj_k
                    flow += weight*(pref_k.value(diff) - pref_k.value(-diff))
            flow = flow / (len(alternatives) - 1)
            netflows.append(flow)
        return netflows

    def compute_ranking(self, scores):
        """Return the ranking given the scores.

        Input:
            scores[i] = score of the ith alternative

        Output:
            ranking[i] = index in 'scores' of the alternative ranked ith.
        """
        # sort k=range(...) in decreasing order of the netflows[k]
        ranking = sorted(range(len(scores)), key=lambda k: scores[k],
                         reverse=True)
        return ranking

    def compute_rr_number(self, verbose=False):
        """Compute the total number of rank reversals of the method.

        Notes:
            * verbose only serves for printing debug information
            * the ranking (self.ranking) must already be computed
        """
        total_rr_quantity = 0
        for i in range(len(self.alternatives)):
            copy_alternatives = self.alternatives[:]
            del copy_alternatives[i]
            scores = self.compute_netflow(alternatives=copy_alternatives)
            ranks = self.compute_ranking(scores)
            # Since we applied the method on n-1 alternatives, the ranks
            # will be in [0, n-1] instead of [0, n]
            for j in range(len(ranks)):
                if ranks[j] >= i:
                    ranks[j] += 1
            rr_quantity = self.compare_rankings(self.ranking, ranks, i, verbose)
            total_rr_quantity += rr_quantity
        return total_rr_quantity

    def compare_rankings(self, init_ranking, new_ranking, deleted_alt,
                         verbose=False):
        """Compute the number of rank reveSRPal between two rankings.

        Inputs:
            init_ranking - ranking obtained with all alternatives
            new_ranking - ranking obtained when deleted_alt is removed from
                          the set of alternatives.
            deleted_alt - please guess
            verbose - print debuging messages
        """
        init_copy = init_ranking[:]
        new_copy = new_ranking[:]
        init_copy.remove(deleted_alt)

        rr_quantity = 0
        while(len(init_copy) > 0):
            j = 0
            while (init_copy[0] != new_copy[j]):
                if (verbose):
                    print("RR between " + str(init_copy[0]) + " and "
                          + str(new_copy[j]) + " when " + str(deleted_alt)
                          + " is deleted")

                rr_quantity += 1
                j += 1
            del init_copy[0]
            del new_copy[j]
        return rr_quantity

    def analyse_rr(self, verbose=False):
        """Compute the pair of alternatives for which rr occurs.

        Output:
            all_rr_instance - dictionary containing the pair of alternatives
                              which have had their rank reversed (key), and the
                              quantity of time this reversal happened (value).
        Note:
            * this function is similar to the one counting the number of rank
              reversals but is reimplemented here for more lisibility.
        """
        all_rr_instances = dict()
        for i in range(len(self.alternatives)):
            copy_alternatives = self.alternatives[:]
            del copy_alternatives[i]
            scores = self.compute_netflow(alternatives=copy_alternatives)
            ranks = self.compute_ranking(scores)
            """Since we applied the method on n-1 alternatives, the ranks
            will be in [0, n-1] instead of [0, n]"""
            for j in range(len(ranks)):
                if ranks[j] >= i:
                    ranks[j] += 1

            rr_instances = self.get_rr(self.ranking, ranks, i, verbose)

            for key in rr_instances:
                all_rr_instances[key] = \
                    all_rr_instances.get(key, 0) + rr_instances.get(key)
        return all_rr_instances

    def get_rr(self, init_ranking, new_ranking, deleted_alt, verbose=False):
        """Compute the number of rank reversal between two rankings.

        Inputs:
            init_ranking - ranking obtained with all alternatives
            new_ranking - ranking obtained when deleted_alt is removed from
                          the set of alternatives.
        """
        init_copy = init_ranking[:]
        new_copy = new_ranking[:]
        init_copy.remove(deleted_alt)

        rr_instances = dict()
        while(len(init_copy) > 0):
            j = 0
            while (init_copy[0] != new_copy[j]):
                if (verbose):
                    print("RR between " + str(init_copy[0]) + " and "
                          + str(new_copy[j]) + " when " + str(deleted_alt)
                          + " is deleted")

                # add occurrence to dict of rank reveSRPals
                a = max(new_copy[j], init_copy[0])
                b = min(new_copy[j], init_copy[0])
                rr_instances[(a, b)] = rr_instances.get((a, b), 0) + 1

                j += 1
            del init_copy[0]
            del new_copy[j]
        return rr_instances


class PrometheeMV(PrometheeII):

    """Use Promethee with missing values by replacing the pij."""

    def __init__(self, alternatives, seed=0, alt_num=-1, ceils=None,
                 weights=None, coefficients=None, method='median'):
        """Constructor."""
        if method == 'median':
            self.method = self.replace_by_median
        elif method == 'mean':
            self.method = self.replace_by_mean

        super().__init__(alternatives=alternatives, seed=seed, alt_num=alt_num,
                         ceils=ceils, coefficients=coefficients,
                         weights=weights)

    def compute_netflow(self, alternatives=None, weights=None, pref_funcs=None):
        """Compute the netflow of the alternatives.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            weights      - list of weights of the criteria.
            pref_functs  - list of the preference functions for all criteria.

        Output :
            netflows[i] = score of the ith alternative in alternatives(input)
        """
        if alternatives is None and pref_funcs is None and weights is None:
            Pi = self.pi
        else:
            Pi = self.compute_pairwise_pref(alternatives, weights, pref_funcs)

        if alternatives is None:
            alternatives = self.alternatives

        netflows = []
        if (len(alternatives) == 1):
                return [1]

        n = len(alternatives)
        for i in range(n):
            flow = 0
            for j in range(n):
                if i == j:
                    continue
                flow += Pi[i][j] - Pi[j][i]
            flow = flow / (len(alternatives) - 1)
            netflows.append(flow)
        print('ok')
        return netflows

    def compute_pairwise_pref(self, alternatives=None, weights=None,
                              pref_funcs=None):
        """Return the pairwise preference matrix Pi with missing evaluations.

        Inputs:
            alternatives - matrix composed of one list of evaluations for each
                           alternative.
            weights - list of weights of the criteria.
            pref_funcs - list of the preference functions for all criteria.

        The missing evaluations are handled by replacing Pc[a, '*'] with the
        mean/median of Pc[a, x] or by using the NNB.

        """
        if alternatives is None:
            alternatives = self.alternatives
        if weights is None:
            weights = self.weights
        if pref_funcs is None:
            pref_funcs = self.pref_functions

        P = self.compute_pairwise_comparisons(alternatives, pref_funcs)
        pi = [[0 for alt in alternatives] for alt in alternatives]
        for i, alti in enumerate(alternatives):
            for j, altj in enumerate(alternatives):
                for c in range(len(weights)):
                    weight = weights[c]
                    pi[i][j] += weight * P[c][i][j]
                    pi[j][i] += weight * P[c][j][i]
        return pi

    def compute_pairwise_comparisons(self, A, pref_functs):
        """Compute the Pcij comparions matrix."""
        n = len(A)
        k = len(A[0])
        P = [[[0 for i in range(n)] for j in range(n)] for t in range(k)]

        for c in range(k):
            for i in range(n):
                for j in range(i + 1, n):
                    if A[i][c] == NULL or A[j][c] == NULL:
                        P[c][i][j] = NULL
                        P[c][j][i] = NULL
                    else:
                        diff = A[i][c] - A[j][c]
                        P[c][i][j] = pref_functs[c].value(diff)
                        P[c][j][i] = pref_functs[c].value(-diff)
        self.replace_missing_comparisons(P, A)
        return P

    def replace_missing_comparisons(self, P, A):
        """Replace missing comparisons."""
        n = len(A)
        k = len(A[0])
        P_copy = copy.deepcopy(P)
        for c in range(k):
            Pc = P_copy[c]
            for i in range(n):
                for j in range(i + 1, n):
                    if Pc[i][j] == NULL:
                        if A[i][c] == NULL and A[j][c] == NULL:
                            P[c][i][j] = 0
                        elif A[j][c] == NULL:
                            P[c][j][i] = self.method(P_copy, i, c, 'col')
                            P[c][i][j] = self.method(P_copy, i, c, 'row')
                        else:
                            P[c][i][j] = self.method(P_copy, j, c, 'col')
                            P[c][j][i] = self.method(P_copy, j, c, 'row')

    def replace_by_median(self, P, y, c, direction):
        """Replace Pc(y,*),Pc(*,y) by median of Pc(y,x), Pc(x,y) for all x.

        Input
            direction : 'row' if Pc(y, *), 'col' if Pc(*, y) to replace
        """
        # print(P)
        # print('c = ' + str(c))
        # print('y = ' + str(y))
        # print('direction = ' + str(direction))
        if direction == 'row':
            values = [pyx for pyx in P[c][y] if pyx != NULL]
        else:
            values = [px[y] for px in P[c] if px[y] != NULL]
        return np.median(values)

    def replace_by_mean(self, P, y, c, direction):
        """Replace Pc(y,*),Pc(*,y) by mean of Pc(y,x), Pc(x,y) for all x.

        Input
            direction : 'row' if Pc(y, *), 'col' if Pc(*, y) to replace
        """
        # print(P)
        # print('c = ' + str(c))
        # print('y = ' + str(y))
        # print('direction = ' + str(direction))
        if direction == 'row':
            values = [pyx for pyx in P[c][y] if pyx != NULL]
        else:
            values = [px[y] for px in P[c] if px[y] != NULL]
        return np.mean(values)
