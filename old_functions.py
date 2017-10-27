### Missing values:
def extrapolate_dominance_sets(dominating_alts, dominated_alts, c):
    """Return average of worst dominant and best dominated."""
    dominated_a_c = [a[c] for a in dominated_alts]
    dominating_a_c = [a[c] for a in dominating_alts]

    if dominating_alts == [] and dominated_alts == []:
        return -1
    elif dominated_alts == []:
        return min(dominating_a_c)
    elif dominating_alts == []:
        return max(dominated_a_c)
    else:
        return (max(dominated_a_c) + min(dominating_a_c))/2


def dominance_best_criteria_subset(alt_tr, c):
    """Find the best subset of criteria to use the dominance idea."""
    indices = list(range(len(alt_tr[0])))
    indice_subsets = helpers.powerset(indices)
    indices_subsets = [ss for ss in indices_subsests
                       if c not in ss and len(ss) > 0]

    MSE = []
    for ss in indices_subsets:
        error = dom_mse(alt_tr, c, ss)
        MSE.append(error)

    best_ss = ss[MSE.index(min(MSE))]
    return best_ss


def dom_mse(alt_tr, c, ss):
    """Evaluate MSE of the dominance estimation using criteria in ss."""
    MSE = 0
    for i in range(len(alt_tr)):
        ev = alt_tr[i][c]
        new_alts = copy.deepcopy(alt_tr)
        new_alts[i][c] = NULL
        new_alts = [[a[k] for k in ss] for a in new_alts]

        new_alts = replace_by_dominance(new_alts, best=False)
        MSE += (ev - new_alts[i][c])**2
    return MSE/len(alt_tr)



