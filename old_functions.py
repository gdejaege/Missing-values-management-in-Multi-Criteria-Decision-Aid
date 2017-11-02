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



