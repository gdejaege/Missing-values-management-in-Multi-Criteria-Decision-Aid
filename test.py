import helpers


indices = list(range(6))

k_ss = helpers.powerset(indices)
k_ss = [ss for ss in k_ss
        if len(ss) > 1]      # to perfom tests on intervals!

print(k_ss)
