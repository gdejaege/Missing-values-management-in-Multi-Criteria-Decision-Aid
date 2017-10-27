"""Helper functions."""

from itertools import chain, combinations

def compute_ranking(scores):
    """Return the ranking given the scores.

    Input:
        scores[i] = score of the ith alternative

    Output:
        ranking[i] = index in 'scores' of the alternative ranked ith.
    """
    # sort k=range(...) in decreasing order of the netflows[k]
    ranking = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    return ranking


def printelem(elem, separator=" ", width=6):
    """Print an element inside a matrix."""
    if isinstance(elem, float) or isinstance(elem, int):
        template = "{:^#" + str(width) + "." + str(width-2) + "g}"
    else:
        template = "{:^" + str(width) + "}"
    print(template.format(elem), end=separator)


def printmatrix(M, separator=" ", width=6, offset=0):
    """Try to print a maximal 3D list in a more readable manner."""
    if isinstance(M, list) or isinstance(M, tuple):
        print(" "*offset + "[", end="")
        new_offset = offset + 2
        if isinstance(M[0], list) or isinstance(M[0], tuple):
            print()
        for el in M:
            printmatrix(el, separator=separator, width=width, offset=new_offset)
        if isinstance(M[0], list) or isinstance(M[0], tuple):
            print(" "*offset + "]")
        else:
            print("]")
    else:
        print(" ", end="")
        printelem(M, width=width)


def print_transpose(M, separator=" ", width=6, offset=0):
    """Print the transpose of a maximal 2D list in a more readable manner."""
    # Check depth
    depth = 0
    t = M
    while(isinstance(t, list) or isinstance(t, tuple)):
        depth += 1
        t = t[0]

    if depth == 1:
        new_M = [[el] for el in M]
        printmatrix(new_M, separator, width, offset)
    elif depth == 2:
        new_M = list(map(list, zip(*M)))
        printmatrix(new_M, separator, width, offset)
    else:
        print("Don't try to transpose a more than 2D array.")


def powerset(iterable):
    """Compute the powerset.

    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    Taken and modified from https://stackoverflow.com/questions/374626/how-can-
                                    i-find-all-the-subsets-of-a-set-with-
                                    exactly-n-elements
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    ps = chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))
    ps = [list(ss) for ss in ps if len(ss) > 0]
    return ps
