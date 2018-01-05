"""Make fake datasets."""

import numpy as np


def main(name, f, x_range, noise=False):
    """Make a fake dataset with f(x) for n x in x_range."""
    output_file = 'data/fake/' + name + '.csv'
    evs = [[x for x in x_range], [f(x) for x in x_range]]
    var = np.std(evs[1])

    evs[1] = [x + np.random.normal(0, var/10) for x in evs[1]]

    f = open(output_file, 'w')

    f.write("Alternatives \n######\n")

    for x, fx in zip(*evs):
        f.write(str(str(x) + ', ' + str(fx) + '\n'))

    f.close()
