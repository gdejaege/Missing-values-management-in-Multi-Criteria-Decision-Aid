"""Make fake datasets."""

def main(name, f, x_range):
    """Makes a fake dataset with f(x) for n x in x_range."""
    output_file = 'data/fake/' + name + '.csv'
    evs = [[x for x in x_range], [f(x) for x in x_range]]

    f = open(output_file, 'w')

    for x, fx in zip(*evs):
        f.write(str(str(x) + ', ' + str(fx) + '\n'))

    f.close()
