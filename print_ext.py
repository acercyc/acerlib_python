def print_dict(d, numFormat='.3e'):
    for x in d:
        print('{}:\t'.format(x), end="")
        if isinstance(d[x], (list, tuple)):
            for xx in d[x]:
                print(('%' + numFormat + '\t') % xx, end="")
        else:
            print(('%' + numFormat + '\t') % (d[x]), end="")

