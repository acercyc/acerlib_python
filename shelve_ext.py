from shelve import *


def save(filename, key, var):
    fid = open(filename, flag='c')
    fid[key] = var
    fid.close()


def load(filename, key):
    fid = open(filename, flag='c')
    try:
        var = fid[key]
    except KeyError:
        print('No data')
        var = []
    fid.close()
    return var


def ls(filename):
    fid = open(filename, flag='c')
    l = list(fid.keys())
    fid.close()
    return l
