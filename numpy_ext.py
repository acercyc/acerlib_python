import numpy as np

def load_listOfArray(fName):
    d = np.load(fName)
    d.files.sort()
    d = [d[vName] for vName in d.files]
    return d

