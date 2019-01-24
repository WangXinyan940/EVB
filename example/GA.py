import sys
import os
import json
import pdb
import numpy as np
import simtk.unit as unit
from scipy import optimize
from jinja2 import Template
sys.path.append("..")
import evb


QMDATA = "qm/"
TEMPFILE = "conf.temp"
VAR_SHAPE = 7
VAR_MIN = -1e3
VAR_MAX = 1e3
INITSIZE = 500
REMAIN = 300
LOWMULT = 200
HIGHMULT = 100
NEWGEN = 200


def findline(text, parser):
    index = []
    for n, l in enumerate(text):
        if parser in l:
            index.append(n)
    return index


def getGaussianGradient(fname):
    """
    Return xyz, gradient.
    """
    with open(fname, "r") as f:
        text = f.readlines()

    start = findline(text, "Input orientation")[0] + 5
    end = findline(text, "Distance matrix (angstroms)")[0] - 1
    data = text[start:end]
    data = [i.strip().split() for i in data]
    xyz = [[float(j) for j in i[-3:]] for i in data]
    xyz = unit.Quantity(value=np.array(xyz), unit=unit.angstrom)

    start = findline(text, "Forces (Hartrees/Bohr)")[0] + 3
    end = findline(text, "Cartesian Forces")[0] - 1
    data = text[start:end]
    data = [i.strip().split() for i in data]
    grad = [[- float(j) * 627.5 for j in i[-3:]] for i in data]
    grad = unit.Quantity(value=np.array(
        grad), unit=unit.kilocalorie_per_mole / unit.bohr)

    return xyz, grad


def genScoreFunc(xyzs, grads, template):
    """
    Generate score func.
    """
    def valid(var):
        """
        Return score::float
        """
        # gen config file
        conf = json.loads(template.render(var=var))
        # gen halmitonian
        H = evb.EVBHamiltonian(conf)
        # calc forces
        error = []
        try:
            for n, xyz in enumerate(xyzs):
                energy, gradient = H.calcEnergyGrad(xyz)
                error.append(gradient - grads[n])
            # compare
            error = [np.abs(i.value_in_unit(
                unit.kilojoule_per_mole / unit.angstrom)) for i in error]
            return sum(error).sum() / len(error) / error[0].shape[0] / error[0].shape[1]
        except:
            return 1e5
    return valid


def combine(a, b, T=0.1):
    res = np.zeros(a.shape)
    for i in range(a.shape[0]):
        res[i] = a[i] if np.random.random() > 0.5 else b[i]
        if np.random.random() < T:
            res[i] += (np.random.random() * 2 - 1) * 10.0
    return res


def randomGen(shape):
    return (np.random.random(shape) - 0.5) * 2e3


def main():
    fnames = os.listdir(QMDATA)
    xyzs, grads = [], []
    for fname in fnames:
        txyz, tgrad = getGaussianGradient(QMDATA + fname)
        xyzs.append(txyz)
        grads.append(tgrad)

    with open(TEMPFILE, "r") as f:
        template = Template("".join(f))

    validfunc = genScoreFunc(xyzs, grads, template)

    var_set = [randomGen((VAR_SHAPE,)) for _ in range(INITSIZE)]
    while True:
        score_set = [[v,validfunc(v)] for v in var_set]
        rem = sorted(score_set, key=lambda v:v[1])[:REMAIN]
        print("E:", rem[0][1])
        print("PARAM:")
        print(rem[0][0],"\n")
        var_set = []
        for _ in range(LOWMULT):
            ri, rj = np.random.randint(0,REMAIN), np.random.randint(0,REMAIN)
            var_set.append(combine(rem[ri][0],rem[rj][0], T=0.1))
        for _ in range(HIGHMULT):
            ri, rj = np.random.randint(0,REMAIN), np.random.randint(0,REMAIN)
            var_set.append(combine(rem[ri][0],rem[rj][0], T=0.25))
        for _ in range(NEWGEN):
            var_set.append(randomGen(var_set[0].shape))



if __name__ == '__main__':
    main()
