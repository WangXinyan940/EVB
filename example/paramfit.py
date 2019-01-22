import sys
import os
import numpy as np
import simtk.unit as unit
from scipy import optimize
from jinja2 import Template
sys.path.append("..")
import evb

QMDATA = "qm/"
TEMPFILE = "conf."


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
    def valid(var):
        """
        Return score::float
        """
        #gen config file
        #gen halmitonian
        H = evb.EVBHamiltonian(conf)
        #calc forces
        error = []
        for n, xyz in enumerate(xyzs):
            energy, gradient = H.calcEnergyGrad(xyz)
            error.append(gradient - grads[n])
        #compare
        error = [i.value for i in error]
    return valid


def main():
    global QMDATA
    global TEMPFILE
    global VAR

    fnames = os.listdir(QMDATA)
    xyzs, grads = [], []
    for fname in fnames:
        txyz, tgrad = getGaussianGradient(QMDATA+fname)
        xyzs.append(txyz)
        grads.append(tgrad)

    with open(TEMPFILE, "r") as f:
        template = Template("".join(f))

    validfunc = genScoreFunc(xyzs, grads, template)
    min_result = optimize.minimize(validfunc, var, method='L-BFGS-B', jac=True, options=dict(maxiter=1000, disp=True, gtol=0.01))


if __name__ == '__main__':
    main()