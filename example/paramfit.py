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
VAR = np.array([9.31456673, 11.47613159, -1.25352744,  0.41868175, -3.44075693, 13.10103445,  1.57885877])


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
        for n, xyz in enumerate(xyzs):
            energy, gradient = H.calcEnergyGrad(xyz)
            error.append(gradient - grads[n])
        # compare
        error = [np.abs(i.value_in_unit(
            unit.kilojoule_per_mole / unit.angstrom)) for i in error]
        return sum(error).sum() / len(error) / error[0].shape[0] / error[0].shape[1]
    return valid


def main():
    """
    Main function to parameterize.
    """
    global QMDATA
    global TEMPFILE
    global VAR

    fnames = os.listdir(QMDATA)
    xyzs, grads = [], []
    for fname in fnames:
        txyz, tgrad = getGaussianGradient(QMDATA + fname)
        xyzs.append(txyz)
        grads.append(tgrad)

    with open(TEMPFILE, "r") as f:
        template = Template("".join(f))

    validfunc = genScoreFunc(xyzs, grads, template)
    min_result = optimize.minimize(
        validfunc, VAR, jac="2-point", method='L-BFGS-B', options=dict(maxiter=1000, disp=True, gtol=0.01))
    print(min_result)


if __name__ == '__main__':
    main()