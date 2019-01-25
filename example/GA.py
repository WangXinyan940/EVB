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
STATE_TEMPFILE = ["state_1.temp", "state_2.temp"]
VIB_UNIT = 0.001
LIM = 50.0
INITSIZE = 300
REMAIN = 150
TOP = 20
LOWMULT = 130
HIGHMULT = 50
NEWGEN = 100

VAR_INIT = np.array([-35,  0.587,  0.5664, -0.0232886395,
                     2.4108,  1.7586,  0.10749,  167845.344,
                     0.1813,  77152.96,  0.010051,  115.705145,
                     0.0283,  112.2809,  8.359835,  13.6440055,
                     0.1076,  162364.304,  0.1900658,  59835.384,
                     0.0624,  94.8828612,  0.1157697,  99.824709,
                     8.4451,  13.6449845])

VAR_LIMIT = ((-4.5e1, 0.000), (0.000, 2.000), (0.000, 2.000), (-1.000, 1.000),
             (0.000, 5.000), (0.000, 10.000), (0.080, 0.120), (1.4e5, 1.9e5),
             (0.140, 0.220), (5.0e4, 8.5e4), (0.005, 0.020), (100.0, 150.0),
             (0.010, 0.050), (100.0, 150.0), (5.000, 13.00), (10.00, 16.00),
             (0.080, 0.120), (1.4e5, 1.9e5), (0.140, 0.220), (4.0e4, 8.5e4),
             (0.020, 0.100), (80.00, 110.0), (0.080, 0.150), (80.00, 120.0),
             (5.000, 13.00), (10.00, 16.00))


def findline(text, parser):
    index = []
    for n, l in enumerate(text):
        if parser in l:
            index.append(n)
    return index


def getGaussianEnergyGradient(fname):
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

    eline = text[findline(text, "SCF Done:")[0]]
    energy = unit.Quantity(value=float(eline.strip().split()[
                           4]) * 627.5, unit=unit.kilocalorie_per_mole)

    start = findline(text, "Forces (Hartrees/Bohr)")[0] + 3
    end = findline(text, "Cartesian Forces")[0] - 1
    data = text[start:end]
    data = [i.strip().split() for i in data]
    grad = [[- float(j) * 627.5 for j in i[-3:]] for i in data]
    grad = unit.Quantity(value=np.array(
        grad), unit=unit.kilocalorie_per_mole / unit.bohr)

    return xyz, energy, grad


def genTotalScore(xyzs, eners, grads, template, state_templates=[]):
    """
    Generate score func.
    """
    def valid(var):
        """
        Return score::float
        """
        # gen state files
        for name, temp in state_templates:
            with open("%s.xml" % name, "w") as f:
                f.write(temp.render(var=np.abs(var)))
        # gen config file
        conf = json.loads(template.render(var=var))
        # gen halmitonian
        H = evb.EVBHamiltonian(conf)
        # calc forces
        calc_ener, calc_grad = [], []
        for n, xyz in enumerate(xyzs):
            energy, gradient = H.calcEnergyGrad(xyz)
            calc_ener.append(energy)
            calc_grad.append(gradient)
        # compare
        calc_ener = np.array(
            [i.value_in_unit(unit.kilojoule / unit.mole) for i in calc_ener])
        ref_ener = np.array(
            [i.value_in_unit(unit.kilojoule / unit.mole) for i in eners])
        var_ener = np.sqrt(
            (np.abs((calc_ener - calc_ener.max()) - (ref_ener - ref_ener.max())) ** 2).sum())

        calc_grad = np.array([i.value_in_unit(
            unit.kilojoule_per_mole / unit.angstrom) for i in calc_grad]).ravel()
        ref_grad = np.array([i.value_in_unit(
            unit.kilojoule_per_mole / unit.angstrom) for i in grads]).ravel()
        var_grad = np.sqrt(((calc_grad - ref_grad) ** 2).mean())
        return var_grad + var_ener
    return valid


def combine(a, b, T=0.1):
    res = np.zeros(a.shape)
    for i in range(a.shape[0]):
        res[i] = a[i] if np.random.random() > 0.5 else b[i]
        if np.random.random() < T:
            res[i] = res[i] * (1 + (np.random.random() - 0.5)
                               * 2 * 0.2)
    return res


def randomGen(shape):
    res = np.zeros(VAR_INIT.shape)
    for n in range(res.shape[0]):
        imin, imax = VAR_LIMIT[n]
        res[n] = np.random.random() * (imax - imin) + imin
    return res


def main():
    fnames = os.listdir(QMDATA)
    xyzs, eners, grads = [], [], []
    for fname in fnames:
        txyz, tener, tgrad = getGaussianEnergyGradient(QMDATA + fname)
        xyzs.append(txyz)
        eners.append(tener)
        grads.append(tgrad)

    with open(TEMPFILE, "r") as f:
        template = Template("".join(f))

    state_templates = []
    for fname in STATE_TEMPFILE:
        with open(fname, "r") as f:
            state_templates.append([fname.split(".")[0], Template("".join(f))])

    validfunc = genTotalScore(
        xyzs, eners, grads, template, state_templates=state_templates)

    var_set = [randomGen(VAR_INIT.shape) for _ in range(INITSIZE)]
    while True:
        score_set = [[v, validfunc(v)] for v in var_set]
        rem = sorted(score_set, key=lambda v: v[1])[:REMAIN]
        print("E:", rem[0][1])
        print("PARAM:")
        print(rem[0][0], "\n")
        var_set = []
        for i, s in rem[:TOP]:
            var_set.append(i)
        for _ in range(LOWMULT):
            ri, rj = np.random.randint(0, REMAIN), np.random.randint(0, REMAIN)
            var_set.append(combine(rem[ri][0], rem[rj][0], T=0.1))
        for _ in range(HIGHMULT):
            ri, rj = np.random.randint(0, REMAIN), np.random.randint(0, REMAIN)
            var_set.append(combine(rem[ri][0], rem[rj][0], T=0.25))
        for _ in range(NEWGEN):
            var_set.append(randomGen(var_set[0].shape))


if __name__ == '__main__':
    main()
