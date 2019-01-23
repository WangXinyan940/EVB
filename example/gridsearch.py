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
import matplotlib.pyplot as plt

QMDATA = "qm/"
TEMPFILE = "conf.temp"
VAR = np.array([-15.0,  0.96,  0.53, 10.00, -0.00045571])


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


def genGradScore(xyzs, grads, template):
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
        calc_grad = []
        for n, xyz in enumerate(xyzs):
            energy, gradient = H.calcEnergyGrad(xyz)
            calc_grad.append(gradient)
        # compare
        calc_grad = np.array([i.value_in_unit(
            unit.kilojoule_per_mole / unit.angstrom) for i in calc_grad]).ravel()
        ref_grad = np.array([i.value_in_unit(
            unit.kilojoule_per_mole / unit.angstrom) for i in grads]).ravel()
        var_grad = ((calc_grad - ref_grad) ** 2).mean()
        return var_grad
    return valid


def genEnergyScore(xyzs, ener, template):
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
        calc_ener = []
        for n, xyz in enumerate(xyzs):
            energy = H.calcEnergy(xyz)
            calc_ener.append(energy)
        # compare
        calc_ener = np.array(
            [i.value_in_unit(unit.kilojoule / unit.mole) for i in calc_ener])
        ref_ener = np.array(
            [i.value_in_unit(unit.kilojoule / unit.mole) for i in ener])
        return np.sqrt((np.abs((calc_ener - calc_ener.max()) - (ref_ener - ref_ener.max())) ** 2).mean())
    return valid


def genTotalScore(xyzs, eners, grads, template):
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
        return var_grad + var_ener * 0.5
    return valid

def drawPicture(xyzs, eners, grads, var, template):
    conf = json.loads(template.render(var=var))
    H = evb.EVBHamiltonian(conf)
    calc_ener, calc_grad = [], []
    for n, xyz in enumerate(xyzs):
        energy, gradient = H.calcEnergyGrad(xyz)
        calc_ener.append(energy)
        calc_grad.append(gradient)
    calc_ener = np.array([i.value_in_unit(unit.kilojoule / unit.mole) for i in calc_ener])
    calc_ener = calc_ener - calc_ener.max()
    ref_ener = np.array([i.value_in_unit(unit.kilojoule / unit.mole) for i in eners])
    ref_ener = ref_ener - ref_ener.max()
    plt.plot([calc_ener.min(),calc_ener.max()],[calc_ener.min(),calc_ener.max()],c='black')
    plt.scatter(calc_ener, ref_ener,c="red")
    plt.xlabel("CALC ENERGY")
    plt.ylabel("REF ENERGY")
    plt.show()

    calc_grad = np.array([i.value_in_unit(unit.kilojoule_per_mole / unit.angstrom) for i in calc_grad]).ravel()
    ref_grad = np.array([i.value_in_unit(unit.kilojoule_per_mole / unit.angstrom) for i in grads]).ravel()
    plt.plot([calc_grad.min(),calc_grad.max()],[calc_grad.min(),calc_grad.max()])
    plt.scatter(calc_grad, ref_grad)
    plt.xlabel("CALC GRADIENT")
    plt.ylabel("REF GRADIENT")
    plt.show()



def main():
    """
    Main function to parameterize.
    """
    global QMDATA
    global TEMPFILE
    global VAR

    fnames = os.listdir(QMDATA)
    xyzs, eners, grads = [], [], []
    for fname in fnames:
        txyz, tener, tgrad = getGaussianEnergyGradient(QMDATA + fname)
        xyzs.append(txyz)
        eners.append(tener)
        grads.append(tgrad)

    with open(TEMPFILE, "r") as f:
        template = Template("".join(f))

    efunc = genEnergyScore(xyzs, eners, template)
    gfunc = genGradScore(xyzs, grads, template)
    tfunc = genTotalScore(xyzs, eners, grads, template)

    v1 = np.linspace(-20.0,-0.0, 20)
    v2 = np.linspace(0.0,2.0, 20)
    v3 = np.linspace(0.0,1.0, 20)
    v4 = np.linspace(0.0,20.0, 20)
    v5 = np.linspace(-0.2,0, 20)

    V1, V2, V3, V4, V5 = np.meshgrid(v1, v2, v3, v4, v5)
    vmin = None
    emin = 99999.0
    for i1 in range(20):
        for i2 in range(20):
            for i3 in range(20):
                for i4 in range(20):
                    for i5 in range(20):
                        var = np.array([V1[i1,i2,i3,i4,i5],V2[i1,i2,i3,i4,i5],V3[i1,i2,i3,i4,i5],V4[i1,i2,i3,i4,i5],V5[i1,i2,i3,i4,i5]])
                        e = tfunc(var)
                        if e < emin:
                            emin = e
                            vmin = var
                            print("E:",e)
                            print("VAR:")
                            print(var)

    drawPicture(xyzs, eners, grads, vmin, template)




if __name__ == '__main__':
    main()
