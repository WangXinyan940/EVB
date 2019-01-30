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
from tempfile import TemporaryDirectory

QMDATA = "qm/"
TEMPFILE = "conf.temp"
STATE_TEMPFILE = ["state_1.temp", "state_2.temp"]
VAR = np.array([-6.457428454719952, 11.322013887117633, -3.102850900246132, 0.09955092345810203,
                -6.896181660007977, 2.0635627394368368, 3.8942164265663797, 36755.58895025612,
                -7.679705400769527, 83209.55839643197, 1.3639416920888712, 117.64547571663184,
                -1.2822632961541458, 106.6704379643974, 7.444835249934883, -2.4242364651414148,
                 3.680612219880417, 0.1838423264006372, -9537.648720433981, 0.22649091440551683,
                 57378.44028386577, -0.0033029558860510166, 21.813491712864778, -0.027380204572135948,
                 112.1508095011575, 13.304617729450348, 2.1571174365925194, 33.238039899626955])

var = np.array([ 1.58426889e+01, -3.24567490e+01, -3.15881775e+01,  3.75988210e+01,
  1.98389210e+01, -3.93109448e+01, -4.14876753e-02, -1.51170294e+01,
  1.90032664e-01, -2.46003778e+01, -2.43502128e+01,  3.15740874e+01,
 -6.24889685e+01,  3.09870718e+00, -2.05621358e+01, -1.17965579e+01,
  3.48064761e+01,  6.17902776e+00, -1.37958287e-01, -1.07909156e+00,
 -7.78078929e+00, -4.44864382e+01,  2.18746567e+00, -5.00271817e+01,
 -4.21610042e+00, -9.99491677e+01, -7.67836073e+01,  2.30073812e+00,])


TEMPDIR = TemporaryDirectory()


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


def genGradScore(xyzs, grads, template, state_templates=[]):
    """
    Generate score func.
    """
    def valid(var):
        """
        Return score::float
        """
        for name, temp in state_templates:
            with open("%s.xml" % name, "w") as f:
                f.write(temp.render(var=np.abs(var)))
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
        var_grad = np.sqrt(((calc_grad - ref_grad) ** 2).mean())
        return var_grad
    return valid


def genEnergyScore(xyzs, ener, template, state_templates=[]):
    """
    Generate score func.
    """
    def valid(var):
        """
        Return score::float
        """
        for name, temp in state_templates:
            with open("%s.xml" % name, "w") as f:
                f.write(temp.render(var=np.abs(var)))
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


def genTotalScore(xyzs, eners, grads, template, state_templates=[]):
    """
    Generate score func.
    """
    def valid(var):
        """
        Return score::float
        """
        # gen state files
        try:
            for name, temp in state_templates:
                with open("%s/%s.xml" % (TEMPDIR.name, name), "w") as f:
                    f.write(temp.render(var=np.abs(VAR)))
            # gen config file
            conf = json.loads(template.render(var=VAR))
            for n, fn in enumerate(state_templates):
                conf["diag"][n][
                    "parameter"] = "%s/%s.xml" % (TEMPDIR.name, fn[0])
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
        except:
            return 10000.0
    return valid


def drawPicture(xyzs, eners, grads, var, template, state_templates=[]):

    for name, temp in state_templates:
        with open("%s/%s.xml" % (TEMPDIR.name, name), "w") as f:
            f.write(temp.render(var=np.abs(VAR)))

    conf = json.loads(template.render(var=VAR))
    for n, fn in enumerate(state_templates):
        conf["diag"][n]["parameter"] = "%s/%s.xml" % (TEMPDIR.name, fn[0])
    H = evb.EVBHamiltonian(conf)
    calc_ener, calc_grad = [], []
    for n, xyz in enumerate(xyzs):
        energy, gradient = H.calcEnergyGrad(xyz)
        calc_ener.append(energy)
        calc_grad.append(gradient)
    calc_ener = np.array(
        [i.value_in_unit(unit.kilojoule / unit.mole) for i in calc_ener])
    calc_ener = calc_ener - calc_ener.max()
    ref_ener = np.array(
        [i.value_in_unit(unit.kilojoule / unit.mole) for i in eners])
    ref_ener = ref_ener - ref_ener.max()
    plt.plot([calc_ener.min() / 4.184, calc_ener.max() / 4.184], [
             calc_ener.min() / 4.184, calc_ener.max() / 4.184], c='black')
    plt.scatter(calc_ener / 4.184, ref_ener / 4.184, c="red")
    plt.xlabel("CALC ENERGY")
    plt.ylabel("REF ENERGY")
    #plt.plot(calc_ener, c="red")
    #plt.plot(ref_ener, c="black")
    # plt.xlabel("Sample")
    #plt.ylabel("Energy (kJ/mol)")
    plt.show()
    cmap = ["k", "r", "y", "g", "b", "m"]
    calc_grad = np.array(
        [i.value_in_unit(unit.kilojoule_per_mole / unit.angstrom) for i in calc_grad])
    ref_grad = np.array(
        [i.value_in_unit(unit.kilojoule_per_mole / unit.angstrom) for i in grads])
    plt.plot([calc_grad.min() / 4.184, calc_grad.max() / 4.184],
             [calc_grad.min() / 4.184, calc_grad.max() / 4.184])
    for n in range(calc_grad.shape[1]):
        plt.scatter(calc_grad[:, n, :].ravel() / 4.184, ref_grad[
                    :, n, :].ravel() / 4.184, c=cmap[n], label="%s" % n)
    plt.legend()
    plt.xlabel("CALC GRADIENT")
    plt.ylabel("REF GRADIENT")
    plt.show()


if __name__ == '__main__':
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

    efunc = genEnergyScore(xyzs, eners, template)
    gfunc = genGradScore(xyzs, grads, template)
    tfunc = genTotalScore(xyzs, eners, grads, template,
                          state_templates=state_templates)

    print(tfunc(VAR))
    drawPicture(xyzs, eners, grads, VAR,
                template, state_templates=state_templates)
    TEMPDIR.cleanup()
