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
STATE_TEMPFILE = ["state_1.temp", "state_2.temp"]
VAR = np.array([-35.000,  0.5870,  0.5664, -0.0232886395,
                 2.4108,  1.7586,  0.10749,  167845.344,
                 0.1813,  77152.96,  0.010051,  115.705145,
                 0.0283,  112.2809,  8.359835,  13.6440055,
                 0.1076,  162364.304,  0.1900658,  59835.384,
                 0.0624,  94.8828612,  0.1157697,  99.824709,
                 8.4451,  13.6449845])


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
            with open("%s.xml"%name, "w") as f:
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
            with open("%s.xml"%name, "w") as f:
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
        for name, temp in state_templates:
            with open("%s.xml"%name, "w") as f:
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


def drawPicture(xyzs, eners, grads, var, template, state_templates=[]):

    for name, temp in state_templates:
        with open("%s.xml"%name, "w") as f:
            f.write(temp.render(var=np.abs(var)))

    conf = json.loads(template.render(var=var))
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
    plt.plot([calc_ener.min(), calc_ener.max()], [
             calc_ener.min(), calc_ener.max()], c='black')
    plt.scatter(calc_ener, ref_ener, c="red")
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
    plt.plot([calc_grad.min(), calc_grad.max()],
             [calc_grad.min(), calc_grad.max()])
    for n in range(calc_grad.shape[1]):
        plt.scatter(calc_grad[:, n, :].ravel(), ref_grad[
                    :, n, :].ravel(), c=cmap[n], label="%s" % n)
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
    tfunc = genTotalScore(xyzs, eners, grads, template, state_templates=state_templates)
    drawPicture(xyzs, eners, grads, VAR, template, state_templates=state_templates)
    def print_func(x, f, accepted):
        print("at minimum %.4f accepted %d"%(f, int(accepted)))
    min_result = optimize.basinhopping(tfunc, VAR, minimizer_kwargs={"method":"L-BFGS-B", "jac":"2-point"}, niter=1, callback=print_func)
    #min_result = optimize.minimize(tfunc, VAR, jac="2-point", hess="2-point", method='L-BFGS-B', options=dict(maxiter=1000, disp=True, gtol=0.0001))
    print(min_result)
    
    drawPicture(xyzs, eners, grads, min_result.x, template, state_templates=state_templates)
