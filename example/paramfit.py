import sys
import os
import json
import pdb
import logging
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
VAR = np.array([-1.02178714e+01,  2.96289306e-01,  1.64063455e-01,  1.16490327e-02,
                 4.81039490e+00,  5.57221210e-01,  1.07854875e-01,  3.67539345e+04,
                 1.83496498e-01,  8.32166788e+04,  1.22943053e-02,  1.11986160e+02,
                 1.79091272e-02,  1.05608702e+02,  5.25430179e+00,  1.77521684e+00,
                 2.71316766e+00,  1.13375097e-01, -9.53444603e+03,  1.98967448e-01,
                 5.73826376e+04,  1.57761737e-02,  8.59267473e+01,  3.47319534e-02,
                 9.42591073e+01,  5.79449948e-03,  1.78091636e+00,  2.05894451e+00,])

TEMPDIR = TemporaryDirectory()
if len(sys.argv) == 1:
    print("LOG file needed.\nExit.")
    exit()
logging.basicConfig(filename=sys.argv[1], level=logging.INFO)


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
                    f.write(temp.render(var=np.abs(var)))
            # gen config file
            conf = json.loads(template.render(var=var))
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
            f.write(temp.render(var=np.abs(var)))

    conf = json.loads(template.render(var=var))
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


def basinhopping(score, var, niter=20, bounds=None, T=1.0, pert=7.0):
    newvar = np.zeros(var.shape)
    newvar[:] = var[:]
    posvar = np.zeros(var.shape)
    posvar[:] = newvar[:]
    posscore = np.inf
    traj = [[score(var), var]]
    for ni in range(niter):
        logging.info("Round %i. Start BFGS." % ni)
        min_result = optimize.minimize(score, newvar, jac="2-point", hess="2-point",
                                       method='L-BFGS-B', options=dict(maxiter=100, disp=True, gtol=0.1, maxls=10))
        logging.info("Result:  " + "  ".join("{}".format(_) for _ in min_result.x))
        t_score = score(min_result.x)
        if t_score < posscore or np.exp(- (t_score - posscore) / T) > np.random.random():
            logging.info("OLD: %.4f  NEW: %.4f accept"%(posscore, t_score))
            traj.append([t_score, min_result.x])
            posvar[:] = min_result.x[:]
            posscore = t_score

        else:
            logging.info("OLD: %.4f  NEW: %.4f reject"%(posscore, t_score))
        while True:
            newvar = posvar + (np.random.random(posvar.shape) * 2 - 1.0) * pert
            if not bounds or bounds(x_new=newvar):
                break
            else:
                continue
        logging.info("Set new var: " + "  ".join("{}".format(_) for _ in newvar))
    sorttraj = sorted(traj, key=lambda x: x[0])
    logging.info("Job finished. min f: {}  min var: {}".format(sorttraj[0][0], " ".join("{}".format(_) for _ in sorttraj[0][1])))
    return sorttraj


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
#    drawPicture(xyzs, eners, grads, VAR, template,
#                state_templates=state_templates)


    class MyBounds(object):

        def __init__(self, xmax=[], xmin=[]):
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)

        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin

    mybounds = MyBounds(xmax=100 * VAR, xmin=- VAR)

    traj = basinhopping(tfunc, VAR, niter=50, T=2.0, pert=2.5)
    #min_result = optimize.minimize(tfunc, VAR, jac="2-point", hess="2-point", method='L-BFGS-B', options=dict(maxiter=1000, disp=True, gtol=0.0001))

    drawPicture(xyzs, eners, grads, traj[0][1],
                template, state_templates=state_templates)
    TEMPDIR.cleanup()
