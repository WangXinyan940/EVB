import sys
import json
import logging
import numpy as np
import simtk.unit as unit
from scipy import optimize
from jinja2 import Template
sys.path.append("..")
import evb
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory
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


def getGaussianHess(fname):
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

    start = findline(text, "The second derivative matrix")[0] + 1
    end = findline(text, "ITU=  0")[0]
    data = text[start:end]
    data = ["{:<72}".format(i) for i in data]
    data = [[i[:21], i[21:31], i[31:41], i[41:51], i[51:61], i[61:71]]
            for i in data]
    data = [[j.strip() if j.strip() else "0.00" for j in i] for i in data]
    ntot = 0
    for line in data[1:]:
        if line[1][0] not in "XYZ":
            ntot += 1
        else:
            break
    hess = np.zeros((ntot, ntot))
    ref = 0
    shift = 0
    for line in data[1:]:
        if line[1][0] in "XYZ":
            ref += 5
            shift = 0
            continue
        for n, item in enumerate(line[1:]):
            if ref + n >= ntot:
                break
            hess[ref + shift, ref + n] = float(item)
            hess[ref + n, ref + shift] = float(item)
        shift += 1
    hess = unit.Quantity(
        value=hess * 627.5, unit=unit.kilocalorie_per_mole / unit.bohr / unit.bohr)

    return xyz, hess


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


def genEnerGradScore(xyzs, eners, grads, template, state_templates=[], a_ener = 1.00, a_grad = 1.00):
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
            return a_grad * var_grad + a_ener * var_ener
        except:
            return 10000.0
    return valid


def genHessScore(xyz, hess, mass, template, state_templates=[], dx=0.00001, a_diag = 1.00, a_offdiag = 1.00):
    """
    Generate score func.
    """
    # mat decomp
    mass_mat = []
    for i in mass:
        mass_mat.append(i)
        mass_mat.append(i)
        mass_mat.append(i)
    mass_mat = np.diag(1. / np.sqrt(mass_mat))
    hess_v = hess.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom ** 2)
    theta = np.dot(mass_mat, np.dot(hess_v, mass_mat))
    qe, qv = np.linalg.eig(theta)
    qvI = np.linalg.inv(qv)
    theta_p = np.dot(qvI, np.dot(theta, qv))

    def valid(var):
        """
        Return score::float
        """
        # gen state files

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
        # calc hess (unit in kJ / mol / A^2)
        oxyz = xyz.value_in_unit(unit.angstrom).ravel()
        dxyz = np.eye(oxyz.shape[0])
        calc_hess = np.zeros(dxyz.shape)
        for gi in range(dxyz.shape[0]):
            txyz = unit.Quantity(
                value=(oxyz + dxyz[:, gi] * dx).reshape((-1, 3)), unit=unit.angstrom)
            tep, tgp = H.calcEnergyGrad(txyz)
            txyz = unit.Quantity(
                value=(oxyz - dxyz[:, gi] * dx).reshape((-1, 3)), unit=unit.angstrom)
            ten, tgn = H.calcEnergyGrad(txyz)
            calc_hess[:, gi] = (
                tgp - tgn).value_in_unit(unit.kilocalorie_per_mole / unit.angstrom).ravel() / 2.0 / dx
        calc_theta = np.dot(mass_mat, np.dot(calc_hess, mass_mat))
        # change basis
        calc_theta_p = np.dot(qvI, np.dot(calc_theta, qv))

        vib_qm, vib_mm = np.diag(theta_p), np.diag(calc_theta_p)
        vib_qm = unit.Quantity(vib_qm, unit.kilocalorie_per_mole / unit.angstrom ** 2 / unit.amu)
        vib_mm = unit.Quantity(vib_mm, unit.kilocalorie_per_mole / unit.angstrom ** 2 / unit.amu)
        vib_qm = vib_qm.value_in_unit(unit.joule / unit.meter ** 2 / unit.kilogram)
        vib_mm = vib_mm.value_in_unit(unit.joule / unit.meter ** 2 / unit.kilogram)
        vib_qm = np.sqrt(np.abs(vib_qm)) / 2. / np.pi / 2.99792458e10 * np.sign(vib_qm)
        vib_mm = np.sqrt(np.abs(vib_mm)) / 2. / np.pi / 2.99792458e10 * np.sign(vib_mm)

        var = (calc_theta_p - theta_p) ** 2
        var_diag = ((vib_qm - vib_mm) ** 2).sum() / vib_mm.shape[0]
        var_offdiag = (var - np.diag(np.diag(var))).sum() / \
            (var.shape[0] ** 2 - var.shape[0])
        return a_diag * var_diag + a_offdiag * var_offdiag

    return valid


def drawEnergy(xyzs, eners, var, template, state_templates=[]):

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


def drawGradient(xyzs, grads, var, template, state_templates=[]):

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
    calc_grad = np.array(
        [i.value_in_unit(unit.kilojoule_per_mole / unit.angstrom) for i in calc_grad])
    ref_grad = np.array(
        [i.value_in_unit(unit.kilojoule_per_mole / unit.angstrom) for i in grads])
    plt.plot([calc_grad.min(), calc_grad.max()],
             [calc_grad.min(), calc_grad.max()])
    for n in range(calc_grad.shape[1]):
        plt.scatter(calc_grad[:, n, :].ravel(), ref_grad[
                    :, n, :].ravel(), label="%s" % n)
    plt.legend()
    plt.xlabel("CALC GRADIENT")
    plt.ylabel("REF GRADIENT")
    plt.show()


def drawHess(xyz, hess, mass, var, template, state_templates=[], dx=0.00001):
    mass_mat = []
    for i in mass:
        mass_mat.append(i)
        mass_mat.append(i)
        mass_mat.append(i)
    mass_mat = np.diag(1. / np.sqrt(mass_mat))
    hess_v = hess.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom ** 2)
    theta = np.dot(mass_mat, np.dot(hess_v, mass_mat))
    qe, qv = np.linalg.eig(theta)
    qvI = np.linalg.inv(qv)
    theta_p = np.dot(qvI, np.dot(theta, qv))

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
    # calc hess (unit in kJ / mol / A^2)
    oxyz = xyz.value_in_unit(unit.angstrom).ravel()
    dxyz = np.eye(oxyz.shape[0])
    calc_hess = np.zeros(dxyz.shape)
    for gi in range(dxyz.shape[0]):
        txyz = unit.Quantity(
            value=(oxyz + dxyz[:, gi] * dx).reshape((-1, 3)), unit=unit.angstrom)
        tep, tgp = H.calcEnergyGrad(txyz)
        txyz = unit.Quantity(
            value=(oxyz - dxyz[:, gi] * dx).reshape((-1, 3)), unit=unit.angstrom)
        ten, tgn = H.calcEnergyGrad(txyz)
        calc_hess[:, gi] = (
            tgp - tgn).value_in_unit(unit.kilocalorie_per_mole / unit.angstrom).ravel() / 2.0 / dx
    calc_theta = np.dot(mass_mat, np.dot(calc_hess, mass_mat))
    # change basis
    calc_theta_p = np.dot(qvI, np.dot(calc_theta, qv))
    var = (calc_theta_p - theta_p) ** 2
    f = plt.imshow(var)
    plt.colorbar(f)
    plt.show()
    vib_qm, vib_mm = np.diag(theta_p), np.diag(calc_theta_p)
    vib_qm = unit.Quantity(vib_qm, unit.kilocalorie_per_mole / unit.angstrom ** 2 / unit.amu)
    vib_mm = unit.Quantity(vib_mm, unit.kilocalorie_per_mole / unit.angstrom ** 2 / unit.amu)
    vib_qm = vib_qm.value_in_unit(unit.joule / unit.meter ** 2 / unit.kilogram)
    vib_mm = vib_mm.value_in_unit(unit.joule / unit.meter ** 2 / unit.kilogram)
    vib_qm = np.sqrt(np.abs(vib_qm)) / 2. / np.pi / 2.99792458e10 * np.sign(vib_qm)
    vib_mm = np.sqrt(np.abs(vib_mm)) / 2. / np.pi / 2.99792458e10 * np.sign(vib_mm)
    plt.scatter(vib_qm, vib_mm)
    vmin = min([vib_qm.min(),vib_mm.min()])
    vmax = max([vib_qm.max(),vib_mm.max()])
    plt.plot([vmin, vmax], [vmin, vmax], c="black", ls="--")
    plt.xlabel("QM Freq")
    plt.ylabel("FF Freq")
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
        logging.info("Result:  " + "  ".join("{}".format(_)
                                             for _ in min_result.x))
        t_score = score(min_result.x)
        if t_score < posscore or np.exp(- (t_score - posscore) / T) > np.random.random():
            logging.info("OLD: %.4f  NEW: %.4f accept" % (posscore, t_score))
            traj.append([t_score, min_result.x])
            posvar[:] = min_result.x[:]
            posscore = t_score

        else:
            logging.info("OLD: %.4f  NEW: %.4f reject" % (posscore, t_score))
        while True:
            newvar = posvar + (np.random.random(posvar.shape) * 2 - 1.0) * pert
            if not bounds or bounds(x_new=newvar):
                break
            else:
                continue
        logging.info("Set new var: " + "  ".join("{}".format(_)
                                                 for _ in newvar))
    sorttraj = sorted(traj, key=lambda x: x[0])
    logging.info("Job finished. min f: {}  min var: {}".format(
        sorttraj[0][0], " ".join("{}".format(_) for _ in sorttraj[0][1])))
    return sorttraj


