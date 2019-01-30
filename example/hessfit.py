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
import matplotlib
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory

HESSFILE = "hess/ts.log"
TEMPFILE = "conf.temp"
STATE_TEMPFILE = ["state_1.temp", "state_2.temp"]
VAR = np.array([-1.02178714e+01,  2.96289306e-01,  1.64063455e-01,  1.16490327e-02,
                4.81039490e+00,  5.57221210e-01,  1.07854875e-01,  3.67539345e+04,
                1.83496498e-01,  8.32166788e+04,  1.22943053e-02,  1.11986160e+02,
                1.79091272e-02,  1.05608702e+02,  5.25430179e+00,  1.77521684e+00,
                2.71316766e+00,  1.13375097e-01, -9.53444603e+03,  1.98967448e-01,
                5.73826376e+04,  1.57761737e-02,  8.59267473e+01,  3.47319534e-02,
                9.42591073e+01,  5.79449948e-03,  1.78091636e+00,  2.05894451e+00, ])

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


def genHessScore(xyz, hess, mass, template, state_templates=[], dx=0.00001):
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
        var = (calc_theta_p - theta_p) ** 2
        var_diag = np.diag(var).sum() / var.shape[0]
        var_offdiag = (var - np.diag(np.diag(var))).sum() / \
            (var.shape[0] ** 2 - var.shape[0])
        return 0.001 * var_diag + var_offdiag

    return valid


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


if __name__ == '__main__':
    xyz, hess = getGaussianHess(HESSFILE)
    mass = [12.011, 1.008, 1.008, 1.008, 35.453, 79.904]

    with open(TEMPFILE, "r") as f:
        template = Template("".join(f))

    state_templates = []
    for fname in STATE_TEMPFILE:
        with open(fname, "r") as f:
            state_templates.append([fname.split(".")[0], Template("".join(f))])

    tfunc = genHessScore(xyz, hess, mass, template, state_templates=state_templates)
#    drawPicture(xyzs, eners, grads, VAR, template,
#                state_templates=state_templates)
    drawHess(xyz, hess, mass, VAR, template, state_templates=state_templates)
    #traj = basinhopping(tfunc, VAR, niter=50, T=2.0, pert=2.5)
    min_result = optimize.minimize(tfunc, VAR, jac="2-point", hess="2-point", method='L-BFGS-B', options=dict(maxiter=1000, disp=True, gtol=0.01))
    print(min_result.x)
    drawHess(xyz, hess, mass, min_result.x, template, state_templates=state_templates)
    
    TEMPDIR.cleanup()
