#!python
# pylint: disable-msg=R0902,R0913,R0914

"""
MS-EVB engine to calculate EVB energy and force.
"""
import os
import json
import sys
import logging
import numpy as np
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit


def distance(a, b):
    """
    Calculate distance between point a and b.
    """
    return np.sqrt(np.power(a - b, 2).sum())


def angle(a, b, c):
    """
    Calculate a-b-c angle in radius.
    """
    v1 = a - b
    v2 = c - b
    r1 = np.sqrt(np.power(v1, 2).sum())
    r2 = np.sqrt(np.power(v2, 2).sum())
    return np.arccos(np.dot(v1, v2) / r1 / r2)


def dihedral(a, b, c, d):
    """
    Calculate the dihedral angle between abc surface and bcd surface (in radius). 
    """
    na = np.cross(c - b, a - b)
    nb = np.cross(b - c, d - c)
    return 0.5 * np.pi - np.arccos(np.abs(np.dot(na, nb)) / np.sqrt(np.power(na, 2).sum()) / np.sqrt(np.power(nb, 2).sum()))


def gradDistance(a, b):
    """
    Calculate the a-b distance and the gradient of a-b distance.
    """
    r = np.sqrt(np.power(a - b, 2).sum())
    ga = (a - b) / r
    return r, ga, - ga


def gradAngle(a, b, c):
    """
    Calculate the a-b-c angle and the gradient.
    WTF, gb is wrong.
    """
    v1 = a - b
    v2 = c - b
    r1 = np.sqrt(np.power(v1, 2).sum())
    r2 = np.sqrt(np.power(v2, 2).sum())
    dt = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    und = np.sqrt(1.0 - dt * dt / r1 ** 2 / r2 ** 2)
    ga = - (v2 / r1 / r2 - v1 * dt / r1 ** 3 / r2) / und
    gb = - (v2 * dt / r1 / r2 ** 3 - (v1 + v2) /
            r1 / r2 + v1 * dt / r1 ** 3 * r2) / und
    gc = - (v1 / r1 / r2 - v2 * dt / r1 / r2 ** 3) / und
    return np.arccos(dt / r1 / r2), ga, gb, gc


def numGradAngle(a, b, c, dt=0.00001):
    dx, dy, dz = np.array([dt, 0.0, 0.0]), np.array(
        [0.0, dt, 0.0]), np.array([0.0, 0.0, dt])
    ga, gb, gc = np.zeros(dx.shape), np.zeros(dx.shape), np.zeros(dx.shape)
    theta = angle(a, b, c)
    ga[0] = (angle(a + dx, b, c) - angle(a - dx, b, c)) / 2.0 / dt
    ga[1] = (angle(a + dy, b, c) - angle(a - dy, b, c)) / 2.0 / dt
    ga[2] = (angle(a + dz, b, c) - angle(a - dz, b, c)) / 2.0 / dt
    gb[0] = (angle(a, b + dx, c) - angle(a, b - dx, c)) / 2.0 / dt
    gb[1] = (angle(a, b + dy, c) - angle(a, b - dy, c)) / 2.0 / dt
    gb[2] = (angle(a, b + dz, c) - angle(a, b - dz, c)) / 2.0 / dt
    gc[0] = (angle(a, b, c + dx) - angle(a, b, c - dx)) / 2.0 / dt
    gc[1] = (angle(a, b, c + dy) - angle(a, b, c - dy)) / 2.0 / dt
    gc[2] = (angle(a, b, c + dz) - angle(a, b, c - dz)) / 2.0 / dt
    return theta, ga, gb, gc


def gradDihedral(a, b, c, d, dt=0.00001):
    """
    Calculate the a-b-c-d dihedral angle and the gradient.
    """
    dx, dy, dz = np.zeros([dt, 0.0, 0.0]), np.zeros(
        [0.0, dt, 0.0]), np.zeros([0.0, 0.0, dt])
    ga, gb, gc, gd = np.zeros(dx.shape), np.zeros(
        dx.shape), np.zeros(dx.shape), np.zeros(dx.shape)
    psi = dihedral(a, b, c, d)
    ga[0] = (dihedral(a + dx, b, c, d) - dihedral(a - dx, b, c, d)) / 2.0 / dt
    ga[1] = (dihedral(a + dy, b, c, d) - dihedral(a - dy, b, c, d)) / 2.0 / dt
    ga[2] = (dihedral(a + dz, b, c, d) - dihedral(a - dz, b, c, d)) / 2.0 / dt
    gb[0] = (dihedral(a, b + dx, c, d) - dihedral(a, b - dx, c, d)) / 2.0 / dt
    gb[1] = (dihedral(a, b + dy, c, d) - dihedral(a, b - dy, c, d)) / 2.0 / dt
    gb[2] = (dihedral(a, b + dz, c, d) - dihedral(a, b - dz, c, d)) / 2.0 / dt
    gc[0] = (dihedral(a, b, c + dx, d) - dihedral(a, b, c - dx, d)) / 2.0 / dt
    gc[1] = (dihedral(a, b, c + dy, d) - dihedral(a, b, c - dy, d)) / 2.0 / dt
    gc[2] = (dihedral(a, b, c + dz, d) - dihedral(a, b, c - dz, d)) / 2.0 / dt
    gd[0] = (dihedral(a, b, c, d + dx) - dihedral(a, b, c, d - dx)) / 2.0 / dt
    gd[1] = (dihedral(a, b, c, d + dy) - dihedral(a, b, c, d - dy)) / 2.0 / dt
    gd[2] = (dihedral(a, b, c, d + dz) - dihedral(a, b, c, d - dz)) / 2.0 / dt
    return psi, ga, gb, gc, gd


class EVBHamiltonian(object):
    """
    Topology of EVB force field.
    """

    def __init__(self, conf):
        try:
            logging.debug("CONF: %s"%str(conf))
            if not isinstance(conf, dict):
                raise BaseException("""conf object "%s" is not a dict"""%str(conf))
            self.diag = []
            self.off_diag = []
            self.V = []
            for d in conf["diag"]:
                pdbname, xmlname = d["topology"], d["parameter"]
                self.V.append(d["V"])
                ff = app.ForceField(xmlname)
                pdb = app.PDBFile(pdbname)
                system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                                         polarization='mutual', mutualInducedTargetEpsilon=0.00001, removeCMMotion=False)
                integrator = mm.LangevinIntegrator(
                    195 * unit.kelvin, 1 / unit.picosecond, 0.0005 * unit.picoseconds)
                if "platform" in conf:
                    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(conf["platform"].upper()))
                else:
                    context = mm.Context(system, integrator)
                self.diag.append(context)
            for offd in conf["off_diag"]:
                self.off_diag.append(offd)
            self.emat = np.zeros((len(self.diag), len(self.diag)))
        except Exception as e:
            logging.error(str(e))
            raise e

    def _calc_energy_from_context(self, xyz, context):
        context.setPositions(xyz)
        state = context.getState(getEnergy=True, getForces=False)
        ene = state.getPotentialEnergy()
        return ene

    def _calc_grad_from_context(self, context):
        state = context.getState(getEnergy=False, getForces=True)
        frc = state.getForces(asNumpy=True)
        return - frc

    def _calc_energy_of_off_diag(self, xyz, conf):
        # unit: kj/mol
        crd = xyz.value_in_unit(unit.angstrom)
        poly = 0.0
        for n, c in enumerate(conf["cv"]):
            if c[0] == "B":
                i, j = c[1] - 1, c[2] - 1
                poly += conf["parameter"][n] * distance(crd[i, :], crd[j, :])
            elif c[0] == "A":
                i, j, k = c[1] - 1, c[2] - 1, c[3] - 1
                poly += conf["parameter"][n] * \
                    angle(crd[i, :], crd[j, :], crd[k, :])
        return conf["A"] * np.exp(poly + conf["parameter"][-1])

    def _calc_grad_of_off_diag(self, xyz, conf):
        crd = xyz.value_in_unit(unit.angstrom)
        e = self._calc_energy_of_off_diag(xyz, conf)
        dpoly = np.zeros(crd.shape)
        for n, c in enumerate(conf["cv"]):
            if c[0] == "B":
                i, j = c[1] - 1, c[2] - 1
                _, gi, gj = gradDistance(crd[i, :], crd[j, :])
                dpoly[i, :] = dpoly[i, :] + conf["parameter"][n] * gi
                dpoly[j, :] = dpoly[j, :] + conf["parameter"][n] * gj
            elif c[0] == "A":
                i, j, k = c[1] - 1, c[2] - 1, c[3] - 1
                _, gi, gj, gk = numGradAngle(crd[i, :], crd[j, :], crd[k, :])
                dpoly[i, :] = dpoly[i, :] + conf["parameter"][n] * gi
                dpoly[j, :] = dpoly[j, :] + conf["parameter"][n] * gj
                dpoly[k, :] = dpoly[k, :] + conf["parameter"][n] * gk
        return unit.Quantity(value=e * dpoly, unit=unit.kilojoule / unit.mole / unit.angstrom)

    def _calc_energy(self, xyz):
        # return energy
        for n, i in enumerate(self.diag):
            etmp = self._calc_energy_from_context(
                xyz, i).value_in_unit(unit.kilojoule / unit.mole)
            self.emat[n, n] = etmp + self.V[n]
        for j in self.off_diag:
            res = self._calc_energy_of_off_diag(
                xyz, j)
            self.emat[j["from"] - 1, j["to"] - 1] = res
            self.emat[j["to"] - 1, j["from"] - 1] = res
        e, v = np.linalg.eig(self.emat)
        return e, v

    def calcEnergy(self, xyz):
        """
        Calculate energy.
        """
        e, v = self._calc_energy(xyz)
        return unit.Quantity(value=np.min(e), unit=unit.kilojoule / unit.mole)

    def calcEnergyGrad(self, xyz):
        """
        Calculate energy and gradient.
        """
        e, v = self._calc_energy(xyz)
        ei = np.argmin(e)

        gradient = unit.Quantity(value=np.zeros(
            xyz.shape), unit=unit.kilojoule / unit.mole / unit.angstrom)
        for n, i in enumerate(self.diag):
            gradient += v[n, ei] * v[n, ei] * \
                self._calc_grad_from_context(i)
        for j in self.off_diag:
            off_grad = self._calc_grad_of_off_diag(xyz, j)
            gradient += 2 * v[j["from"] - 1, ei] * \
                v[j["to"] - 1, ei] * off_grad
        return unit.Quantity(value=np.min(e), unit=unit.kilojoule / unit.mole), gradient

if __name__ == '__main__':
    print("MS-EVB with OPENMM")
    print("==========Test==========")
    with open(sys.argv[1], 'r') as f:
        conf = json.loads("".join(f))
    evb = EVBHamiltonian(conf)
    init = app.PDBFile(sys.argv[2])
    e, f = evb.calcEnergyGrad(init.getPositions(asNumpy=True))
    print(e)
    print(f)
