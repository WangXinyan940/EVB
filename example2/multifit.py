import gevent
from gevent import monkey
monkey.patch_socket()
import json
import logging
import socket
import numpy as np
import simtk.unit as unit
from jinja2 import Template
import evb
from gevent.lock import BoundedSemaphore
sem = BoundedSemaphore(50)


class EVBServer(object):

    def __init__(self, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._port = port
        self.s.bind(("127.0.0.1", port))
        self.H = None

    def listen(self, max_link=500):
        logging.info("Listening port %i" % self._port)
        self.s.listen(max_link)
        while True:
            sock, addr = self.s.accept()
            logging.info("Server:", self._port, "  Client:", addr)
            try:
                buff = []
                while True:
                    d = sock.recv(1024)
                    buff.append(d)
                    if len(d) < 1024:
                        break
                data = b"".join(buff).decode("utf-8")
                if data[:4] == "INIT":
                    logging.info("Initializing.")
                    conf = json.loads(data[4:])
                    self._initialize(conf)
                    sock.send("FINISH".encode("utf-8"))

                elif self.H is None:
                    logging.warn("Doing calculation before initialized.")
                    sock.send("ERROR".encode("utf-8"))
                    sock.close()
                    continue
                elif data[:4] == "ENER":
                    logging.info("Calculate energy.")
                    xyz = np.array([float(i) for i in data[4:].split()])
                    xyz = xyz.reshape((-1, 3))
                    ener = self._energy(xyz)
                    sock.send(("%16.8f" % ener).encode("utf-8"))
                elif data[:4] == "GRAD":
                    logging.info("Calculate gradient.")
                    xyz = np.array([float(i) for i in data[4:].split()])
                    xyz = xyz.reshape((-1, 3))
                    ener, grad = self._gradient(xyz)
                    grad = " ".join("%16.8f" % i for i in grad.ravel())
                    sock.send(("%16.8f " % ener + grad).encode("utf-8"))
                else:
                    logging.warn("Unknown message. Did nothing.")
                    sock.send("ERROR".encode("utf-8"))
            except Exception as e:
                logging.error("COLLAPSE:", str(e))
            finally:
                sock.close()

    def _initialize(self, conf):
        if self.H:
            del self.H
        self.H = evb.EVBHamiltonian(conf)

    def _energy(self, xyz):
        xyz = unit.Quantity(xyz, unit.angstrom)
        energy = self.H.calcEnergy(xyz)
        return energy.value_in_unit(unit.kilojoule_per_mole)

    def _gradient(self, xyz):
        xyz = unit.Quantity(xyz, unit.angstrom)
        energy, grad = self.H.calcEnergyGrad(xyz)
        return energy.value_in_unit(unit.kilojoule_per_mole), grad.value_in_unit(unit.kilojoule_per_mole / unit.angstrom)


class EVBClient(object):
    """
    The client of EVBHalmitonian.
    Have similar interfaces with evb.EVBHalmitonian.
    Support coroutine.
    """

    def __init__(self, port_list=[]):
        self.port_list = port_list
        self.pi = 0

    def initialize(self, conf):
        for pt in self.port_list:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", pt))
            data = "INIT" + json.dumps(conf)
            s.send(data.encode("utf-8"))
            ret = s.recv(1024).decode("utf-8")
            if ret == "FINISH":
                ans = 0
            else:
                ans = 1
                break
            s.close()
        return ans

    def calcEnergy(self, xyz):
        xyz_no_unit = xyz.value_in_unit(unit.angstrom)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        pt = self.port_list[self.pi % len(self.port_list)]
        self.pi += 1
        print(pt)
        s.connect(("127.0.0.1", pt))
        data = "ENER" + " ".join("%16.8f" % i for i in xyz_no_unit.ravel())
        s.send(data.encode("utf-8"))
        ret = s.recv(1024).decode("utf-8")
        if ret == "ERROR":
            ans = 1
        else:
            ans = 0
        s.close()
        return ans, unit.Quantity(float(ret), unit.kilojoule_per_mole)

    def calcEnergyGrad(self, xyz):
        xyz_no_unit = xyz.value_in_unit(unit.angstrom)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", np.random.choice(self.port_list)))
        data = "GRAD" + " ".join("%16.8f" % i for i in xyz_no_unit.ravel())
        s.send(data.encode("utf-8"))
        buff = []
        while True:
            d = s.recv(1024)
            buff.append(d)
            if len(d) < 1024:
                break
        ret = b"".join(buff).decode("utf-8")
        if ret == "ERROR":
            ans = 1
        else:
            ans = 0
        s.close()
        ret = np.array([float(i) for i in ret.strip().split()])
        return ans, unit.Quantity(ret[0], unit.kilojoule_per_mole), unit.Quantity(ret[1:].reshape((-1, 3)), unit.kilojoule_per_mole / unit.angstrom)


def multigenHessScore(xyz, hess, mass, template, portlist, state_templates=[], dx=0.00001, a_diag=1.00, a_offdiag=1.00):
    """
    Generate score func.
    """
    # mat decomp
    mass_mat = np.diag(1. / np.sqrt(mass))
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
            conf["diag"][n]["parameter"] = "%s/%s.xml" % (TEMPDIR.name, fn[0])
        # gen halmitonian
        client = EVBClient(portlist)
        ret = client.initialize(conf)
        # calc hess (unit in kJ / mol / A^2)
        oxyz = xyz.value_in_unit(unit.angstrom).ravel()
        dxyz = np.eye(oxyz.shape[0])
        calc_hess = np.zeros(dxyz.shape)

        def func(gi):
            txyz = unit.Quantity(
                value=(oxyz + dxyz[:, gi] * dx).reshape((-1, 3)), unit=unit.angstrom)
            _, tep, tgp = client.calcEnergyGrad(txyz)
            calc_hess[:, gi] = calc_hess[
                :, gi] + tgp.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom).ravel()
        # parallel
            txyz = unit.Quantity(
                value=(oxyz - dxyz[:, gi] * dx).reshape((-1, 3)), unit=unit.angstrom)
            _, ten, tgn = client.calcEnergyGrad(txyz)
            calc_hess[:, gi] = calc_hess[
                :, gi] - tgn.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom).ravel()
        gevent.joinall([gevent.spawn(func, gi) for gi in range(dxyz.shape[0])])

        calc_hess = calc_hess / 2.0 / dx
        calc_theta = np.dot(mass_mat, np.dot(calc_hess, mass_mat))
        # change basis
        calc_theta_p = np.dot(qvI, np.dot(calc_theta, qv))

        vib_qm, vib_mm = np.diag(theta_p), np.diag(calc_theta_p)
        vib_qm = unit.Quantity(
            vib_qm, unit.kilocalorie_per_mole / unit.angstrom ** 2 / unit.amu)
        vib_mm = unit.Quantity(
            vib_mm, unit.kilocalorie_per_mole / unit.angstrom ** 2 / unit.amu)
        vib_qm = vib_qm.value_in_unit(
            unit.joule / unit.meter ** 2 / unit.kilogram)
        vib_mm = vib_mm.value_in_unit(
            unit.joule / unit.meter ** 2 / unit.kilogram)
        vib_qm = np.sqrt(np.abs(vib_qm)) / 2. / np.pi / \
            2.99792458e10 * np.sign(vib_qm)
        vib_mm = np.sqrt(np.abs(vib_mm)) / 2. / np.pi / \
            2.99792458e10 * np.sign(vib_mm)

        var = (calc_theta_p - theta_p) ** 2
        var_diag = (((vib_qm - vib_mm) / vib_qm) ** 2).sum() / vib_mm.shape[0]
        var_offdiag = (var - np.diag(np.diag(var))).sum() / \
            (var.shape[0] ** 2 - var.shape[0])
        return a_diag * var_diag + a_offdiag * var_offdiag

    return valid
