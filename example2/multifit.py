from gevent import monkey
monkey.patch_all()
import gevent
from gevent import pool
import json
import logging
import socket
import numpy as np
import simtk.unit as unit
from jinja2 import Template
import matplotlib.pyplot as plt
import evb
from gevent.lock import BoundedSemaphore
sem = BoundedSemaphore(200)


class ParamFail(BaseException):
    pass


class InitFail(BaseException):
    pass


class EVBServer(object):

    def __init__(self, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._port = port
        self.s.bind(("127.0.0.1", port))
        self.H = None

    def listen(self, max_link=50000):
        logging.info("Listening port %i" % self._port)
        self.s.listen(max_link)
        while True:
            sock, addr = self.s.accept()
            logging.info("Server: %i  Client: %s:%i" %
                         (self._port, addr[0], addr[1]))
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
                    sock.send(("%18.10f" % ener).encode("utf-8"))
                    logging.info("Finish.")
                elif data[:4] == "GRAD":
                    logging.info("Calculate gradient.")
                    xyz = np.array([float(i) for i in data[4:].split()])
                    xyz = xyz.reshape((-1, 3))
                    ener, grad = self._gradient(xyz)
                    grad = " ".join("%18.10f" % i for i in grad.ravel())
                    sock.send(("%18.10f " % ener + grad).encode("utf-8"))
                    logging.info("Finish.")
                else:
                    logging.warn("Unknown message. Did nothing.")
                    sock.send("ERROR".encode("utf-8"))
                sock.close()
            except Exception as e:
                logging.error("COLLAPSE: " + str(type(e)) + str(e))
                sock.send("ERROR".encode("utf-8"))
                sock.close()

    def _initialize(self, conf):
        for _ in range(4):
            try:
                self.H = evb.EVBHamiltonian(conf)
                return
            except ValueError as e:
                logging.error("INIT FAIL: " + str(type(e)) + str(e) + " RETRY: %i"%(_ + 1))
                continue
            except FileNotFoundError as e:
                logging.error("INIT FAIL: " + str(type(e)) + str(e) + " RETRY: %i"%(_ + 1))
                continue
            except Exception as e:
                logging.error("INIT FAIL: " + str(type(e)) + str(e))
                raise e
        try:
            self.H = evb.EVBHamiltonian(conf)
        except Exception as e:
            logging.error("INIT FAIL: " + str(type(e)) + str(e))
            raise e

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
        self.state = 0  # 0: haven't init, 1: init success, 2: init fail

    def initialize(self, conf):
        def init(pt, n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", pt))
            data = "INIT" + json.dumps(conf)
            s.send(data.encode("utf-8"))
            ret = s.recv(1024).decode("utf-8")
            if not ret == "FINISH":
                logging.error("INIT FAIL!!!")
                s.close()
                return 1
            s.close()
            return 0

        res = gevent.joinall([gevent.spawn(init, pt, n)
                        for n, pt in enumerate(self.port_list)])
        if sum([i.value for i in res]) > 1e-5:
            raise InitFail("Init fail. Job stop.")

    def calcEnergy(self, xyz):
        try:
            xyz_no_unit = xyz.value_in_unit(unit.angstrom)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            pt = np.random.choice(self.port_list)
            s.connect(("127.0.0.1", pt))
            data = "ENER" + " ".join("%18.10f" % i for i in xyz_no_unit.ravel())
            s.send(data.encode("utf-8"))
            ret = s.recv(1024).decode("utf-8")
            if ret == "ERROR":
                ans = 1
                s.close()
                raise ParamFail("Value NaNs")
            else:
                ans = 0
            s.close()
            return ans, unit.Quantity(float(ret), unit.kilojoule_per_mole)
        except ParamFail as e:
            raise e
        except BaseException as e:
            logging.debug("Client error " + str(type(e)) + str(e))
            s.close()
            raise e

    def calcEnergyGrad(self, xyz):
        xyz_no_unit = xyz.value_in_unit(unit.angstrom)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        pt = np.random.choice(self.port_list)
        s.connect(("127.0.0.1", pt))
        try:
            data = "GRAD" + " ".join("%18.10f" %
                                     i for i in xyz_no_unit.ravel())
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
                s.close()
                raise ParamFail("Value NaNs")
            else:
                ans = 0
            s.close()
            ret = np.array([float(i) for i in ret.strip().split()])
            return ans, unit.Quantity(ret[0], unit.kilojoule_per_mole), unit.Quantity(ret[1:].reshape((-1, 3)), unit.kilojoule_per_mole / unit.angstrom)
        except ParamFail as e:
            raise e
        except BaseException as e:
            logging.debug("Client error " + str(type(e)) + str(e))
            s.close()
            raise e


def multigenHessScore(xyz, hess, mass, template, portlist, state_templates=[], dx=0.00001, a_diag=1.00, a_offdiag=1.00):
    """
    Generate score func.
    """
    # mat decomp
    mass_mat = np.diag(1. / np.sqrt(mass.value_in_unit(unit.amu)))
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
            client = EVBClient(portlist)
            client.initialize(conf)
            # calc hess (unit in kJ / mol / A^2)
            oxyz = xyz.value_in_unit(unit.angstrom).ravel()
            dxyz = np.eye(oxyz.shape[0])
            calc_hess = np.zeros(dxyz.shape)

            def func(gi, hmat):
                sem.acquire()
                try:
                    txyz = unit.Quantity(
                        value=(oxyz + dxyz[:, gi] * dx).reshape((-1, 3)), unit=unit.angstrom)
                    _, tep, tgp = client.calcEnergyGrad(txyz)
                    if _ == 0:
                        tgp = tgp.value_in_unit(
                            unit.kilocalorie_per_mole / unit.angstrom).ravel()
                # parallel
                    txyz = unit.Quantity(
                        value=(oxyz - dxyz[:, gi] * dx).reshape((-1, 3)), unit=unit.angstrom)
                    _, ten, tgn = client.calcEnergyGrad(txyz)
                    if _ == 0:
                        tgn = tgn.value_in_unit(
                            unit.kilocalorie_per_mole / unit.angstrom).ravel()
                    hmat[:, gi] = (tgp - tgn) / 2.0 / dx
                    sem.release()
                    return 0
                except ParamFail as e:
                    sem.release()
                    return 1

            # for gi in range(dxyz.shape[0]):
            #    func(gi, calc_hess)

            # for gi in range(dxyz.shape[0]):
            #    pl.spawn(func(gi))
            # pl.join()

            ret = gevent.joinall([gevent.spawn(func, gi, calc_hess)
                                  for gi in range(dxyz.shape[0])])
            if sum([i.value for i in ret]) > 1e-5:
                raise ParamFail("Emmmmmm")
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
            s_qm = np.sort(np.abs(vib_qm))[6:]
            s_mm = np.sort(np.abs(vib_mm))[6:]
            var_diag = ((s_qm - s_mm) ** 2).sum() / s_mm.shape[0]
            var_offdiag = (var - np.diag(np.diag(var))).sum() / \
                (var.shape[0] ** 2 - var.shape[0])
            return a_diag * var_diag + a_offdiag * var_offdiag
        except ParamFail as e:
            logging.debug(str(e))
            return 1e12
    return valid


def multigenEnerGradScore(xyzs, eners, grads, template, portlist, state_templates=[], a_ener=1.00, a_grad=1.00):
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
            client = EVBClient(portlist)
            client.initialize(conf)
            # calc forces
            calc_ener = np.zeros((len(xyzs),))
            calc_grad = np.zeros((len(xyzs), xyzs[0].ravel().shape[0]))

            def func(n, e_array, g_array):
                sem.acquire()
                try:
                    _, energy, gradient = client.calcEnergyGrad(xyzs[n])
                    if _ == 0:
                        e_array[n] = energy.value_in_unit(
                            unit.kilojoule / unit.mole)
                        g_array[n, :] = gradient.value_in_unit(
                            unit.kilojoule_per_mole / unit.angstrom).ravel()
                    sem.release()
                    return 0
                except BaseException as e:
                    if not isinstance(e, ParamFail):
                        logging.debug("Parallel: " + str(type(e)) + str(e))
                    sem.release()
                    return 1

            ret = gevent.joinall([gevent.spawn(func, n, calc_ener, calc_grad)
                                  for n in range(len(xyzs))])
            if sum([1 if ((i.value == 1) or (i.value is None)) else 0 for i in ret]) > 1e-5:
                raise ParamFail("input parameter is unphysical")
            # compare
            ref_ener = np.array(
                [i.value_in_unit(unit.kilojoule / unit.mole) for i in eners])
            var_ener = np.sqrt(
                (np.abs((calc_ener - calc_ener.max()) - (ref_ener - ref_ener.max())) ** 2).sum())

            calc_grad = calc_grad.ravel()
            ref_grad = np.array([i.value_in_unit(
                unit.kilojoule_per_mole / unit.angstrom).ravel() for i in grads]).ravel()

            var_grad = np.sqrt(((calc_grad - ref_grad) ** 2).mean())
            return a_grad * var_grad + a_ener * var_ener
        except ParamFail as e:
            logging.debug("REPORT: " + str(e))
            return 1e12
    return valid

def multigenEnerScore(xyzs, eners, template, portlist, state_templates=[], a_ener=1.00, a_grad=1.00):
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
            client = EVBClient(portlist)
            client.initialize(conf)
            # calc forces
            calc_ener = np.zeros((len(xyzs),))

            def func(n, e_array):
                sem.acquire()
                try:
                    _, energy = client.calcEnergy(xyzs[n])
                    if _ == 0:
                        e_array[n] = energy.value_in_unit(
                            unit.kilojoule / unit.mole)
                    sem.release()
                    return 0
                except BaseException as e:
                    if not isinstance(e, ParamFail):
                        logging.debug("Parallel: " + str(type(e)) + str(e))
                    sem.release()
                    return 1

            ret = gevent.joinall([gevent.spawn(func, n, calc_ener)
                                  for n in range(len(xyzs))])
            if sum([1 if ((i.value == 1) or (i.value is None)) else 0 for i in ret]) > 1e-5:
                raise ParamFail("input parameter is unphysical")
            # compare
            ref_ener = np.array(
                [i.value_in_unit(unit.kilojoule / unit.mole) for i in eners])
            var_ener = np.sqrt(
                (np.abs((calc_ener - calc_ener.max()) - (ref_ener - ref_ener.max())) ** 2).sum())
            return var_ener
        except ParamFail as e:
            logging.debug("REPORT: " + str(e))
            return 1e12
    return valid


def multidrawGradient(xyzs, eners, grads, var, template, portlist, state_templates=[]):

    for name, temp in state_templates:
        with open("%s/%s.xml" % (TEMPDIR.name, name), "w") as f:
            f.write(temp.render(var=np.abs(var)))
    # gen config file
    conf = json.loads(template.render(var=var))
    for n, fn in enumerate(state_templates):
        conf["diag"][n]["parameter"] = "%s/%s.xml" % (TEMPDIR.name, fn[0])
    # gen halmitonian
    client = EVBClient(portlist)
    client.initialize(conf)
    # calc forces
    calc_ener = np.zeros((len(xyzs),))
    calc_grad = np.zeros((len(xyzs), xyzs[0].ravel().shape[0]))

    def func(n, e_array, g_array):
        sem.acquire()
        try:
            _, energy, gradient = client.calcEnergyGrad(xyzs[n])
            if _ == 0:
                e_array[n] = energy.value_in_unit(
                    unit.kilojoule / unit.mole)
                g_array[n, :] = gradient.value_in_unit(
                    unit.kilojoule_per_mole / unit.angstrom).ravel()
            sem.release()
            return 0
        except BaseException as e:
            if not isinstance(e, ParamFail):
                logging.debug("Parallel: " + str(type(e)) + str(e))
            sem.release()
            return 1


    gevent.joinall([gevent.spawn(func, n, calc_ener, calc_grad)
                    for n in range(len(xyzs))])
    # compare
    calc_ener = calc_ener - calc_ener.max()
    ref_ener = np.array([i.value_in_unit(unit.kilojoule_per_mole) for i in eners])
    ref_ener = ref_ener - ref_ener.max()
    plt.scatter(calc_ener, ref_ener)
    plt.plot([calc_ener.min(), calc_ener.max()],
             [calc_ener.min(), calc_ener.max()])
    plt.xlabel("CALC ENERGY")
    plt.ylabel("REF ENERGY")
    plt.show()
    calc_grad = calc_grad.ravel()
    ref_grad = np.array([i.value_in_unit(
        unit.kilojoule_per_mole / unit.angstrom).ravel() for i in grads]).ravel()

    plt.plot([calc_grad.min(), calc_grad.max()],
             [calc_grad.min(), calc_grad.max()])
    plt.scatter(calc_grad.ravel(), ref_grad.ravel())
    plt.xlabel("CALC GRADIENT")
    plt.ylabel("REF GRADIENT")
    plt.show()


def multidrawHess(xyz, hess, mass, var, template, portlist, state_templates=[], dx=0.00001):
    mass_mat = np.diag(1. / np.sqrt(mass.value_in_unit(unit.amu)))
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
    client = EVBClient(portlist)
    client.initialize(conf)
    # calc hess (unit in kJ / mol / A^2)
    oxyz = xyz.value_in_unit(unit.angstrom).ravel()
    dxyz = np.eye(oxyz.shape[0])
    calc_hess = np.zeros(dxyz.shape)

    def func(gi, hmat):
        sem.acquire()
        txyz = unit.Quantity(
            value=(oxyz + dxyz[:, gi] * dx).reshape((-1, 3)), unit=unit.angstrom)
        _, tep, tgp = client.calcEnergyGrad(txyz)
        tgp = tgp.value_in_unit(
            unit.kilocalorie_per_mole / unit.angstrom).ravel()
    # parallel
        txyz = unit.Quantity(
            value=(oxyz - dxyz[:, gi] * dx).reshape((-1, 3)), unit=unit.angstrom)
        _, ten, tgn = client.calcEnergyGrad(txyz)
        tgn = tgn.value_in_unit(
            unit.kilocalorie_per_mole / unit.angstrom).ravel()
        hmat[:, gi] = (tgp - tgn) / 2.0 / dx
        sem.release()
    # for gi in range(dxyz.shape[0]):
    #    func(gi, calc_hess)

    # for gi in range(dxyz.shape[0]):
    #    pl.spawn(func(gi))
    # pl.join()

    gevent.joinall([gevent.spawn(func, gi, calc_hess)
                    for gi in range(dxyz.shape[0])])

    calc_theta = np.dot(mass_mat, np.dot(calc_hess, mass_mat))
    # change basis
    calc_theta_p = np.dot(qvI, np.dot(calc_theta, qv))
    var = (calc_theta_p - theta_p) ** 2
    f = plt.imshow(var)
    plt.colorbar(f)
    plt.show()
    vib_qm, vib_mm = np.diag(theta_p), np.diag(calc_theta_p)
    vib_qm = unit.Quantity(
        vib_qm, unit.kilocalorie_per_mole / unit.angstrom ** 2 / unit.amu)
    vib_mm = unit.Quantity(
        vib_mm, unit.kilocalorie_per_mole / unit.angstrom ** 2 / unit.amu)
    vib_qm = vib_qm.value_in_unit(unit.joule / unit.meter ** 2 / unit.kilogram)
    vib_mm = vib_mm.value_in_unit(unit.joule / unit.meter ** 2 / unit.kilogram)
    vib_qm = np.sqrt(np.abs(vib_qm)) / 2. / np.pi / \
        2.99792458e10 * np.sign(vib_qm)
    vib_mm = np.sqrt(np.abs(vib_mm)) / 2. / np.pi / \
        2.99792458e10 * np.sign(vib_mm)
    plt.scatter(vib_qm, vib_mm)
    vmin = min([vib_qm.min(), vib_mm.min()])
    vmax = max([vib_qm.max(), vib_mm.max()])
    plt.plot([vmin, vmax], [vmin, vmax], c="black", ls="--")
    plt.xlabel("QM Freq")
    plt.ylabel("FF Freq")
    plt.show()
