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

var = np.array([-15.0,  0.96,  0.53, 1.00, 10.00, -0.00045571])

with open("conf.temp", "r") as f:
    template = Template("".join(f))

conf = json.loads(template.render(var=var))
H = evb.EVBHamiltonian(conf)

xyz, energy, grad = getGaussianEnergyGradient("qm/CH3BrCl-30.log")
xyz_no_unit = xyz.value_in_unit(unit.angstrom)

dt = 1e-6
delta = dt * np.eye(xyz_no_unit.ravel().shape[0])
calc_grad = np.zeros(grad.shape).ravel()
for n in range(calc_grad.shape[0]):
    dpos = unit.Quantity(value=(xyz_no_unit.ravel() +
                                delta[n]).reshape(-1, 3), unit=unit.angstrom)
    dneg = unit.Quantity(value=(xyz_no_unit.ravel() -
                                delta[n]).reshape(-1, 3), unit=unit.angstrom)
    anal_grad = (H.calcEnergy(dpos) - H.calcEnergy(dneg)) / \
        2. / unit.Quantity(value=dt, unit=unit.angstrom)
    calc_grad[n] = anal_grad.value_in_unit(
        unit.kilojoule / unit.mole / unit.angstrom)
print("ANALY GRAD")
e, g = H.calcEnergyGrad(xyz)
print(g.value_in_unit(unit.kilojoule / unit.mole / unit.angstrom))
print("\nNUM GRAD")
print(calc_grad.reshape(-1, 3))
print("\nVAR")
var = np.sqrt(((g.value_in_unit(unit.kilojoule / unit.mole /
                                unit.angstrom) - calc_grad.reshape(-1, 3)) ** 2).mean())
print(g.value_in_unit(unit.kilojoule / unit.mole / unit.angstrom) - calc_grad.reshape(-1, 3))
print(var)
