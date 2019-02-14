import sys
from evb import *
import json
from jinja2 import Template
from simtk import unit

with open("conf.temp", "r") as f:
    temp = Template("".join(f))
conf = temp.render(var=np.zeros((15,)))

H = EVBHamiltonian(json.loads(conf.strip()))

for fname in sys.argv[1:]:
    print(fname, ":")
    with open(fname, "r") as f:
        text = [[float(j) for j in i.strip().split()] for i in f]
    xyz = unit.Quantity(np.array(text), unit.angstrom)
    H.calcEnergy(xyz)