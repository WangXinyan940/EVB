import sys
from evb import *
import json
from multifit import *
from jinja2 import Template
from simtk import unit

with open("conf.temp", "r") as f:
    temp = Template("".join(f))
conf = json.loads(temp.render(var=np.zeros((15,))))
portlist = [5000, 5000, 5000, 5000]

xyz, hess = getCHKHess("freq.fchk")

while True:
    client = EVBClient(portlist)
    client.initialize(conf)
    _ = client.calcEnergyGrad(xyz)