import sys
from evb import *
import json
from multifit import *
from jinja2 import Template
from simtk import unit

with open("conf.temp", "r") as f:
    temp = Template("".join(f))
portlist = [9000, 9000, 9000, 9000]
xyz, hess = getCHKHess("freq.fchk")

while True:
    conf = json.loads(temp.render(var=np.random.random((15,))))
    client = EVBClient(portlist)
    client.initialize(conf)
    _ = client.calcEnergyGrad(xyz)