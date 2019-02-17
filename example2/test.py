import multifit
import json
import numpy as np
import os

TEMPFILE = "conf.temp"
VAR = np.array([-2.99996370e+01, -7.41000364e+02, -7.86000000e+02,  2.99022095e+00,
       -6.60355792e+00,  2.95636207e+00,  6.33177734e-01,  2.54112121e+00,
        2.51113116e-01,  9.14101252e+00, -5.41697473e+00,  2.17409499e-01,
       -8.81051218e+00, -7.28065122e-01, -3.18229527e+00])
portlist = [i for i in range(5000, 5012)]

from fit import *
HESSFILE = "freq.fchk"
xyz, hess = getCHKHess(HESSFILE)
mass = getCHKMass(HESSFILE)

xyzs, eners, grads = [], [], []
FORCEDIR = "force/"
for fname in os.listdir(FORCEDIR):
    if fname.split(".")[-1] == "log":
        xyz, energy, grad = getGaussEnergyGradient(FORCEDIR+fname)
        xyzs.append(xyz)
        eners.append(energy)
        grads.append(grad)

with open(TEMPFILE, "r") as f:
    template = Template("".join(f))

conf = json.loads(template.render(var=VAR))

multifit.multidrawGradient(xyzs, eners, grads, VAR, template, portlist)
