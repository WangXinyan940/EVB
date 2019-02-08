import multifit
import json
import numpy as np
import os

TEMPFILE = "conf.temp"
VAR = np.array([ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00])


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

score = multifit.multigenEnerGradScore(xyzs, eners, grads, template, [5000, 5001, 5002, 5003])
s = score(VAR)

score2 = genEnerGradScore(xyzs, eners, grads, template)
s2 = score2(VAR)
print(s, s2)