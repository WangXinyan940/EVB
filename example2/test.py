import multifit
import json
import numpy as np

TEMPFILE = "conf.temp"
VAR = np.array([ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00,  0.00000000e+00])


from fit import *
HESSFILE = "freq.fchk"
xyz, hess = getCHKHess(HESSFILE)
mass = getCHKMass(HESSFILE)

with open(TEMPFILE, "r") as f:
    template = Template("".join(f))

conf = json.loads(template.render(var=VAR))

score = multifit.multigenHessScore(xyz, hess, mass, template, [5000, 5001, 5002, 5003])
s, q = score(VAR)

score2 = genHessScore(xyz, hess, mass, template)
s2, q2 = score2(VAR)

print(s, s2)

f = plt.scatter(q,q2)
plt.show()