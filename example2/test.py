import multifit
import json
import numpy as np
import os

TEMPFILE = "conf.temp"
STATE_TEMPFILE = ["state_1.temp", "state_2.temp", "state_3.temp"]
VAR = np.array([1.0082351104674565,  0.7968146375146995,  0.9999951128571232,  0.7212050525825665,
                0.863906121439402,  1.5704453747427713,  0.4443723834413093,  0.568017431815148,  
                0.8224205279585167,  1.0 , 1.0,  1.0,  1.0,  1.0,  1.0,  0.6933527682819242,  
                1.4101692240472414,  1.1343124251958379,  0.968147309220588,  1.0275090180266526,
                0.873410828837772]) * np.array([-2.99996370e+01, -7.41000364e+02, -7.86000000e+02,  2.99022095e+00,
       -6.60355792e+00,  2.95636207e+00,  6.33177734e-01,  2.54112121e+00,
        2.51113116e-01,  9.14101252e+00, -5.41697473e+00,  2.17409499e-01,
       -8.81051218e+00, -7.28065122e-01, -3.18229527e+00, 0.0969, 
       239797.592, 0.10148, 216287.696, 0.152, 
       187957.832, ])

portlist = [7000]

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

state_templates = []
for fname in STATE_TEMPFILE:
    with open(fname, "r") as f:
        state_templates.append([fname.split(".")[0], Template("".join(f))])

conf = json.loads(template.render(var=VAR))

multifit.multidrawGradient(xyzs, eners, grads, VAR, template, portlist, state_templates=state_templates)
