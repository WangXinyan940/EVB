from fit import *
import sys
import os
import logging

QMDATA = "qm/"
HESSFILE = "hess/ts.log"
TEMPFILE = "conf.temp"
STATE_TEMPFILE = ["state_1.temp", "state_2.temp"]
VAR = np.array([ 0.00000000e+00, -1.02178714e+01,  2.96289306e-01,  1.64063455e-01,  
                 1.16490327e-02,  4.81039490e+00,  5.57221210e-01,  1.08700000e-01,
                 1.67845344e+05,  1.77920000e-01,  7.71529600e+04,  3.81845978e-02,
                 1.15710000e+02,  5.63082620e-02,  1.12320000e+02,  8.39782623e+00, 
                 13.655595693963415, 0.10900000000000001, 162364.304, 0.19563, 
                 59835.3840, 0.09128107114810141, 94.99, 0.14908052069524466, 
                 100.33, 8.397826228907983, 13.655595693963415])

BOUND = [(-5.0e+02, 5.0e+02), 
         (-5.0e+02, 5.0e+02), 
         (-1.0e+01, 1.0e+01), 
         (-1.0e+01, 1.0e+01), 
         (-1.0e+01, 1.0e+01), 
         (-5.0e+01, 5.0e+01), 
         ( 0.0e+00, 1.0e+01), 
         ( 9.0e-01, 1.1e-00), 
         ( 1.2e+05, 2.0e+05), 
         ( 1.5e-01, 2.0e-01), 
         ( 2.0e+04, 1.0e+05), 
         ( 0.0e+00, 1.0e+00), 
         ( 1.0e+02, 1.5e+02), 
         ( 5.0e-02, 6.5e-02), 
         ( 9.0e+01, 1.7e+02), 
         ( 6.0e+00, 1.5e+01), 
         ( 6.0e+00, 1.5e+01), 
         ( 9.0e-01, 1.2e-00), 
         ( 1.2e+05, 2.0e+05), 
         ( 1.5e-01, 2.0e-01),
         ( 3.0e+04, 9.0e+04), 
         ( 0.0e+00, 1.0e+00), 
         ( 7.0e+01, 1.1e+02), 
         ( 1.0e-01, 2.0e-01), 
         ( 9.0e+01, 1.2e+02), 
         ( 6.0e+00, 1.5e+01), 
         ( 6.0e+00, 1.5e+01)]


def main():
    fnames = os.listdir(QMDATA)
    xyzs, eners, grads = [], [], []
    for fname in fnames:
        txyz, tener, tgrad = getGaussianEnergyGradient(QMDATA + fname)
        xyzs.append(txyz)
        eners.append(tener)
        grads.append(tgrad)

    hxyz, hess = getGaussianHess(HESSFILE)
    mass = [12.011, 1.008, 1.008, 1.008, 35.453, 79.904]

    with open(TEMPFILE, "r") as f:
        template = Template("".join(f))

    state_templates = []
    for fname in STATE_TEMPFILE:
        with open(fname, "r") as f:
            state_templates.append([fname.split(".")[0], Template("".join(f))])

    hfunc = genHessScore(hxyz, hess, mass, template,
                         state_templates=state_templates, a_diag=1.0, a_offdiag=1.00)
    tfunc = genEnerGradScore(xyzs, eners, grads, template,
                             state_templates=state_templates)
    score = lambda v:hfunc(v) + tfunc(v)

    #drawEnergy(xyzs, eners, VAR, template, state_templates=state_templates)
    #drawGradient(xyzs, grads, VAR, template, state_templates=state_templates)
    #drawHess(hxyz, hess, mass, VAR, template, state_templates=state_templates)

    best = bayesianoptimizing(score, BOUND, niter=1000, kappa=1.5, init_value=[VAR], gpr_sample=100000000)
    print(best)

    drawEnergy(xyzs, eners, best[0], template,
               state_templates=state_templates)
    drawGradient(xyzs, grads, best[0], template,
                 state_templates=state_templates)
    drawHess(hxyz, hess, mass, best[0], template,
             state_templates=state_templates)


if __name__ == '__main__':
    main()
