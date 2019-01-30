from fit import *
import sys
import os
import logging

QMDATA = "qm/"
HESSFILE = "hess/ts.log"
TEMPFILE = "conf.temp"
STATE_TEMPFILE = ["state_1.temp", "state_2.temp"]
VAR = np.array([-1.02178714e+01,  2.96289306e-01,  1.64063455e-01,  1.16490327e-02,
                4.81039490e+00,  5.57221210e-01,  1.07854875e-01,  3.67539345e+04,
                1.83496498e-01,  8.32166788e+04,  1.22943053e-02,  1.11986160e+02,
                1.79091272e-02,  1.05608702e+02,  5.25430179e+00,  1.77521684e+00,
                2.71316766e+00,  1.13375097e-01, -9.53444603e+03,  1.98967448e-01,
                5.73826376e+04,  1.57761737e-02,  8.59267473e+01,  3.47319534e-02,
                9.42591073e+01,  5.79449948e-03,  1.78091636e+00,  2.05894451e+00, ])


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
                         state_templates=state_templates, a_diag=0.1, a_offdiag=10.00)
    tfunc = genEnerGradScore(xyzs, eners, grads, template,
                             state_templates=state_templates)
    score = lambda x: hfunc(x) + tfunc(x)

    drawEnergy(xyzs, eners, VAR, template, state_templates=state_templates)
    drawGradient(xyzs, grads, VAR, template, state_templates=state_templates)
    drawHess(hxyz, hess, mass, VAR, template, state_templates=state_templates)
    min_result = optimize.minimize(score, VAR, jac="2-point", hess="2-point",
                                   method='L-BFGS-B', options=dict(maxiter=1000, disp=True, gtol=0.01))
    print(min_result.x)
    drawEnergy(xyzs, eners, min_result.x, template,
               state_templates=state_templates)
    drawGradient(xyzs, grads, min_result.x, template,
                 state_templates=state_templates)
    drawHess(hxyz, hess, mass, min_result.x, template,
             state_templates=state_templates)


if __name__ == '__main__':
    main()
