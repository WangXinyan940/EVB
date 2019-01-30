from fit import *
import sys, os
import logging

QMDATA = "qm/"
TEMPFILE = "conf.temp"
STATE_TEMPFILE = ["state_1.temp", "state_2.temp"]
VAR = np.array([-1.02178714e+01,  2.96289306e-01,  1.64063455e-01,  1.16490327e-02,
                 4.81039490e+00,  5.57221210e-01,  1.07854875e-01,  3.67539345e+04,
                 1.83496498e-01,  8.32166788e+04,  1.22943053e-02,  1.11986160e+02,
                 1.79091272e-02,  1.05608702e+02,  5.25430179e+00,  1.77521684e+00,
                 2.71316766e+00,  1.13375097e-01, -9.53444603e+03,  1.98967448e-01,
                 5.73826376e+04,  1.57761737e-02,  8.59267473e+01,  3.47319534e-02,
                 9.42591073e+01,  5.79449948e-03,  1.78091636e+00,  2.05894451e+00,])


def main():
    fnames = os.listdir(QMDATA)
    xyzs, eners, grads = [], [], []
    for fname in fnames:
        txyz, tener, tgrad = getGaussianEnergyGradient(QMDATA + fname)
        xyzs.append(txyz)
        eners.append(tener)
        grads.append(tgrad)

    with open(TEMPFILE, "r") as f:
        template = Template("".join(f))

    state_templates = []
    for fname in STATE_TEMPFILE:
        with open(fname, "r") as f:
            state_templates.append([fname.split(".")[0], Template("".join(f))])

    tfunc = genEnerGradScore(xyzs, eners, grads, template, state_templates=state_templates)

    drawEnergy(xyzs, eners, VAR, template, state_templates=state_templates)
    drawGradient(xyzs, grads, VAR, template, state_templates=state_templates)
    traj = basinhopping(tfunc, VAR, niter=50, T=2.0, pert=2.5)
    drawEnergy(xyzs, eners, traj[0][1], template, state_templates=state_templates)
    drawGradient(xyzs, grads, traj[0][1], template, state_templates=state_templates)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("LOG file needed.\nExit.")
        exit()
    logging.basicConfig(filename=sys.argv[1], level=logging.INFO)
    main()