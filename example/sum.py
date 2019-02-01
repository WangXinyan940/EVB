from fit import *
import sys
import os
import logging

QMDATA = "qm/"
HESSFILE = "hess/ts.log"
TEMPFILE = "conf.temp"
STATE_TEMPFILE = ["state_1.temp", "state_2.temp"]
VAR = np.array([ 9.01746177e-00, -1.02177314e+01,  3.27273398e-01,  1.87726494e-01,
                -7.57667288e-04,  4.82145772e+00,  5.76181083e-01,  1.06418266e-01,
                 1.67845344e+05,  1.83136518e-01,  7.71529600e+04, -2.37615102e-10,
                 1.15709687e+02,  2.45093702e-02,  1.12320315e+02,  8.39795056e+00,
                 1.36556712e+01,  1.06408057e-01,  1.62364304e+05,  1.97054913e-01,
                 5.98353844e+04,  4.15599127e-02,  9.49902495e+01,  8.24973847e-02,
                 1.00329125e+02,  8.39794694e+00,  1.36556918e+01,  5.00000000e+00,
                 5.00000000e+00])


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
    score = lambda x: 0.6 * hfunc(x) + tfunc(x)

    drawEnergy(xyzs, eners, VAR, template, state_templates=state_templates)
    drawGradient(xyzs, grads, VAR, template, state_templates=state_templates)
    drawHess(hxyz, hess, mass, VAR, template, state_templates=state_templates)
#    min_result = optimize.minimize(score, VAR, jac="2-point", hess="2-point",
#                                   method='L-BFGS-B', options=dict(maxiter=1000, disp=True, gtol=0.01))
#    print(min_result.x)
#    drawEnergy(xyzs, eners, min_result.x, template,
#               state_templates=state_templates)
#    drawGradient(xyzs, grads, min_result.x, template,
#                 state_templates=state_templates)
#    drawHess(hxyz, hess, mass, min_result.x, template,
#             state_templates=state_templates)
    traj = basinhopping(score, VAR, niter=50, bounds=None, T=1.0, pert=25.0, inner_iter=1000)
    print(traj[0][1])
    drawEnergy(xyzs, eners, traj[0][1], template,
               state_templates=state_templates)
    drawGradient(xyzs, grads, traj[0][1], template,
                 state_templates=state_templates)
    drawHess(hxyz, hess, mass, traj[0][1], template,
             state_templates=state_templates)


if __name__ == '__main__':
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(sys.argv[1])
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    main()
