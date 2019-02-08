from fit import *
from multifit import *
import sys
import logging

HESSFILE = "freq.fchk"
TEMPFILE = "conf.temp"
STATE_TEMPFILE = []
VAR = np.array([1.00, 1.00, 1.00, 0.00, 0.00,
                0.00, 0.00, 0.00, 0.00, 0.00, 
                0.00, 0.00, 0.00, 0.00, 0.00])
portlist = [5000, 5001, 5002, 5003]

def main():
    xyz, hess = getCHKHess(HESSFILE)
    mass = getCHKMass(HESSFILE)

    with open(TEMPFILE, "r") as f:
        template = Template("".join(f))

    state_templates = []
    for fname in STATE_TEMPFILE:
        with open(fname, "r") as f:
            state_templates.append([fname.split(".")[0], Template("".join(f))])

    tfunc = multigenHessScore(xyz, hess, mass, template, portlist,
                         state_templates=state_templates, a_diag=0.1, a_offdiag=1000.00)
#    drawPicture(xyzs, eners, grads, VAR, template,
#                state_templates=state_templates)
    multidrawHess(xyz, hess, mass, VAR, template, portlist, state_templates=state_templates)
    #traj = basinhopping(tfunc, VAR, niter=50, T=2.0, pert=2.5)
    min_result = optimize.minimize(tfunc, VAR, jac="2-point", hess="2-point",
                                   method='L-BFGS-B', options=dict(maxiter=1000, disp=True, gtol=0.01))
    print(min_result.x)
    multidrawHess(xyz, hess, mass, min_result.x, template, portlist,
             state_templates=state_templates)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("LOG file needed.\nExit.")
        exit()
    logging.basicConfig(filename=sys.argv[1], level=logging.INFO)
    main()
