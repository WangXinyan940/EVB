from fit import *
from multifit import *
import sys
import logging
import os

HESSFILE = "freq.fchk"
TEMPFILE = "conf.temp"
STATE_TEMPFILE = ["state_1.temp", "state_2.temp", "state_3.temp"]
VAR = np.array([-2.99996370e+01, -7.41000364e+02, -7.86000000e+02,  2.99022095e+00,
       -6.60355792e+00,  2.95636207e+00,  6.33177734e-01,  2.54112121e+00,
        2.51113116e-01,  9.14101252e+00, -5.41697473e+00,  2.17409499e-01,
       -8.81051218e+00, -7.28065122e-01, -3.18229527e+00, 0.0969, 
       239797.592, 0.10148, 216287.696, 0.152, 
       187957.832, ])
portlist = [i for i in range(5000,5012)]

def main():
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

    hfunc = multigenHessScore(xyz, hess, mass, template, portlist,
                         state_templates=state_templates, a_diag=1.0, a_offdiag=50.00)
    gfunc = multigenEnerGradScore(xyzs, eners, grads, template, portlist)
    efunc = multigenEnerScore(xyzs, eners, template, portlist)
    tfunc = lambda v: hfunc(v) + gfunc(v)
#    drawPicture(xyzs, eners, grads, VAR, template,
#                state_templates=state_templates)
#    multidrawHess(xyz, hess, mass, VAR, template, portlist, state_templates=state_templates)
    #min_result = optimize.minimize(efunc, VAR, jac="2-point", method="L-BFGS-B", options=dict(maxiter=200, disp=True, gtol=0.01, maxls=10))
    #print(min_result)
    pfunc = lambda x:efunc(x * VAR)
    traj = basinhopping(pfunc, np.zeros(VAR.shape) + 1, niter=100, T=5.0, pert=0.15)
    print(traj[0][0])
    multidrawGradient(xyzs, eners, grads, traj[0][0] * VAR, template, portlist)
    #multidrawHess(xyz, hess, mass, traj[0][0], template, portlist,
    #         state_templates=state_templates)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("LOG file needed.\nExit.")
        exit()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    fh = logging.FileHandler(sys.argv[1])
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    main()
