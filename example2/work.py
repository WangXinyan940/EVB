from fit import *
from multifit import *
import sys
import logging
import os

HESSFILE = "freq.fchk"
TEMPFILE = "conf.temp"
STATE_TEMPFILE = []
VAR = np.array([365.1560798272886,  -370.9091564905557,  23.7498531682483,  -5.578171430511826,
                6.746490111859442,  -86.06126054764621,  24.930598724683293,  -33.86685593068339,
                23.986752808835867,  48.63218346604885,  -115.93874045209743,  35.274995414925925,
                54.723615215648444,  -81.82033046918013,  46.382985481108726])
portlist = [i for i in range(5000,5017)]

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
    tfunc = lambda v: hfunc(v) + gfunc(v)
#    drawPicture(xyzs, eners, grads, VAR, template,
#                state_templates=state_templates)
#    multidrawHess(xyz, hess, mass, VAR, template, portlist, state_templates=state_templates)
    traj = basinhopping(gfunc, VAR, niter=100, T=5.0, pert=15.0)
    print(traj[0][0])
    multidrawHess(xyz, hess, mass, traj[0][0], template, portlist,
             state_templates=state_templates)


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
