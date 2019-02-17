from fit import *
from multifit import *
import sys
import logging
import os

HESSFILE = "freq.fchk"
TEMPFILE = "conf.temp"
STATE_TEMPFILE = []
VAR = np.array([0.0, -741.0, -786.0,
                2.9815490408547145,  -6.60189337219787,  2.964041042557548,  0.6425293989814105,
                2.5422464186777733,  0.25567283154978065,  9.141012520108227,  -5.41697473407947,
                0.21740949899967177,  -8.810512182322086,  -0.7280651221148342,  -3.1822952690936885])
portlist = [i for i in range(5000, 5012)]


def main():
    xyz, hess = getCHKHess(HESSFILE)
    mass = getCHKMass(HESSFILE)

    xyzs, eners, grads = [], [], []
    FORCEDIR = "force/"
    for fname in os.listdir(FORCEDIR):
        if fname.split(".")[-1] == "log":
            xyz_, energy_, grad_ = getGaussEnergyGradient(FORCEDIR + fname)
            xyzs.append(xyz_)
            eners.append(energy_)
            grads.append(grad_)

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
    with open("varitem.log", "r") as f:
        text = f.readlines()

    text = [i.strip().split() for i in text if "Score:" in i]
    init = [[np.array([float(j) for j in i[10:]]), float(i[8])] for i in text if float(i[8]) < 1e6]
    bound = np.array([[-1000, 1000], [-1000, 1000], [-1000, 1000], [-30, 30], [-30, 30], 
                      [-30, 30], [-30, 30], [-30, 30], [-30, 30], [-30, 30],
                      [-30, 30], [-30, 30], [-30, 30], [-30, 30], [-30, 30]])

    s, v = bayesianoptimizing(gfunc, bound, 200, init=init, kappa=1.5, gpr_sample=10000, return_traj=False)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("LOG file needed.\nExit.")
        exit()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    fh = logging.FileHandler(sys.argv[1])
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.addHandler(fh)
    main()
