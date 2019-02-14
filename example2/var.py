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
portlist = [i for i in range(5000, 5017)]


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
    var_list = []
    for i in range(2 ** VAR.shape[0] - 1):
        var_list.append("{:0>15}".format(bin(i)[2:]))

    def f(v, l):
        ret = np.zeros(v.shape)
        for n, i in enumerate(l):
            if i == "0":
                ret[n] = - v[n]
            else:
                ret[n] = v[n]
    var_list = [f(VAR, i) for i in var_list]
    result = []
    for v in var_list:
        min_result = optimize.minimize(gfunc, newvar, jac="2-point", method="L-BFGS-B",
                                       options=dict(maxiter=200, disp=True, gtol=0.1, maxls=10))
        logging.info("Score: %.6f" % min_result.fun + " Result:  " +
                     "  ".join("{}".format(_) for _ in min_result.x))
        result.append([min_result.fun, min_result.x])
    sort_result = sorted(result, key=lambda x: x[0])
    logging.info("Minimum score: %.6f" % sort_result[0][0] + " Result:  " +
                 "  ".join("{}".format(_) for _ in sort_result[0][1]))
    multidrawHess(xyz, hess, mass, sort_result[0][0], template, portlist,
                  state_templates=state_templates)


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
    logger.addHandler(ch)
    logger.addHandler(fh)
    main()
