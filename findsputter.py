import matplotlib.pyplot as pl
import numpy as np
import everest
import george
from george import kernels


def SputterGP(fpix,ferr):

    t = np.linspace(0,len(fpix),len(fpix))
    flat_ferr = np.sum(ferr.reshape((len(ferr)), -1), axis=1)

    flux = np.sum(fpix.reshape((len(fpix)), -1), axis=1)
    fnorm = flux - np.mean(flux)
    kernel = np.var(fnorm) * kernels.ExpSquaredKernel(0.5)
    gp = george.GP(kernel)

    gp.compute(t, flat_ferr)
    pred, pred_var = gp.predict(fnorm, t, return_var=True)

    return pred, pred_var
