# Author: Travis Oliphant
from __future__ import division, print_function, absolute_import

__all__ = ['odeint']

from . import _odepack
from copy import copy
import warnings

class ODEintWarning(Warning):
    pass

_msgs = {2: "Integration successful.",
         1: "Nothing was done; the integration time was 0.",
         -1: "Excess work done on this call (perhaps wrong Dfun type).",
         -2: "Excess accuracy requested (tolerances too small).",
         -3: "Illegal input detected (internal error).",
         -4: "Repeated error test failures (internal error).",
         -5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
         -6: "Error weight became zero during problem.",
         -7: "Internal workspace insufficient to finish (internal error)."
         }


def odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=0,
           ml=None, mu=None, rtol=None, atol=None, tcrit=None, h0=0.0,
           hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12,
           mxords=5, printmessg=0):


    if ml is None:
        ml = -1  # changed to zero inside function call
    if mu is None:
        mu = -1  # changed to zero inside function call
    t = copy(t)
    y0 = copy(y0)
    output = _odepack.odeint(func, y0, t, args, Dfun, col_deriv, ml, mu,
                             full_output, rtol, atol, tcrit, h0, hmax, hmin,
                             ixpr, mxstep, mxhnil, mxordn, mxords)
    if output[-1] < 0:
        warning_msg = _msgs[output[-1]] + " Run with full_output = 1 to get quantitative information."
        warnings.warn(warning_msg, ODEintWarning)
    elif printmessg:
        warning_msg = _msgs[output[-1]]
        warnings.warn(warning_msg, ODEintWarning)

    if full_output:
        output[1]['message'] = _msgs[output[-1]]

    output = output[:-1]
    if len(output) == 1:
        return output[0]
    else:
        return output
