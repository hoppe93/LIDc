# Wrapper functions for the C++ functions

import liblid


def integrate_dream(do, x0, nhat, line_averaged=False):
    """
    Evaluate the line-integrated density for the given DREAMOutput object
    (or name of file containing DREAMOutput object).

    :param do:   DREAMOutput object or name of HDF5 file containing DREAMOutput object.
    :param x0:   Coordinates of the detector origin.
    :param nhat: Coordinates of detector viewing direction.
    """
    if type(do) == str:
        t, n, L = liblid.integrate_dream_h5(do, x0, nhat)
    elif type(do) == dict:
        t, n, L = liblid.integrate_dream(do, x0, nhat)
    else:   # Assume DREAMOutput
        d = {
            'grid': {
                't': do.grid.t[:],
                'r': do.grid.r[:],
                'r_f': do.grid.r_f[:],
                'dr': do.grid.dr[:],
                'R0': do.grid.R0[:],
                # eq
                'eq': {
                    'ROverR0': do.grid.eq.ROverR0[:],
                    'ROverR0_f': do.grid.eq.ROverR0_f[:],
                    'Z': do.grid.eq.Z[:],
                    'Z_f': do.grid.eq.Z_f[:],
                    'theta': do.grid.theta[:]
                }
            },
            'eqsys': {
                'n_cold': do.eqsys.n_cold[:]
            }
        }

        t, n, L = liblid.integrate_dream(d, x0, nhat)

    if line_averaged:
        return t, n/L
    else:
        return t, n


