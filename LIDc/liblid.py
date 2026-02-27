# Wrapper functions for the C++ functions

import liblid


def integrate_dream(do, x0, nhat, time=None, line_averaged=False):
    """
    Evaluate the line-integrated density for the given DREAMOutput object
    (or name of file containing DREAMOutput object).

    :param do:            DREAMOutput object or name of HDF5 file containing DREAMOutput object.
    :param x0:            Coordinates of the detector origin.
    :param nhat:          Coordinates of detector viewing direction.
    :param time:          Time point for which to evaluate the line-integrated density.
    :param line_averaged: Return the line-averaged density instead of the line-integrated density.
    """
    if type(do) == str:
        t, n, L = liblid.integrate_dream_h5(do, x0, nhat, time)
    elif type(do) == dict:
        t, n, L = liblid.integrate_dream(do, x0, nhat, time)
    else:   # Assume DREAMOutput
        d = {
            'grid': {
                't': do.grid.t[:],
                'r': do.grid.r[:],
                'r_f': do.grid.r_f[:],
                'dr': do.grid.dr[:],
                'R0': do.grid.eq.R0[:],
                'Z0': do.grid.eq.Z0[:],
                # eq
                'eq': {
                    'RMinusR0': do.grid.eq.RMinusR0[:],
                    'RMinusR0_f': do.grid.eq.RMinusR0_f[:],
                    'ZMinusZ0': do.grid.eq.ZMinusZ0[:],
                    'ZMinusZ0_f': do.grid.eq.ZMinusZ0_f[:],
                    'theta': do.grid.eq.theta[:]
                }
            },
            'eqsys': {
                'n_cold': do.eqsys.n_cold[:]
            }
        }

        t, n, L = liblid.integrate_dream(d, x0, nhat, time)

    if line_averaged:
        return t, n/L
    else:
        return t, n


def greens_function(filename, x0s, nhats):
    """
    Evaluate the Green's function for equilibrium in the given LUKE equilibrium
    (or object), and the given lines-of-sight.

    :param filename:      Name of file containing LUKE equilibrium to use.
    :param x0s:           Coordinates of the detector origin.
    :param nhats:         Coordinates of detector viewing direction.
    """
    return liblid.greens_function(filename, x0s, nhats)

