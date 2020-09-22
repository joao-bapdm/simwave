import numpy as np
from numpy.linalg import solve


def coeff(k, M, h=1, all=False):
    """ Coefficients for even order approximation of first derivative

    Parameters
    ----------
    k : int
        Index of wanted coefficient, i.e., k from g_k. k is
        smaller or equal to M / 2.
    M : int
        Order of approximation. Has to be an even number.
    h : float, optional
        Finite difference step size. Defaults to 1.
    all : bool, optional
        Ignores k, returning all coefficients up to M / 2.
        Defaults to False.

    Returns
    -------
    numpy array
        Coefficient(s) for approximation of order M.
    """

    if not M % 2 == 0:
        raise ValueError("Order M has to be an even number.")
    elif k > M / 2:
        raise ValueError(f"Expression only goes up to g_{M//2}!")

    M_ = M // 2
    A = [[m ** (2 * k + 1) for m in range(1, M_ + 1)] for k in range(M_)]
    b = [1 / (2 * h) if x == 0 else 0 for x in range(M // 2)]
    x = solve(A, b)

    x = x[k - 1] if not all else x

    return list(x)

def get_vd_coefficients(space_order, spacing):
    coefs = []
    for s in list(spacing):
        coefs+=coeff(k=1, M=space_order, h=s, all=True)

    return np.float32(coefs)



"""
Return a list of finite differences coefficients according to the space order

Parameters
----------
space_order : int
    Spatial order

Returns
----------
list
    List of FD coefficients
"""
def get_coefficients(space_order):

    coeffs = {
        2 : [1, -2, 1],
        4 : [-1/12, 16/12, -30/12, 16/12, -1/12],
        6 : [2/180, -27/180, 270/180, -490/180, 270/180, -27/180, 2/180],
        8 : [-9/5040, 128/5040, -1008/5040, 8064/5040, -14350/5040, 8064/5040, -1008/5040, 128/5040, -9/5040]
    }

    return coeffs[space_order]

"""
Calculate dt with CFL conditions
Based on https://library.seg.org/doi/pdf/10.1190/1.1444605 for the acoustic case

Parameters
----------
dimension : int
    Domain dimension. 2 (2D) or 3 (3D)
space_order : int
    Spatial order
spacing : tuple(float, float, float)
    Spacing between grid points
vel_model : grid
    Velocity model

Returns
----------
float
    dt in seconds
"""
def calc_dt(dimension, space_order, spacing, vel_model):

    # 2nd order in time
    a1 = 4

    # FD coeffs to the specific space order
    fd_coeffs = get_coefficients(space_order)

    a2 = dimension * np.sum( np.abs(fd_coeffs) )

    coeff = np.sqrt(a1/a2)

    # The CFL condtion is then given by
    # dt <= coeff * h / max(velocity)
    dt = coeff * np.min(spacing) / np.max(vel_model)

    return dt

"""
Calc the number of timesteps

Parameters
----------
time : int
    Propagation simulation time in miliseconds

dt : float
    Timestep variation in seconds

Returns
----------
int
    Number of timesteps
"""
def calc_num_timesteps(time, dt):

    return int(np.floor( (time/1000) / dt))
