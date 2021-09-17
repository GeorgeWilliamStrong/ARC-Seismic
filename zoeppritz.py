# >>> Import Libraries

import numpy as np
from numpy import sin, cos, tan
import cmath as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rc('axes', axisbelow=True)
import random
from scipy import stats
from matplotlib.ticker import MaxNLocator
from scipy.fftpack import hilbert
from tqdm import tqdm


# >>> Isotropic Zoeppritz Solver

def isotropic_zoeppritz(vp1, vp2, vs1, vs2, p1, p2, i_angle):
    """
    input: upper and lower p-wave velocities (vp1, vp2), s-wave velocities
    (vs1, vs2), densities (p1, p2) and angle of incidence (i_angle)

    output: scattering matrix sm as defined in Aki & Richards (1980)
    containing all possible reflection/transmission coefficients
    """

    # use Snell's law and the ray parameter to determine all angles
    theta1 = np.radians(i_angle)
    ray_p = sin(theta1) / vp1
    theta2 = cm.asin(ray_p * vp2)
    phi1 = cm.asin(ray_p * vs1)
    phi2 = cm.asin(ray_p * vs2)

    # initialise P and R from the matrix form of the zoeppritz equations
    p = np.array([[-sin(theta1), -cos(phi1), sin(theta2), cos(phi2)],
                  [cos(theta1), -sin(phi1), cos(theta2), -sin(phi2)],
                  [2 * p1 * vs1 * sin(phi1) * cos(theta1), p1 * vs1 *
                   (1 - 2 * (sin(phi1) ** 2)), 2 * p2 * vs2 * sin(phi2) *
                   cos(theta2), p2 * vs2 * (1 - 2 * (sin(phi2) ** 2))],
                  [-p1 * vp1 * (1 - 2 * (sin(phi1) ** 2)), p1 * vs1 *
                   sin(2 * phi1), p2 * vp2 * (1 - 2 * (sin(phi2) ** 2)),
                   -p2 * vs2 * sin(2 * phi2)]])

    r = np.array([[sin(theta1), cos(phi1), -sin(theta2), -cos(phi2)],
                  [cos(theta1), -sin(phi1), cos(theta2), -sin(phi2)],
                  [2 * p1 * vs1 * sin(phi1) * cos(theta1), p1 * vs1 *
                   (1 - 2 * (sin(phi1) ** 2)), 2 * p2 * vs2 * sin(phi2) *
                   cos(theta2), p2 * vs2 * (1 - 2 * (sin(phi2) ** 2))],
                  [p1 * vp1 * (1 - 2 * (sin(phi1) ** 2)), -p1 * vs1 *
                   sin(2 * phi1), - p2 * vp2 * (1 - 2 * (sin(phi2) ** 2)),
                   p2 * vs2 * sin(2 * phi2)]])

    # invert P and solve for the scattering matrix: Q = P^{-1} R
    sm = np.dot(np.linalg.inv(p), r)

    # fix the sign of the imaginary component
    for i in range(4):
        for j in range(4):
            sm[i][j] = complex(sm[i][j].real, -sm[i][j].imag)

    return sm


# >>> Elastic Tensors

def isotropic_c(vp, vs, p):
    """
    input: p-wave velocity (vp), s-wave velocity (vs), density (p)

    output: 6x6 isotropic elastic tensor (c)
    """

    c = np.zeros((6, 6), dtype=float)
    mu = vs ** 2 * p  # shear modulus
    k = (vp ** 2 * p) - (4. / 3.) * mu  # bulk modulus
    c[0][0] = k + (4. / 3.) * mu
    c[1][1] = k + (4. / 3.) * mu
    c[2][2] = k + (4. / 3.) * mu
    c[3][3] = mu
    c[4][4] = mu
    c[5][5] = mu
    c[0][1] = k - (2. / 3.) * mu
    c[0][2] = k - (2. / 3.) * mu
    c[1][2] = k - (2. / 3.) * mu
    c[1][0] = c[0][1]  # exploiting tensor symmetry
    c[2][0] = c[0][2]
    c[2][1] = c[1][2]

    return c


def thomsen_c(vp, vs, p, epsilon, delta, gamma):
    """
    input: p-wave velocity (vp), s-wave velocity (vs), density (p)
    and Thomsen parameters (epsilon, gamma, delta)

    output: 6x6 vertical transverse isotropic elastic tensor (c)
    """

    c = np.zeros((6, 6), dtype=float)

    # convert delta into delta star for exact anisotropic result
    d_star = (1 - ((vs ** 2) / (vp ** 2))) * ((2 * delta) - epsilon)

    c[0][0] = 2. * ((vp ** 2) * p) * epsilon + ((vp ** 2) * p)
    c[1][1] = c[0][0]
    c[2][2] = (vp ** 2) * p
    c[3][3] = (vs ** 2) * p
    c[4][4] = c[3][3]
    c[5][5] = 2. * ((vs ** 2) * p) * gamma + ((vs ** 2) * p)
    c[0][1] = c[0][0] - 2. * c[5][5]
    c[0][2] = (np.sqrt((c[2][2] ** 2) * (2. * d_star + 1.) + c[2][2] * (c[0][0] - 3. * c[3][3]) + c[3][3] * (
            2. * c[3][3] - c[0][0])) / np.sqrt(2.)) - c[3][3]
    c[1][0] = c[0][1]
    c[1][2] = c[0][2]
    c[2][0] = c[0][2]
    c[2][1] = c[0][2]

    return c


def bond_transformation(c, angle):
    """
    input: 6x6 elastic tensor (c), transformation angle (angle)

    output: 6x6 elastic tensor rotated about X2-axis by angle
    """

    angle = np.radians(angle)
    m = np.zeros((6, 6), dtype=float)
    m[0][0] = cos(angle) ** 2
    m[0][2] = sin(angle) ** 2
    m[0][4] = -sin(2 * angle)
    m[1][1] = 1
    m[2][0] = sin(angle) ** 2
    m[2][2] = cos(angle) ** 2
    m[2][4] = sin(2 * angle)
    m[3][3] = cos(angle)
    m[3][5] = sin(angle)
    m[4][0] = 0.5 * sin(2 * angle)
    m[4][2] = -0.5 * sin(2 * angle)
    m[4][4] = cos(2 * angle)
    m[5][3] = -sin(angle)
    m[5][5] = cos(angle)
    c_rot = np.dot(np.dot(m, c), m.T)  # Bond transformation

    return c_rot


# >>> Incident Slowness Vectors

def monoclinic_christoffel(c, s):
    """
    input: 6x6 elastic tensor (c) and 3D slowness vector (s)

    output: 3x3 monoclinic christoffel matrix (c_mat) in complex form
    """

    c_mat = np.zeros((3, 3), dtype=complex)
    c_mat[0][0] = c[0][0] * s[0] ** 2 + c[5][5] * s[1] ** 2 + c[4][4] * s[2] ** 2 + 2 * c[0][5] * s[0] * s[1]
    c_mat[1][1] = c[5][5] * s[0] ** 2 + c[1][1] * s[1] ** 2 + c[3][3] * s[2] ** 2 + 2 * c[1][5] * s[0] * s[1]
    c_mat[2][2] = c[4][4] * s[0] ** 2 + c[3][3] * s[1] ** 2 + c[2][2] * s[2] ** 2 + 2 * c[3][4] * s[0] * s[1]
    c_mat[0][1] = c[0][5] * s[0] ** 2 + c[1][5] * s[1] ** 2 + c[3][4] * s[2] ** 2 + (c[0][1] + c[5][5]) * s[0] * s[1]
    c_mat[0][2] = (c[0][2] + c[4][4]) * s[0] * s[2] + (c[2][5] + c[3][4]) * s[1] * s[2]
    c_mat[1][2] = (c[2][5] + c[3][4]) * s[0] * s[2] + (c[1][2] + c[3][3]) * s[1] * s[2]
    c_mat[1][0] = c_mat[0][1]  # exploiting tensor symmetry
    c_mat[2][0] = c_mat[0][2]
    c_mat[2][1] = c_mat[1][2]

    return c_mat


def incident_slowness(a_angle, i_angle, c, p, test_vel=False):
    """
    input: azimuthal angle (a_angle), angle of incidence (i_angle) in
    degrees, 6x6 elastic tensor (c), density (p) and an option to check
    that the velocities are sensible (e.g. test_vel=True)

    output: 3x1 incident slowness matrix (s) containing each incident
    slowness vector (pi, si, ti)
    """

    # initialise incident slowness matrix s
    s = np.zeros((3, 3), dtype=complex)

    # define unit vector (n) parallel to wavefront normal
    n = np.array([cos(np.radians(a_angle)) * sin(np.radians(i_angle)),
                  sin(np.radians(a_angle)) * sin(np.radians(i_angle)),
                  cos(np.radians(i_angle))])

    # calculate density normalised monoclinic christoffel matrix
    m = monoclinic_christoffel(c, n) / float(p)

    # calculate eigenvalues of M and sort in ascending order
    w = np.sort(np.linalg.eig(m)[0])

    # assign eigenvalues vp, vs, vt based on magnitude: vt < vs < vp
    vt, vs, vp = np.sqrt(w)

    # test - check velocities are sensible
    if test_vel == True:
        print('vp, vs, vt =', vp, vs, vt)

    # define incident wave slowness vectors pi, si, ti
    s[0] = np.array([n[0] / vp, n[1] / vp, n[2] / vp], dtype=complex)  # spi
    s[1] = np.array([n[0] / vs, n[1] / vs, n[2] / vs], dtype=complex)  # ssi
    s[2] = np.array([n[0] / vt, n[1] / vt, n[2] / vt], dtype=complex)  # sti

    return s


# >>> Reflected Slowness Vectors

def vertical_slowness_components(s1, s2, p, c):
    """
    input: horizontal slowness components (s1, s2), density (p)
    and 6x6 elastic tensor (c)

    express bicubic equation for s3 as a polynomial and then
    find the roots (i.e. values of s3^2 where polynomial = 0)

    output: 3 unsorted vertical slowness components (s3)
    """

    w = (c[2][2] * c[3][3] * c[4][4] - c[2][2] * c[3][4] ** 2)

    x = (c[0][0] * c[2][2] * c[3][3] * s1 ** 2 - 2 * c[0][1] * c[2][2] * c[3][4] * s1 * s2 -
         c[0][2] ** 2 * c[3][3] * s1 ** 2 + 2 * c[0][2] * c[1][2] * c[3][4] * s1 * s2 -
         2 * c[0][2] * c[2][5] * c[3][3] * s1 * s2 + 2 * c[0][2] * c[2][5] * c[3][4] * s1 ** 2 -
         2 * c[0][2] * c[3][3] * c[4][4] * s1 ** 2 + 2 * c[0][2] * c[3][4] ** 2 * s1 ** 2 +
         2 * c[0][5] * c[2][2] * c[3][3] * s1 * s2 - 2 * c[0][5] * c[2][2] * c[3][4] * s1 ** 2 +
         c[1][1] * c[2][2] * c[4][4] * s2 ** 2 - c[1][2] ** 2 * c[4][4] * s2 ** 2 +
         2 * c[1][2] * c[2][5] * c[3][4] * s2 ** 2 - 2 * c[1][2] * c[2][5] * c[4][4] * s1 * s2 -
         2 * c[1][2] * c[3][3] * c[4][4] * s2 ** 2 + 2 * c[1][2] * c[3][4] ** 2 * s2 ** 2 -
         2 * c[1][5] * c[2][2] * c[3][4] * s2 ** 2 + 2 * c[1][5] * c[2][2] * c[4][4] * s1 * s2 +
         c[2][2] * c[3][3] * c[5][5] * s2 ** 2 - c[2][2] * c[3][3] * p -
         2 * c[2][2] * c[3][4] * c[5][5] * s1 * s2 + c[2][2] * c[4][4] * c[5][5] * s1 ** 2 -
         c[2][2] * c[4][4] * p - c[2][5] ** 2 * c[3][3] * s2 ** 2 +
         2 * c[2][5] ** 2 * c[3][4] * s1 * s2 - c[2][5] ** 2 * c[4][4] * s1 ** 2 -
         4 * c[2][5] * c[3][3] * c[4][4] * s1 * s2 + 4 * c[2][5] * c[3][4] ** 2 * s1 * s2 -
         c[3][3] * c[4][4] * p + c[3][4] ** 2 * p)

    y = (c[0][0] * c[1][1] * c[2][2] * s1 ** 2 * s2 ** 2 - c[0][0] * c[1][2] ** 2 * s1 ** 2 * s2 ** 2 -
         2 * c[0][0] * c[1][2] * c[2][5] * s1 ** 3 * s2 - 2 * c[0][0] * c[1][2] * c[3][3] * s1 ** 2 * s2 ** 2 -
         2 * c[0][0] * c[1][2] * c[3][4] * s1 ** 3 * s2 + 2 * c[0][0] * c[1][5] * c[2][2] * s1 ** 3 * s2 +
         c[0][0] * c[2][2] * c[5][5] * s1 ** 4 - c[0][0] * c[2][2] * p * s1 ** 2 -
         c[0][0] * c[2][5] ** 2 * s1 ** 4 - 2 * c[0][0] * c[2][5] * c[3][3] * s1 ** 3 * s2 -
         2 * c[0][0] * c[2][5] * c[3][4] * s1 ** 4 + c[0][0] * c[3][3] * c[4][4] * s1 ** 4 -
         c[0][0] * c[3][3] * p * s1 ** 2 - c[0][0] * c[3][4] ** 2 * s1 ** 4 -
         c[0][1] ** 2 * c[2][2] * s1 ** 2 * s2 ** 2 + 2 * c[0][1] * c[0][2] * c[1][2] * s1 ** 2 * s2 ** 2 +
         2 * c[0][1] * c[0][2] * c[2][5] * s1 ** 3 * s2 + 2 * c[0][1] * c[0][2] * c[3][3] * s1 ** 2 * s2 ** 2 +
         2 * c[0][1] * c[0][2] * c[3][4] * s1 ** 3 * s2 - 2 * c[0][1] * c[0][5] * c[2][2] * s1 ** 3 * s2 +
         2 * c[0][1] * c[1][2] * c[2][5] * s1 * s2 ** 3 + 2 * c[0][1] * c[1][2] * c[3][4] * s1 * s2 ** 3 +
         2 * c[0][1] * c[1][2] * c[4][4] * s1 ** 2 * s2 ** 2 - 2 * c[0][1] * c[1][5] * c[2][2] * s1 * s2 ** 3 -
         2 * c[0][1] * c[2][2] * c[5][5] * s1 ** 2 * s2 ** 2 + 2 * c[0][1] * c[2][5] ** 2 * s1 ** 2 * s2 ** 2 +
         2 * c[0][1] * c[2][5] * c[3][3] * s1 * s2 ** 3 + 4 * c[0][1] * c[2][5] * c[3][4] * s1 ** 2 * s2 ** 2 +
         2 * c[0][1] * c[2][5] * c[4][4] * s1 ** 3 * s2 + 2 * c[0][1] * c[3][3] * c[4][4] * s1 ** 2 * s2 ** 2 -
         2 * c[0][1] * c[3][4] ** 2 * s1 ** 2 * s2 ** 2 + 2 * c[0][1] * c[3][4] * p * s1 * s2 -
         c[0][2] ** 2 * c[1][1] * s1 ** 2 * s2 ** 2 - 2 * c[0][2] ** 2 * c[1][5] * s1 ** 3 * s2 -
         c[0][2] ** 2 * c[5][5] * s1 ** 4 + c[0][2] ** 2 * p * s1 ** 2 +
         2 * c[0][2] * c[0][5] * c[1][2] * s1 ** 3 * s2 + 2 * c[0][2] * c[0][5] * c[2][5] * s1 ** 4 +
         2 * c[0][2] * c[0][5] * c[3][3] * s1 ** 3 * s2 + 2 * c[0][2] * c[0][5] * c[3][4] * s1 ** 4 -
         2 * c[0][2] * c[1][1] * c[2][5] * s1 * s2 ** 3 - 2 * c[0][2] * c[1][1] * c[3][4] * s1 * s2 ** 3 -
         2 * c[0][2] * c[1][1] * c[4][4] * s1 ** 2 * s2 ** 2 + 2 * c[0][2] * c[1][2] * c[1][5] * s1 * s2 ** 3 +
         2 * c[0][2] * c[1][2] * c[5][5] * s1 ** 2 * s2 ** 2 -
         2 * c[0][2] * c[1][5] * c[2][5] * s1 ** 2 * s2 ** 2 + 2 * c[0][2] * c[1][5] * c[3][3] * s1 * s2 ** 3 -
         2 * c[0][2] * c[1][5] * c[3][4] * s1 ** 2 * s2 ** 2 -
         4 * c[0][2] * c[1][5] * c[4][4] * s1 ** 3 * s2 + 2 * c[0][2] * c[2][5] * p * s1 * s2 +
         2 * c[0][2] * c[3][3] * c[5][5] * s1 ** 2 * s2 ** 2 + 2 * c[0][2] * c[3][4] * p * s1 * s2 -
         2 * c[0][2] * c[4][4] * c[5][5] * s1 ** 4 + 2 * c[0][2] * c[4][4] * p * s1 ** 2 -
         c[0][5] ** 2 * c[2][2] * s1 ** 4 + 2 * c[0][5] * c[1][1] * c[2][2] * s1 * s2 ** 3 -
         2 * c[0][5] * c[1][2] ** 2 * s1 * s2 ** 3 - 2 * c[0][5] * c[1][2] * c[2][5] * s1 ** 2 * s2 ** 2 -
         4 * c[0][5] * c[1][2] * c[3][3] * s1 * s2 ** 3 - 2 * c[0][5] * c[1][2] * c[3][4] * s1 ** 2 * s2 ** 2 +
         2 * c[0][5] * c[1][2] * c[4][4] * s1 ** 3 * s2 + 2 * c[0][5] * c[1][5] * c[2][2] * s1 ** 2 * s2 ** 2 -
         2 * c[0][5] * c[2][2] * p * s1 * s2 - 2 * c[0][5] * c[2][5] * c[3][3] * s1 ** 2 * s2 ** 2 +
         2 * c[0][5] * c[2][5] * c[4][4] * s1 ** 4 + 4 * c[0][5] * c[3][3] * c[4][4] * s1 ** 3 * s2 -
         2 * c[0][5] * c[3][3] * p * s1 * s2 - 4 * c[0][5] * c[3][4] ** 2 * s1 ** 3 * s2 +
         2 * c[0][5] * c[3][4] * p * s1 ** 2 + c[1][1] * c[2][2] * c[5][5] * s2 ** 4 -
         c[1][1] * c[2][2] * p * s2 ** 2 - c[1][1] * c[2][5] ** 2 * s2 ** 4 -
         2 * c[1][1] * c[2][5] * c[3][4] * s2 ** 4 - 2 * c[1][1] * c[2][5] * c[4][4] * s1 * s2 ** 3 +
         c[1][1] * c[3][3] * c[4][4] * s2 ** 4 - c[1][1] * c[3][4] ** 2 * s2 ** 4 -
         c[1][1] * c[4][4] * p * s2 ** 2 - c[1][2] ** 2 * c[5][5] * s2 ** 4 + c[1][2] ** 2 * p * s2 ** 2 +
         2 * c[1][2] * c[1][5] * c[2][5] * s2 ** 4 + 2 * c[1][2] * c[1][5] * c[3][4] * s2 ** 4 +
         2 * c[1][2] * c[1][5] * c[4][4] * s1 * s2 ** 3 + 2 * c[1][2] * c[2][5] * p * s1 * s2 -
         2 * c[1][2] * c[3][3] * c[5][5] * s2 ** 4 + 2 * c[1][2] * c[3][3] * p * s2 ** 2 +
         2 * c[1][2] * c[3][4] * p * s1 * s2 + 2 * c[1][2] * c[4][4] * c[5][5] * s1 ** 2 * s2 ** 2 -
         c[1][5] ** 2 * c[2][2] * s2 ** 4 - 2 * c[1][5] * c[2][2] * p * s1 * s2 +
         2 * c[1][5] * c[2][5] * c[3][3] * s2 ** 4 - 2 * c[1][5] * c[2][5] * c[4][4] * s1 ** 2 * s2 ** 2 +
         4 * c[1][5] * c[3][3] * c[4][4] * s1 * s2 ** 3 - 4 * c[1][5] * c[3][4] ** 2 * s1 * s2 ** 3 +
         2 * c[1][5] * c[3][4] * p * s2 ** 2 - 2 * c[1][5] * c[4][4] * p * s1 * s2 -
         c[2][2] * c[5][5] * p * s1 ** 2 - c[2][2] * c[5][5] * p * s2 ** 2 + c[2][2] * p ** 2 +
         c[2][5] ** 2 * p * s1 ** 2 + c[2][5] ** 2 * p * s2 ** 2 + 2 * c[2][5] * c[3][3] * p * s1 * s2 +
         2 * c[2][5] * c[3][4] * p * s1 ** 2 + 2 * c[2][5] * c[3][4] * p * s2 ** 2 +
         2 * c[2][5] * c[4][4] * p * s1 * s2 + 4 * c[3][3] * c[4][4] * c[5][5] * s1 ** 2 * s2 ** 2 -
         c[3][3] * c[4][4] * p * s1 ** 2 - c[3][3] * c[4][4] * p * s2 ** 2 -
         c[3][3] * c[5][5] * p * s2 ** 2 + c[3][3] * p ** 2 - 4 * c[3][4] ** 2 * c[5][5] * s1 ** 2 * s2 ** 2 +
         c[3][4] ** 2 * p * s1 ** 2 + c[3][4] ** 2 * p * s2 ** 2 + 2 * c[3][4] * c[5][5] * p * s1 * s2 -
         c[4][4] * c[5][5] * p * s1 ** 2 + c[4][4] * p ** 2)

    z = (c[0][0] * c[1][1] * c[3][3] * s1 ** 2 * s2 ** 4 +
         2 * c[0][0] * c[1][1] * c[3][4] * s1 ** 3 * s2 ** 3 +
         c[0][0] * c[1][1] * c[4][4] * s1 ** 4 * s2 ** 2 - c[0][0] * c[1][1] * p * s1 ** 2 * s2 ** 2 +
         2 * c[0][0] * c[1][5] * c[3][3] * s1 ** 3 * s2 ** 3 +
         4 * c[0][0] * c[1][5] * c[3][4] * s1 ** 4 * s2 ** 2 +
         2 * c[0][0] * c[1][5] * c[4][4] * s1 ** 5 * s2 - 2 * c[0][0] * c[1][5] * p * s1 ** 3 * s2 +
         c[0][0] * c[3][3] * c[5][5] * s1 ** 4 * s2 ** 2 - c[0][0] * c[3][3] * p * s1 ** 2 * s2 ** 2 +
         2 * c[0][0] * c[3][4] * c[5][5] * s1 ** 5 * s2 - 2 * c[0][0] * c[3][4] * p * s1 ** 3 * s2 +
         c[0][0] * c[4][4] * c[5][5] * s1 ** 6 - c[0][0] * c[4][4] * p * s1 ** 4 -
         c[0][0] * c[5][5] * p * s1 ** 4 + c[0][0] * p ** 2 * s1 ** 2 -
         c[0][1] ** 2 * c[3][3] * s1 ** 2 * s2 ** 4 - 2 * c[0][1] ** 2 * c[3][4] * s1 ** 3 * s2 ** 3 -
         c[0][1] ** 2 * c[4][4] * s1 ** 4 * s2 ** 2 + c[0][1] ** 2 * p * s1 ** 2 * s2 ** 2 -
         2 * c[0][1] * c[0][5] * c[3][3] * s1 ** 3 * s2 ** 3 -
         4 * c[0][1] * c[0][5] * c[3][4] * s1 ** 4 * s2 ** 2 -
         2 * c[0][1] * c[0][5] * c[4][4] * s1 ** 5 * s2 + 2 * c[0][1] * c[0][5] * p * s1 ** 3 * s2 -
         2 * c[0][1] * c[1][5] * c[3][3] * s1 * s2 ** 5 - 4 * c[0][1] * c[1][5] * c[3][4] * s1 ** 2 * s2 ** 4 -
         2 * c[0][1] * c[1][5] * c[4][4] * s1 ** 3 * s2 ** 3 + 2 * c[0][1] * c[1][5] * p * s1 * s2 ** 3 -
         2 * c[0][1] * c[3][3] * c[5][5] * s1 ** 2 * s2 ** 4 -
         4 * c[0][1] * c[3][4] * c[5][5] * s1 ** 3 * s2 ** 3 -
         2 * c[0][1] * c[4][4] * c[5][5] * s1 ** 4 * s2 ** 2 + 2 * c[0][1] * c[5][5] * p * s1 ** 2 * s2 ** 2 -
         c[0][5] ** 2 * c[3][3] * s1 ** 4 * s2 ** 2 - 2 * c[0][5] ** 2 * c[3][4] * s1 ** 5 * s2 -
         c[0][5] ** 2 * c[4][4] * s1 ** 6 + c[0][5] ** 2 * p * s1 ** 4 +
         2 * c[0][5] * c[1][1] * c[3][3] * s1 * s2 ** 5 + 4 * c[0][5] * c[1][1] * c[3][4] * s1 ** 2 * s2 ** 4 +
         2 * c[0][5] * c[1][1] * c[4][4] * s1 ** 3 * s2 ** 3 - 2 * c[0][5] * c[1][1] * p * s1 * s2 ** 3 +
         2 * c[0][5] * c[1][5] * c[3][3] * s1 ** 2 * s2 ** 4 +
         4 * c[0][5] * c[1][5] * c[3][4] * s1 ** 3 * s2 ** 3 +
         2 * c[0][5] * c[1][5] * c[4][4] * s1 ** 4 * s2 ** 2 - 2 * c[0][5] * c[1][5] * p * s1 ** 2 * s2 ** 2 -
         2 * c[0][5] * c[3][3] * p * s1 * s2 ** 3 - 4 * c[0][5] * c[3][4] * p * s1 ** 2 * s2 ** 2 -
         2 * c[0][5] * c[4][4] * p * s1 ** 3 * s2 + 2 * c[0][5] * p ** 2 * s1 * s2 +
         c[1][1] * c[3][3] * c[5][5] * s2 ** 6 - c[1][1] * c[3][3] * p * s2 ** 4 +
         2 * c[1][1] * c[3][4] * c[5][5] * s1 * s2 ** 5 - 2 * c[1][1] * c[3][4] * p * s1 * s2 ** 3 +
         c[1][1] * c[4][4] * c[5][5] * s1 ** 2 * s2 ** 4 - c[1][1] * c[4][4] * p * s1 ** 2 * s2 ** 2 -
         c[1][1] * c[5][5] * p * s2 ** 4 + c[1][1] * p ** 2 * s2 ** 2 - c[1][5] ** 2 * c[3][3] * s2 ** 6 -
         2 * c[1][5] ** 2 * c[3][4] * s1 * s2 ** 5 - c[1][5] ** 2 * c[4][4] * s1 ** 2 * s2 ** 4 +
         c[1][5] ** 2 * p * s2 ** 4 - 2 * c[1][5] * c[3][3] * p * s1 * s2 ** 3 -
         4 * c[1][5] * c[3][4] * p * s1 ** 2 * s2 ** 2 - 2 * c[1][5] * c[4][4] * p * s1 ** 3 * s2 +
         2 * c[1][5] * p ** 2 * s1 * s2 - c[3][3] * c[5][5] * p * s1 ** 2 * s2 ** 2 -
         c[3][3] * c[5][5] * p * s2 ** 4 + c[3][3] * p ** 2 * s2 ** 2 -
         2 * c[3][4] * c[5][5] * p * s1 ** 3 * s2 - 2 * c[3][4] * c[5][5] * p * s1 * s2 ** 3 +
         2 * c[3][4] * p ** 2 * s1 * s2 - c[4][4] * c[5][5] * p * s1 ** 4 -
         c[4][4] * c[5][5] * p * s1 ** 2 * s2 ** 2 + c[4][4] * p ** 2 * s1 ** 2 + c[5][5] * p ** 2 * s1 ** 2 +
         c[5][5] * p ** 2 * s2 ** 2 - p ** 3)

    return np.roots(np.array([w, x, y, z], dtype=complex))


def reflected_transmitted_slowness(si, c1, c2, p1, p2, tol=1e-6):
    """
    input: incident slowness vector si (pi, si, ti), 6x6 upper and
    lower elastic tensors (c1, c2), upper and lower densities (p1, p2)

    output: 2x3 reflected/transmitted slowness matrix (rt) containing
    each corresponding slowness vector
    """

    # initialise reflected/transmitted slowness matrix SRT
    srt = np.zeros((2, 3, 3), dtype=complex)

    # solve bicubic equations for s31 and s32, take sqrt and sort
    s31 = np.sqrt(np.sort(vertical_slowness_components(si[0], si[1], p1, c1)))
    s32 = np.sqrt(np.sort(vertical_slowness_components(si[0], si[1], p2, c2)))

    # remove imaginary and real parts if insignificant and enforce +ve sign convention
    for i in range(3):
        if abs(s31[i].imag) < tol:
            s31[i] = s31[i].real
        if abs(s32[i].imag) < tol:
            s32[i] = s32[i].real
        if abs(s31[i].real) < tol:
            s31[i] = complex(0, abs(s31[i].imag))
        if abs(s32[i].real) < tol:
            s32[i] = complex(0, abs(s32[i].imag))

    # reflected slowness vectors
    srt[0][0] = np.array([si[0], si[1], s31[0]], dtype=complex)  # spr
    srt[0][1] = np.array([si[0], si[1], s31[1]], dtype=complex)  # ssr
    srt[0][2] = np.array([si[0], si[1], s31[2]], dtype=complex)  # str

    # transmitted slowness vectors
    srt[1][0] = np.array([si[0], si[1], s32[0]], dtype=complex)  # spt
    srt[1][1] = np.array([si[0], si[1], s32[1]], dtype=complex)  # sst
    srt[1][2] = np.array([si[0], si[1], s32[2]], dtype=complex)  # stt

    return srt


# >>> Polarization Vectors

def select_eigenvector(w, v, p, tol):
    """
    input: eigenvalues (w), eigenvectors (v), density (p) which is
    also corresponding eigenvector, tolerance (tol)

    output: correct polarization eigenvector
    """

    for i in range(3):
        if w[i] > p - tol and w[i] < p + tol:
            if v.T[i][0] < 0:
                return -1. * v.T[i]
            else:
                return v.T[i]


def polarization(srt, c1, c2, p1, p2, tol=1e-4):
    """
    input: 2x3 reflected/transmitted slowness matrix (srt), 6x6
    upper and lower elastic tensors (c1, c2) and upper and lower
    densities (p1, p2)

    output: 2x3 polarization matrix containing the 6 corresponding
    polarization vectors (prt)
    """

    # initialise reflected/transmitted polarization matrix PRT
    prt = np.zeros((2, 3, 3), dtype=complex)

    # find possible eigenvalues and corresponding eigenvectors
    prval, prvec = np.linalg.eig(monoclinic_christoffel(c1, srt[0][0]))
    srval, srvec = np.linalg.eig(monoclinic_christoffel(c1, srt[0][1]))
    trval, trvec = np.linalg.eig(monoclinic_christoffel(c1, srt[0][2]))
    ptval, ptvec = np.linalg.eig(monoclinic_christoffel(c2, srt[1][0]))
    stval, stvec = np.linalg.eig(monoclinic_christoffel(c2, srt[1][1]))
    ttval, ttvec = np.linalg.eig(monoclinic_christoffel(c2, srt[1][2]))

    # select correct eigenvectors for eigenvalue p1/p2
    prt[0][0] = select_eigenvector(prval, prvec, p1, tol)
    prt[0][1] = select_eigenvector(srval, srvec, p1, tol)
    prt[0][2] = select_eigenvector(trval, trvec, p1, tol)
    prt[1][0] = select_eigenvector(ptval, ptvec, p2, tol)
    prt[1][1] = select_eigenvector(stval, stvec, p2, tol)
    prt[1][2] = select_eigenvector(ttval, ttvec, p2, tol)

    return prt


# >>> Impedance Matrices

def x_impedence(sp, ss, st, pp, ps, pt, c):
    """
    input: slowness vectors (sp, ss, st), polarization vectors
    (pp, ps, pt) and 6x6 elastic tensor (c)

    output: 3x3 impedence matrix (x)
    """

    x = np.zeros((3, 3), dtype=complex)
    x[0][0] = pp[0]
    x[0][1] = ps[0]
    x[0][2] = pt[0]
    x[1][0] = pp[1]
    x[1][1] = ps[1]
    x[1][2] = pt[1]
    x[2][0] = -(c[0][2] * pp[0] + c[2][5] * pp[1]) * sp[0] - (c[1][2] * pp[1] + c[2][5] * pp[0]) * sp[1] - c[2][2] * \
        pp[2] * sp[2]
    x[2][1] = -(c[0][2] * ps[0] + c[2][5] * ps[1]) * ss[0] - (c[1][2] * ps[1] + c[2][5] * ps[0]) * ss[1] - c[2][2] * \
        ps[2] * ss[2]
    x[2][2] = -(c[0][2] * pt[0] + c[2][5] * pt[1]) * st[0] - (c[1][2] * pt[1] + c[2][5] * pt[0]) * st[1] - c[2][2] * \
        pt[2] * st[2]

    return x


def y_impedence(sp, ss, st, pp, ps, pt, c):
    """
    input: slowness vectors (sp, ss, st), polarization vectors
    (pp, ps, pt), 6x6 elastic tensor (c)

    output: 3x3 impedence matrix (y)
    """

    y = np.zeros((3, 3), dtype=complex)
    y[0][0] = -(c[4][4] * sp[0] + c[3][4] * sp[1]) * pp[2] - (c[4][4] * pp[0] + c[3][4] * pp[1]) * sp[2]
    y[0][1] = -(c[4][4] * ss[0] + c[3][4] * ss[1]) * ps[2] - (c[4][4] * ps[0] + c[3][4] * ps[1]) * ss[2]
    y[0][2] = -(c[4][4] * st[0] + c[3][4] * st[1]) * pt[2] - (c[4][4] * pt[0] + c[3][4] * pt[1]) * st[2]
    y[1][0] = -(c[3][4] * sp[0] + c[3][3] * sp[1]) * pp[2] - (c[3][4] * pp[0] + c[3][3] * pp[1]) * sp[2]
    y[1][1] = -(c[3][4] * ss[0] + c[3][3] * ss[1]) * ps[2] - (c[3][4] * ps[0] + c[3][3] * ps[1]) * ss[2]
    y[1][2] = -(c[3][4] * st[0] + c[3][3] * st[1]) * pt[2] - (c[3][4] * pt[0] + c[3][3] * pt[1]) * st[2]
    y[2][0] = pp[2]
    y[2][1] = ps[2]
    y[2][2] = pt[2]

    return y


# >>> Anisotropic Zoeppritz Equations

def anisotropic_zoeppritz_equations(xu, xl, yu, yl):
    """
    input: 3x3 upper and lower impedence matrices (xu, xl, yu, yl)

    output: 3x3 reflection and transmission matrices (r, t)
    """

    try:
        a = np.dot((np.linalg.inv(xu)), xl) + np.dot((np.linalg.inv(yu)), yl)
        b = np.dot((np.linalg.inv(xu)), xl) - np.dot((np.linalg.inv(yu)), yl)
        r = np.dot(b, (np.linalg.inv(a)))
        t = np.dot(2., (np.linalg.inv(a)))
    except:

        # solve without Yu^(-1) if Yu is singular
        try:
            a = np.dot(np.dot(np.linalg.inv(xu), xl), np.dot(np.linalg.inv(yl), yu)) + np.eye(3)
            b = np.dot(np.dot(np.linalg.inv(xu), xl), np.dot(np.linalg.inv(yl), yu)) - np.eye(3)
            r = np.dot(np.linalg.inv(a), b)
            t = np.dot(2., np.dot(np.dot(np.linalg.inv(yl), yu), np.linalg.inv(a)))
        except:

            # solve without Xu^(-1) if Xu is singular
            try:
                a = np.eye(3) + np.dot(np.dot(np.linalg.inv(yu), yl),
                                       np.dot(np.linalg.inv(xl), xu))
                b = np.eye(3) - np.dot(np.dot(np.linalg.inv(yu), yl),
                                       np.dot(np.linalg.inv(xl), xu))
                r = np.dot(np.linalg.inv(a), b)
                t = np.dot(2., np.dot(np.dot(np.linalg.inv(xl), xu),
                                      np.linalg.inv(a)))
            except:

                # solve without Xu^(-1) and Yu^(-1) if both singular
                try:
                    a = np.dot((np.linalg.inv(yl)), yu) + \
                        np.dot((np.linalg.inv(xl)), xu)
                    b = np.dot((np.linalg.inv(yl)), yu) - \
                        np.dot((np.linalg.inv(xl)), xu)
                    c = np.dot(2., np.dot((np.linalg.inv(yl)), yu))
                    d = np.dot((np.linalg.inv(xl)), xu)
                    r = np.dot((np.linalg.inv(a)), b)
                    t = np.dot(np.dot(c, np.linalg.inv(a)), d)
                except:
                    raise Exception('inversion error')

    return r, t


# >>> Anisotropic Zoeppritz Solver

def anisotropic_zoeppritz(c1, c2, p1, p2, i_angle, a_angle, p_white=1e-7):
    """
    input: 6x6 upper and lower elastic tensors (c1, c2), upper and
    lower densities (p1, p2), incident angle (i_angle), azimuthal
    angle (a_angle) in degrees

    output: 3x3 reflection and transmission matrices (r, t)
    """

    # prewhiten elastic tensors to stabilize the solution
    c1 = c1 + p_white * np.linalg.norm(c1)
    c2 = c2 + p_white * np.linalg.norm(c2)

    # calculate incident slowness vectors
    s = incident_slowness(a_angle, i_angle, c1, p1)

    # calculate reflected/transmitted slowness vectors
    srt = reflected_transmitted_slowness(s[0], c1, c2, p1, p2)

    # calculate reflected/transmitted polarization vectors
    prt = polarization(srt, c1, c2, p1, p2)

    # determine upper and lower X and Y impedence matrices
    xu = x_impedence(srt[0][0], srt[0][1], srt[0][2], prt[0][0], prt[0][1], prt[0][2], c1)
    xl = x_impedence(srt[1][0], srt[1][1], srt[1][2], prt[1][0], prt[1][1], prt[1][2], c2)
    yu = y_impedence(srt[0][0], srt[0][1], srt[0][2], prt[0][0], prt[0][1], prt[0][2], c1)
    yl = y_impedence(srt[1][0], srt[1][1], srt[1][2], prt[1][0], prt[1][1], prt[1][2], c2)

    # solve for 3x3 reflection and transmission matrices (r, t)
    r, t = anisotropic_zoeppritz_equations(xu, xl, yu, yl)

    return r, t


# >>> Plotting

def plot_coeff(x, mag, phase, angle=True, maxoff=5000):
    """
        input: x-axis list (x), magnitude list (mag), phase list (phase) and
        if angle=True plot angle on x-axis else plot offset on x-axis
        
        output: plot magnitude (solid line) and phase (dotted line) vs angle
        """

    fig, ax1 = plt.subplots(figsize=(11, 6))
    plt.grid()
    if angle == True:
        ax1.set_xlabel('Angle (deg.)')
    else:
        ax1.set_xlabel('Offset (m)')
    plt.ylim(0, 1.005)
    ax1.plot(x, mag, color='C0', label='magnitude')
    ax1.set_ylabel('Magnitude')
    ax2 = ax1.twinx()
    ax2.plot(x, phase, linestyle=':', color='C0', label='phase')
    ax2.set_ylabel('Phase (deg.)')
    plt.ylim(-200, 200)
    if angle == True:
        plt.xlim(0, 90)
    else:
        plt.xlim(0, maxoff)
    # plt.savefig('zoeppritz.png', dpi=1000) # uncomment to save figure as image
    plt.show()


def isotropic_plot(vp1, vp2, vs1, vs2, p1, p2, config='Rpp', d=0, maxoff=5000, res=300):
    """
        input: upper and lower p-wave velocities (vp1, vp2), s-wave velocities
        (vs1, vs2), densities (p1, p2), coeff type (e.g. config='Rpp' for p-p
        reflection or 'Tps' for p-s transmission), depth (d) if > 0 converts
        x-axis to offset, max offset (maxoff), and resolution (res)
        
        output: plots graph of magnitude, phase vs angle of incidence
        """

    # create list of incidence angles
    angles = np.linspace(0, 90, res)

    # create index to extract correct coefficient from scattering matrix Q
    if config == 'Rpp':
        ind = (0, 0)
    elif config == 'Rps':
        ind = (1, 0)
    elif config == 'Tpp':
        ind = (2, 0)
    elif config == 'Tps':
        ind = (3, 0)
    else:
        raise Exception('config type not recognized')

    # compute coeff for each angle and convert to magnitude and phase
    coeff = np.array([isotropic_zoeppritz(vp1, vp2, vs1, vs2, p1, p2, i)[ind[0]][ind[1]] for i in angles])
    mag = abs(coeff)
    phase = np.degrees(np.array([cm.phase(i) for i in coeff]))

    # plot mag and phase vs angle or offset
    if d > 0:
        offsets = [np.tan(np.radians(angles[i])) * d * 2 for i in range(0, res)]
        plot_coeff(offsets, mag, phase, angle=False, maxoff=maxoff)
    else:
        plot_coeff(angles, mag, phase)


def anisotropic_plot(c1, c2, p1, p2, a_angle, config='Rpp', d=0, maxoff=5000, res=300, p_white=1e-7):
    """
    input: 6x6 upper and lower elastic tensors (c1, c2), upper and
    lower densities (p1, p2), azimuthal angle (a_angle) in degrees,
    coeff type (e.g. config='Rpp' for p-p reflection or 'Tps' for
    p-s transmission), depth (d) if > 0 converts x-axis to offset, 
    max offset (maxoff), resolution (res) and pre-whitening (p_white)

    output: plots graph of magnitude, phase vs angle of incidence
    """

    # create list of incidence angles
    angles = np.linspace(0, 90, res)

    # create index to extract correct coefficient from scattering matrix Q
    if config == 'Rpp':
        ind = (0, 0, 0)
    elif config == 'Rps':
        ind = (0, 1, 0)
    elif config == 'Rpt':
        ind = (0, 2, 0)
    elif config == 'Tpp':
        ind = (1, 0, 0)
    elif config == 'Tps':
        ind = (1, 1, 0)
    elif config == 'Tpt':
        ind = (1, 2, 0)
    else:
        raise Exception('config type not recognized')

    # compute coeff for each angle and convert to magnitude and phase
    coeff = np.array([anisotropic_zoeppritz(c1, c2, p1, p2, i,
                                            a_angle, p_white=p_white)[ind[0]][ind[1]][ind[2]] for i in angles])
    mag = abs(coeff)
    phase = np.degrees(np.array([cm.phase(i) for i in coeff]))

    # plot mag and phase vs angle or offset
    if d > 0:
        offsets = [np.tan(np.radians(angles[i])) * d * 2 for i in range(0, res)]
        plot_coeff(offsets, mag, phase, angle=False, maxoff=maxoff)
    else:
        plot_coeff(angles, mag, phase)
