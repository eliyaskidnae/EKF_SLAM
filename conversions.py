import math
import Feature as f
import numpy as np
from math import *
import math
from Feature import *

def v2v(v):
    """
    Identity transformation. Returns the same vector.

    :param v: input vector
    :return: output vector
    """

    return v

def J_v2v(v):
    """
    Jacobian of the identity transformation. Returns the identity matrix of the same dimensionality as the input vector.

    :param v: input vector
    :return: Identity matrix of the same dimensionality as the input vector.
    """
    return np.eye(len(v))

def p2c(p):  # polar to cartesian conversion
    """
    Converts from a 2D Polar coordinate to its corresponding 2D Cartesian coordinate:

    .. math::
        p & = \\begin{bmatrix} \\rho \\\\ \\theta \\end{bmatrix} \\\\
        c &= p2c \\left(\\begin{bmatrix} x \\\\ y \\end{bmatrix} =  \\begin{bmatrix} \\rho \\cos(\\theta) \\\\ \\rho \\sin(\\theta) \\end{bmatrix}\\right)
        :label: eq-p2c

    :param p: point in polar coordinates
    :return: point in cartesian coordinates
    """
    r,theta= p
    x = r*cos(theta)
    y = r*sin(theta)
    
    return f.CartesianFeature(np.array([x,y]).reshape(2,1))
    


def J_p2c(p):  # polar to cartesian conversion
    """
    Jacobian of the 2D Polar to cartesian conversion:

    .. math::
        J_{p2c} = \\begin{bmatrix} \\frac{\\partial x}{\\partial \\rho} & \\frac{\\partial x}{\\partial \\theta} \\\\ \\frac{\\partial y}{\\partial \\rho} & \\frac{\\partial y}{\\partial \\theta} \\end{bmatrix} = \\begin{bmatrix} \\cos(\\theta) & -\\rho \\sin(\\theta) \\\\ \\sin(\\theta) & \\rho \\cos(\\theta) \\end{bmatrix}
        :label: eq-Jp2c

    :param p: linearization point in polar coordinates
    :return: Jacobian matrix :math:`J_{p2c}` (eq. :eq:`eq-Jp2c`)
    """
    r,theta= p   
    a = cos(theta[0])
    b = r[0]*cos(theta[0])
    c = sin(theta[0])
    d = -r[0]*sin(theta[0])
    # print(r,theta)
    
    J=np.array([ [a   ,     b ],
                 [c   ,     d ]]).reshape(2,2)
    # print("Jacobian",J)
    
    return J
def c2p(c):  # cartesian to spherical conversion
    """
    2D Cartesian to polar conversion:

    .. math::
        c &= \\begin{bmatrix} x \\\\ y \\end{bmatrix} \\\\
        p & = c2p\\left(\\begin{bmatrix} \\rho \\\\ \\theta \\end{bmatrix} = \\begin{bmatrix} \\sqrt{x^2+y^2} \\\\ atan2(y,x) \\end{bmatrix}\\right)
        :label: eq-c2p

    :param c: point in cartesian coordinates
    :return: point in polar coordinates
    """
    x , y = c
    r= math.sqrt(x**2 + y**2)
    theta = math.atan2(y,x)
    
    return f.PolarFeature(np.array([[r],[theta]]).reshape(2,1))


def J_c2p(c):  # cartesian to spherical conversion
    """
    Jacobian of the 2D Cartesian to polar conversion:

    .. math::
        J_{c2p} = \\begin{bmatrix} \\frac{\\partial \\rho}{\\partial x} & \\frac{\\partial \\rho}{\\partial y} \\\\ \\frac{\\partial \\theta}{\\partial x} & \\frac{\\partial \\theta}{\\partial y} \\end{bmatrix} = \\begin{bmatrix} \\frac{x}{\\sqrt{x^2+y^2}} & \\frac{y}{\\sqrt{x^2+y^2}} \\\\ -\\frac{y}{x^2+y^2} & \\frac{x}{x^2+y^2} \\end{bmatrix}
        :label: eq-Jc2p

    :param c: point in cartesian coordinates
    :return: Jacobian matrix :math:`J_{c2p}` (eq. :eq:`eq-Jc2p`)
    """
    x , y = c
    r= math.sqrt(x**2 + y**2)
    theta = math.atan2(y,x)
    a =x/r
    b = y/r
    c = (-y / (x**2 + y**2))
    d = (x / (x**2 + y**2))
    J_c2p = np.array([[a,            b],
                      [c,            d]
    ]).reshape(2,2)
    
    # print("J2p" , J_c2p)
    return J_c2p
    
    


def s2c(s):  # spherical to cartesian conversion
    """
    .. image:: img/3D_Spherical.png
        :width: 300px
        :align: center
        :alt: Spherical to Cartesian conversion image

    3D Spherical to cartesian conversion:

    .. math::
        s & = \\begin{bmatrix} \\rho \\\\ \\theta \\\\ \\varphi \\end{bmatrix} \\\\
        c &= s2c \\left(\\begin{bmatrix} x \\\\ y \\\\ z \\end{bmatrix} =  \\begin{bmatrix} \\rho \\sin(\\theta) \\cos(\\varphi) \\\\ \\rho \\sin(\\theta) \\sin(\\varphi) \\\\ \\rho \\cos(\\theta) \\end{bmatrix}\\right)
        :label: eq-s2c

    :param s: point in spherical coordinates
    :return: point in cartesian coordinates
    """
    r, theta, phi = s
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return np.array([x, y, z])



def J_s2c( s):  # spherical to cartesian conversion
    """
    Jacobian of the 3D Spherical to cartesian conversion:

    .. math::
        J_{s2c} = \\begin{bmatrix} \\frac{\\partial x}{\\partial \\rho} & \\frac{\\partial x}{\\partial \\theta} & \\frac{\\partial x}{\\partial \\varphi} \\\\ \\frac{\\partial y}{\\partial \\rho} & \\frac{\\partial y}{\\partial \\theta} & \\frac{\\partial y}{\\partial \\varphi} \\\\ \\frac{\\partial z}{\\partial \\rho} & \\frac{\\partial z}{\\partial \\theta} & \\frac{\\partial z}{\\partial \\varphi} \\end{bmatrix} = \\begin{bmatrix} \\sin(\\theta)\\cos(\\varphi) & \\rho\\cos(\\theta)\\cos(\\varphi) & -\\rho\\sin(\\theta)\\sin(\\varphi) \\\\ \\sin(\\theta)\\sin(\\varphi) & \\rho\\cos(\\theta)\\sin(\\varphi) & \\rho\\sin(\\theta)\\cos(\\varphi) \\\\ \\cos(\\theta) & -\\rho\\sin(\\theta) & 0 \\end{bmatrix}
        :label: eq-Js2c

    :param s: linearization point in spherical coordinates
    :return: Jacobian matrix :math:`J_{s2c}` (eq. :eq:`eq-Js2c`)
    """

    r, theta, phi = s
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    sin_phi = sin(phi)
    cos_phi = cos(phi)

    return np.array([
        [sin_theta * cos_phi, r * cos_theta * cos_phi, -r * sin_theta * sin_phi],
        [sin_theta * sin_phi, r * cos_theta * sin_phi, r * sin_theta * cos_phi],
        [cos_theta, -r * sin_theta, 0]
    ]).reshape(3,3)



def c2s(c):  # cartesian to spherical conversion
    """
    3D Cartesian to spherical conversion:

    .. math::
        c &= \\begin{bmatrix} x \\\\ y \\\\ z \\end{bmatrix} \\\\
        s & = c2s \\left(\\begin{bmatrix} \\rho \\\\ \\theta \\\\ \\varphi \\end{bmatrix} = \\begin{bmatrix} \\sqrt{x^2+y^2+z^2} \\\\ atan2(\\sqrt{x^2+y^2},{z}) \\\\ atan2({y},{x}) \\end{bmatrix}\\right)
        :label: eq-c2s

    :param c: point in cartesian coordinates
    :return: point in spherical coordinates
    """

    x, y, z = c
    r =   sqrt(x**2 + y**2 + z**2)
    theta = atan2(sqrt(x**2 + y**2), z)
    phi = atan2(y, x)
    return np.array([r, theta, phi])

def J_c2s(c):  # cartesian to spherical conversion
    """
    Jacobian of the 3D Cartesian to spherical conversion:

    .. math::
        J_{c2s} = \\begin{bmatrix} \\frac{\\partial \\rho}{\\partial x} & \\frac{\\partial \\rho}{\\partial y} & \\frac{\\partial \\rho}{\\partial z} \\\\ \\frac{\\partial \\theta}{\\partial x} & \\frac{\\partial \\theta}{\\partial y} & \\frac{\\partial \\theta}{\\partial z} \\\\ \\frac{\\partial \\varphi}{\\partial x} & \\frac{\\partial \\varphi}{\\partial y} & \\frac{\\partial \\varphi}{\\partial z} \\end{bmatrix} = \\begin{bmatrix} \\frac{x}{\\sqrt{x^2+y^2+z^2}} & \\frac{y}{\\sqrt{x^2+y^2+z^2}} & \\frac{z}{\\sqrt{x^2+y^2+z^2}} \\\\ \\frac{y}{x^2+y^2} & \\frac{x}{x^2+y^2} & 0 \\\\ \\frac{-x z}{(x^2+y^2)\\sqrt{x^2+y^2}} & \\frac{-y z}{(x^2+y^2)\\sqrt{x^2+y^2}} & \\frac{\\sqrt{x^2+y^2}}{x^2+y^2} \\end{bmatrix}
        :label: eq-Jc2s

    :param c: linearization point in cartesian coordinates
    :return: Jacobian matrix :math:`J_{c2s}` (eq. :eq:`eq-Jc2s`)
    """

    x, y, z = c
    r = sqrt(x**2 + y**2 + z**2)
    r_sqrt_xy = sqrt(x**2 + y**2)
    r_cube = r**3

    return np.array([
        [x / r, y / r, z / r],
        [y / r_sqrt_xy, -x / r_sqrt_xy, 0],
        [-x * z / r_cube, -y * z / r_cube, r_sqrt_xy / r_cube]
    ]).reshape(3,3)



