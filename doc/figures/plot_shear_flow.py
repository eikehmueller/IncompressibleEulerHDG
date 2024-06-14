"""Compute and plot the initial pressure for the double layer shear flow in

    Guzmán, J., Shu, C.W. and Sequeira, F.A., 2017.
    "H (div) conforming and DG methods for incompressible Euler’s equations. "
    IMA Journal of Numerical Analysis, 37(4), pp.1733-1771.
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate

rho = 1 / 30


def Q_0x(z):
    """Initial velocity in the x-direction"""
    return np.where(
        z <= 0.0,
        np.tanh((np.pi + 2 * z) / (4 * np.pi * rho)),
        np.tanh((np.pi - 2 * z) / (4 * np.pi * rho)),
    )


def f(z):
    """Right hand side for the one-dimensional elliptic problem"""
    return np.where(
        z <= 0.0,
        1 - np.tanh((np.pi + 2 * z) / (4 * np.pi * rho)) ** 2,
        -1 + np.tanh((np.pi - 2 * z) / (4 * np.pi * rho)) ** 2,
    ) / (np.pi * rho)


Y = 0
Y_p = 0

h = 1e-3
X = np.arange(0, 1, h)
Z = (2 * X - 1) * np.pi

kmax = 28
fourier_coefficients = np.empty(kmax)
for k in range(kmax):
    fourier_coefficient = (
        1
        / np.pi
        * integrate.quad(
            lambda x: f(x),
            -np.pi,
            +np.pi,
            weight="sin",
            wvar=2 * k + 1,
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]
    )
    Y += fourier_coefficient * np.sin((2 * k + 1) * Z)
    Y_p += fourier_coefficient / (1 + (2 * k + 1) ** 2) * np.sin((2 * k + 1) * Z)
    fourier_coefficients[k] = fourier_coefficient


plt.clf()
plt.plot(X, Q_0x(Z), linewidth=2, color="green", label="$Q_{0,x}(y)$")
plt.plot(X, f(Z), linewidth=2, color="blue", label="$g(y)$ [exact]")
plt.plot(
    X, Y, linewidth=2, color="blue", linestyle="--", label="$g(y)$ [Fourier expansion]"
)
plt.plot(X, Y_p, linewidth=2, color="red", label="$p_{0,y}(y)$")
plt.plot([0, 1], [0, 0], linewidth=1, color="black", linestyle="--")
ax = plt.gca()
ax.set_xlim(0, 1)
ax.set_xlabel("y")
plt.legend(loc="upper right")
plt.savefig("shear_flow_initial_condition.pdf", bbox_inches="tight")

plt.clf()
K = np.arange(kmax)
plt.plot(
    np.abs(fourier_coefficients),
    marker="s",
    markersize=4,
    color="blue",
    linewidth=2,
    markeredgewidth=2,
    label="$|a_k|$ [expansion of $g(y)$]",
)
plt.plot(
    np.abs(fourier_coefficients) / (1 + (2 * K + 1) ** 2),
    marker="o",
    markersize="4",
    color="red",
    linewidth=2,
    markeredgewidth=2,
    markerfacecolor="white",
    label="$|a_k|/(1+(2k+1)^2)$ [expansion of $p_{0,y}(y)$]",
)

ax = plt.gca()
ax.set_yscale("log")
ax.set_xlabel("index $k$")
ax.set_ylabel("Fourier coefficient")
ax.grid(axis="y")
plt.legend(loc="lower left")
plt.savefig("shear_flow_fourier_coefficients.pdf", bbox_inches="tight")
