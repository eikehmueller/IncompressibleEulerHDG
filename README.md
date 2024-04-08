# Hybridisable DG discretisations of the incompressible Euler equations

Implementation for several advanced discretisation methods and timestepping methods for the incompressible Euler equations.

## Overview
The goal of this work is to implement higher order hybridisable DG discretisations in space and time for the incompressible Euler equations similar to what has been done in [Ueckermann and Lermusiaux (2016)](https://www.sciencedirect.com/science/article/pii/S0021999115007688?casa_token=aQP8a2IuX-MAAAAA:7KlvJnlSAoFO229d61uDrHxbyoJiYnoeE7laDV0pfrGDENnq4cmYVRGkXLTZuZnbmkX19hF_lQ) for the incompressible Navier Stokes equations. This is achieved by extending the methods in [Guzmán,Shu,Sequeira (2017)](https://academic.oup.com/imajna/article/37/4/1733/2670304?login=false).

### Mathematical details

The incompressible Euler equations that are to be solved here are given by

$$
\begin{aligned}
\frac{dQ}{dt} + \nabla p + (Q\cdot \nabla) Q &= f\\
\nabla\cdot Q &= 0
\end{aligned}
$$

A stationary exact solution is given by $f = 0$

$$
\begin{aligned}
    Q_s(x,y) &= \begin{pmatrix}-C(x) S(y)\\S(x) C(y)\end{pmatrix}\\
    p_s(x,y) &= p_0 - C(x) C(y)
\end{aligned}
$$

where $S(z) = \sin\left(\frac{2z-1}{2}\pi\right)$ and $C(z) = \cos\left(\frac{2z-1}{2}\pi\right)$.

From this a divergence free time-dependent solution can be constructed as

$$
\begin{aligned}
    Q(x,y,t) &= \Psi(t) Q_s(x,y)\\
    p(x,y,t) &= \Psi(t)^2 p_s(x,y)
\end{aligned}
$$

and $f = \frac{d\Psi}{dt} Q_s(x,y)$.

For further details on the used discretisation and timestepping methods see [here](https://github.com/eikehmueller/IncompressibleEulerHDG/blob/gh-pages/discretisation.pdf).

## Installation
To install this package run 
```python -m pip install .```
as usual after installing the dependencies (see below).

If you want to edit the code, you might prefer to install in editable mode with
```python -m pip install --editable .```

## Running the code
The main script is `driver.py` in the `src` directory. Run

```
python driver.py --help
```

to see a list of command line options.

## Dependencies
### Firedrake
See [here](https://www.firedrakeproject.org/download.html) for Firedrake installation instructions.