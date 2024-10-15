# Hybridisable DG discretisations of the incompressible Euler equations

Implementation for several advanced discretisation methods and timestepping methods for the incompressible Euler equations.

## Overview
The goal of this work is to implement higher order hybridisable DG discretisations in space and time for the incompressible Euler equations similar to what has been done in [Ueckermann and Lermusiaux (2016)](https://www.sciencedirect.com/science/article/pii/S0021999115007688?casa_token=aQP8a2IuX-MAAAAA:7KlvJnlSAoFO229d61uDrHxbyoJiYnoeE7laDV0pfrGDENnq4cmYVRGkXLTZuZnbmkX19hF_lQ) for the incompressible Navier Stokes equations. This is achieved by extending the methods in [Guzmán, Shu, Sequeira (2017)](https://academic.oup.com/imajna/article/37/4/1733/2670304?login=false).

### Mathematical details

The incompressible Euler equations that are to be solved here are given by

$$
\begin{aligned}
\frac{dQ}{dt} + \nabla p + (Q\cdot \nabla) Q &= f\\
\nabla\cdot Q &= 0
\end{aligned}
$$

The code implements a range of numerical schemes:

1. The **fully implicit** method based on a **conforming $\text{RT}_0\times \text{DG}_0$ discretisation** described in [Guzmán, Shu, Sequeira (2017)](https://academic.oup.com/imajna/article/37/4/1733/2670304?login=false), see [conforming_implicit.py](IncompressibleEulerHDG/tree/main/src/timesteppers/conforming_implicit.py).
2. The **fully implicit** method based on a **$[\text{DG}_{k+1}]^2\times \text{DG}_k$ DG discretisation** also described in [Guzmán, Shu, Sequeira (2017)](https://academic.oup.com/imajna/article/37/4/1733/2670304?login=false), see [dg_implicit.py](IncompressibleEulerHDG/tree/main/src/timesteppers/dg_implicit.py)
3. A **fully implicit hybridisable DG variant** of the fully implicit DG discretisation, see [hdg_implicit.py](IncompressibleEulerHDG/tree/main/src/timesteppers/hdg_implicit.py)
4. A **hybridisable DG variant** of the DG discretisation which uses **Chorin's projection method** to split the implicit update into the computation of a tentative velocity followed by a pressure correction that enforces the divergence-free constraint on the velocity, see [hdg_implicit.py](IncompressibleEulerHDG/tree/main/src/timesteppers/hdg_implicit.py).
5. A generalisation of the **hybridisable DG variant** to IMEX timesteppers. The computation of the update at each stage can be done either fully implicitly or with a Richardson iteration that is preconditioned with a projection method, [hdg_imex.py](IncompressibleEulerHDG/tree/main/src/timesteppers/hdg_imex.py).

A stationary exact solution of the incompressible Euler equations in the domain $\Omega = [0,1]\times [0,1]$ is given by $f = 0$ and

$$
\begin{aligned}
    Q_s(x,y) &= \left(-C(x) S(y),S(x) C(y)\right)^\top\\
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

For further details on the used discretisation and timestepping methods see [arxiv preprint](http://arxiv.org/abs/2410.09790).

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