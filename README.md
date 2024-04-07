# Hybridisable DG discretisations of the incompressible Euler equations

Implementation for several advanced discretisation methods and timestepping methods for the incompressible Euler equations.

## Overview
The goal of this work is to implement higher order hybridisable DG discretisations in space and time for the incompressible Euler equations similar to what has been done in [Ueckermann and Lermusiaux (2016)](https://www.sciencedirect.com/science/article/pii/S0021999115007688?casa_token=aQP8a2IuX-MAAAAA:7KlvJnlSAoFO229d61uDrHxbyoJiYnoeE7laDV0pfrGDENnq4cmYVRGkXLTZuZnbmkX19hF_lQ) for the incompressible Navier Stokes equations. This is achieved by extending the methods in [Guzmán,Shu,Sequeira (2017)](https://academic.oup.com/imajna/article/37/4/1733/2670304?login=false).


### Mathematical details
For further details on the used discretisation and timestepping methods see

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