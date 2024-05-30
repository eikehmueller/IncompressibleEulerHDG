"""
Utility classes
"""

import numpy as np
from firedrake import *

__all__ = ["Averager", "gridspacing"]


class Averager:
    """Compute running average

    Computes the average S_n = 1/n sum_{j=1,n} x_n of n numbers on the fly
    """

    def __init__(self):
        """Initialise new instance"""
        self.reset()

    @property
    def value(self):
        """Return current value of average"""
        return self._average

    @property
    def n_samples(self):
        """Return number of processed samples since last reset"""
        return self._n_samples

    def update(self, x):
        """Include another number in the average

        :arg x: number to include
        """
        self._n_samples += 1
        self._average += (x - self._average) / self._n_samples

    def reset(self):
        """Reset the averager"""
        self._n_samples = 0
        self._average = 0

    def __repr__(self):
        """Internal string representation"""
        return f"{self.value} (averaged over {self.n_samples} samples)"


def gridspacing(mesh):
    """Compute smallest and largest edge length of 2d mesh

    :arg mesh: given mesh
    """

    # construct field for 1/h_F on facets
    V_coord = FunctionSpace(mesh, "DGT", 1)
    V_h = FunctionSpace(mesh, "DGT", 0)
    x, y = SpatialCoordinate(mesh)
    coords_x = Function(V_coord).interpolate(x)
    coords_y = Function(V_coord).interpolate(y)
    hF = Function(V_h)
    domain = "{[i]: 0 <= i < A.dofs}"
    instructions = """
    for i
        A[i] = sqrt((Bx[2*i]-Bx[2*i+1])*(Bx[2*i]-Bx[2*i+1]) + (By[2*i]-By[2*i+1])*(By[2*i]-By[2*i+1]))
    end
    """
    par_loop(
        (domain, instructions),
        dx,
        {
            "A": (hF, WRITE),
            "Bx": (coords_x, READ),
            "By": (coords_y, READ),
        },
    )
    h_min = np.min(hF.dat.data)
    h_max = np.max(hF.dat.data)
    return h_min, h_max
