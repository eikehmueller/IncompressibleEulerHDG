"""Common functionality shared by all timesteppers for the incompressible Euler equations

All timesteppers are expected to subclass the abstract subclass provided in this file.
"""

# pylint: disable=wildcard-import,unused-wildcard-import

from abc import ABC, abstractmethod
from firedrake import *
import finat

__all__ = ["IncompressibleEuler", "Averager"]


class IncompressibleEuler(ABC):
    """Abstract base class for timesteppers of incompressible Euler equation

    Provides common functionality such as projection to the BDM subspace and an interface
    for the solve method. Constructs a fields that stored 1/h_F on the trace space for the
    stabilisation terms.
    """

    def __init__(self, mesh, degree, dt, label=None):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg degree: polynomial degree of pressure space
        :arg dt: timestep size
        :arg label: name of timestepping method
        """
        self._mesh = mesh
        self.degree = degree
        self._dt = dt
        self._label = label

        # function spaces for H(div) velocity projection
        el_trace = finat.ufl.BrokenElement(
            finat.ufl.FiniteElement(
                "DGT", cell=self._mesh.ufl_cell(), degree=self.degree + 1
            )
        )
        el_nedelec = finat.ufl.BrokenElement(
            finat.ufl.FiniteElement(
                "N1curl", cell=self._mesh.ufl_cell(), degree=self.degree
            )
        )
        V_DG = VectorFunctionSpace(self._mesh, "DG", self.degree + 1)
        V_nedelec = FunctionSpace(self._mesh, el_nedelec)
        V_broken_trace = FunctionSpace(self._mesh, el_trace)
        self._W = V_DG * V_nedelec * V_broken_trace

        # construct field for 1/h_F on facets
        V_coord = FunctionSpace(self._mesh, "DGT", 1)
        V_hinf = FunctionSpace(self._mesh, "DGT", 0)
        x, y = SpatialCoordinate(self._mesh)
        coords_x = Function(V_coord).interpolate(x)
        coords_y = Function(V_coord).interpolate(y)
        self._hF_inv = Function(V_hinf)
        domain = "{[i]: 0 <= i < A.dofs}"
        instructions = """
        for i
            A[i] = 1/sqrt((Bx[2*i]-Bx[2*i+1])*(Bx[2*i]-Bx[2*i+1]) + (By[2*i]-By[2*i+1])*(By[2*i]-By[2*i+1]))
        end
        """
        par_loop(
            (domain, instructions),
            dx,
            {
                "A": (self._hF_inv, WRITE),
                "Bx": (coords_x, READ),
                "By": (coords_y, READ),
            },
        )

    @property
    def label(self):
        """Label of method"""
        return self._label

    def project_bdm(self, Q):
        """Project velocity from DG space to BDM space.

        After the projection the resulting velocity Q* has continuous normals

        :arg Q: velocity to project
        """
        u, gamma, omega = TrialFunctions(self._W)
        w, sigma, q = TestFunctions(self._W)

        n = FacetNormal(self._mesh)
        a_hdiv_projection = (
            inner(u, sigma) * dx
            + 2 * avg(inner(u, n) * q) * dS
            + inner(u, n) * q * ds
            + inner(gamma, w) * dx
            + 2 * avg(omega * inner(n, w)) * dS
            + omega * inner(n, w) * ds
        )

        b_hdiv_projection = inner(Q, sigma) * dx + 2 * avg(inner(Q, n) * q) * dS
        projected_state = Function(self._W)
        solve(a_hdiv_projection == b_hdiv_projection, projected_state)
        return projected_state.subfunctions[0]

    @abstractmethod
    def solve(self, Q_initial, p_initial, f_rhs, T_final):
        """Propagate solution forward in time for a given initial velocity and pressure

        The solution is computed to the final time to T_final with nt timesteps; returns
        the final velocity and pressure

        :arg Q_initial: initial velocity, provided as an expression
        :arg p_initial: initial pressure, provided as an expression
        :arg f_rhs: function which returns an expression for a given time
        :arg T_final: final time
        """


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
