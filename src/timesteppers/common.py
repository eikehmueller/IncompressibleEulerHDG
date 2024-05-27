"""Common functionality shared by all timesteppers for the incompressible Euler equations

All timesteppers are expected to subclass the abstract subclass provided in this file.
"""

# pylint: disable=wildcard-import,unused-wildcard-import

from abc import ABC, abstractmethod
from firedrake import *
import finat

__all__ = ["IncompressibleEuler"]


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
        self._Q = Function(V_DG)
        b_hdiv_projection = (
            inner(self._Q, sigma) * dx + 2 * inner(avg(self._Q), avg(q * n)) * dS
        )
        self._projected_state = Function(self._W)
        problem_hdiv_projection = LinearVariationalProblem(
            a_hdiv_projection, b_hdiv_projection, self._projected_state
        )
        solver_parameters = {"ksp_type": "gmres", "pc_type": "jacobi"}
        self._solver_hdiv_projection = LinearVariationalSolver(
            problem_hdiv_projection, solver_parameters=solver_parameters
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
        self._Q.assign(Q)
        self._solver_hdiv_projection.solve()
        return self._projected_state.subfunctions[0]

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
