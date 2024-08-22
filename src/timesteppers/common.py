"""Common functionality shared by all timesteppers for the incompressible Euler equations

All timesteppers are expected to subclass the abstract subclass provided in this file.
"""

# pylint: disable=wildcard-import,unused-wildcard-import

from abc import ABC, abstractmethod
from firedrake import *
from firedrake.__future__ import interpolate

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

        self.V_BDM = FunctionSpace(self._mesh, "BDM", self.degree + 1)
        shapes = (self.V_BDM.finat_element.space_dimension(), np.prod(self.V_BDM.shape))
        domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
        instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
        self.inverse_multiplicity = Function(self.V_BDM)
        par_loop((domain, instructions), dx, {"w": (self.inverse_multiplicity, INC)})
        with self.inverse_multiplicity.dat.vec as wv:
            wv.reciprocal()
        # volume
        V_DG0 = FunctionSpace(self._mesh, "DG", 0)
        self.domain_volume = assemble(Function(V_DG0).interpolate(1) * dx)

    def get_timesteps(self, t_final, warmup):
        """Compute the number of timsteps

        :arg t_final: final time
        :arg warmup: perform a single timestep only
        """
        nt = 1 if warmup else int(t_final // self._dt)  # number of timesteps
        assert nt * self._dt - t_final < 1.0e-12  # check that dt divides the final time
        return nt

    @property
    def label(self):
        """Label of method"""
        return self._label

    def project_bdm(self, Q):
        """Project velocity from DG space to BDM space.

        After the projection the resulting velocity Q* has continuous normals.

        Original implementation of interpolation with averaging by Pablo Brubeck, Oxford.

        :arg Q: velocity to project
        """
        # Interpolate to BDM space
        Q_interpolate = assemble(interpolate(Q, self.V_BDM, access=INC))
        # Scale by multiplicity
        with Q_interpolate.dat.vec as uv, self.inverse_multiplicity.dat.vec_ro as wv:
            uv.pointwiseMult(uv, wv)
        # apply boundary conditions
        bc = DirichletBC(self.V_BDM, (0, 0), "on_boundary")
        bc.apply(Q_interpolate)
        return Q_interpolate

    def _tracer_advection(self, chi, q, u, project_onto_cg=True):
        """Passive tracer advection term

        :arg chi: test function in tracer space
        :arg q: tracer field
        :arg u: advection velocity
        :arg project_to_cg: project velocity onto CG space
        """
        n = FacetNormal(self._mesh)
        if project_onto_cg:
            degree = u.function_space().ufl_element().degree()
            V_CG = VectorFunctionSpace(self._mesh, "CG", degree)
            u_ = Function(V_CG).project(u)
        else:
            u_ = u
        un = 1 / 2 * (inner(u_, n) + abs(inner(u_, n)))
        return (
            q * div(chi * u_) * dx
            - (chi("+") - chi("-")) * (un("+") * q("+") - un("-") * q("-")) * dS
        )

    @abstractmethod
    def solve(self, Q_initial, p_initial, q_initial, f_rhs, T_final, warmup=False):
        """Propagate solution forward in time for a given initial velocity and pressure

        The solution is computed to the final time to T_final with nt timesteps; returns
        the final velocity and pressure

        :arg Q_initial: initial velocity, provided as an expression
        :arg p_initial: initial pressure, provided as an expression
        :arg q_initial: initial tracer field, provided as expression
        :arg f_rhs: function which returns an expression for a given time
        :arg T_final: final time
        :arg warmup: perform warmup iteration?
        """
