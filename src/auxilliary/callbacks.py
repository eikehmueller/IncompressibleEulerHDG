"""Classes for performing regular callbacks during ttimestepping"""

import functools
from abc import ABC, abstractmethod
from firedrake import *
from firedrake.output import VTKFile

__all__ = ["AnimationCallback"]


class Callback(ABC):
    """Abstract base class"""

    @abstractmethod
    def __call__(self, Q, p, t):
        """Invoke the callback for particular velocity/pressure fields at a given time

        :arg Q: velocity field at time t
        :arg p: pressure field at time t
        :arg t: time t
        """

    @abstractmethod
    def reset(self):
        """Reset callback"""


class AnimationCallback(Callback):
    """Save fields to disk"""

    def __init__(self, filename):
        """Initialise new instance

        :arg filename: name of file to write fields to
        """
        self.filename = filename
        self.reset()

    def reset(self):
        """Re-open file"""
        self.outfile = VTKFile(self.filename, mode="w")

    @functools.cache
    def vorticity_solver(self, Q):
        """Return cached vorticity solver and input/output fields

        Construct linear variational solver for vorticity project. Returns this solver
        together with a handle to the field velocity that needs to be assigned to and
        a handle to the computed vorticity

        :arg Q: velocity field
        """
        self.Q_proxy = Function(Q.function_space())
        degree = Q.function_space().ufl_element().degree()
        mesh = Q.function_space().mesh()
        V_vorticity = FunctionSpace(mesh, "CG", degree)
        tau = TestFunction(V_vorticity)
        xi = TrialFunction(V_vorticity)
        omega = Function(V_vorticity, name="vorticity")
        epsilon = as_matrix([[0, +1], [-1, 0]])
        n = FacetNormal(mesh)
        b_rhs = (
            -inner(epsilon, outer(grad(tau), self.Q_proxy)) * dx
            + tau * inner(epsilon, outer(n, self.Q_proxy)) * ds
        )
        a_project = tau * xi * dx
        lvp = LinearVariationalProblem(a_project, b_rhs, omega)
        lvs = LinearVariationalSolver(lvp)
        return lvs, omega, self.Q_proxy

    def __call__(self, Q, p, t, q_tracer=None):
        """Save velocity, pressure and vorticity fields to disk at a given time

        :arg Q: velocity field at time t
        :arg p: pressure field at time t
        :arg t: time t
        :arg q_tracer: passive tracer field at time t
        """
        lvs, omega, Q_ = self.vorticity_solver(Q)
        Q_.assign(Q)
        lvs.solve()
        fields = [Q, p, omega]
        if q_tracer:
            fields.append(q_tracer)
        self.outfile.write(*fields, time=t)
