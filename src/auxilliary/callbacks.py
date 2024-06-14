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
        chi = TestFunction(V_vorticity)
        psi = TrialFunction(V_vorticity)
        xi = Function(V_vorticity, name="vorticity")
        b_rhs = chi * curl(self.Q_proxy) * dx
        a_project = chi * psi * dx
        lvp = LinearVariationalProblem(a_project, b_rhs, xi)
        lvs = LinearVariationalSolver(lvp)
        return lvs, xi, self.Q_proxy

    def __call__(self, Q, p, t):
        """Save velocity, pressure and vorticity fields to disk at a given time

        :arg Q: velocity field at time t
        :arg p: pressure field at time t
        :arg t: time t
        """
        lvs, xi, Q_ = self.vorticity_solver(Q)
        Q_.assign(Q)
        lvs.solve()
        self.outfile.write(Q, p, xi, time=t)
