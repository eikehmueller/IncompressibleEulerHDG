"""Collection of model problems"""

from abc import ABC, abstractmethod
from firedrake import *

__all__ = ["TaylorGreen"]


class ModelProblem(ABC):
    """Abstract base class for model problems"""

    def __init__(self, V_Q, V_p):
        """Initialise new instance

        :arg V_Q: velocity function space
        :arg V_p: pressure function space
        """
        self.V_Q = V_Q
        self.V_p = V_p

    @abstractmethod
    def initial_condition(self):
        """Return initial condition"""

    @abstractmethod
    def f_rhs(self):
        """Return expression for right hand side forcing"""

    @abstractmethod
    def solution(self, t):
        """Return the solution at some finite time t

        :arg t: time at which to evaluate the solution
        """


class TaylorGreen(ModelProblem):
    """Taylor Green vortex"""

    def __init__(self, V_Q, V_p, forcing="exponential", kappa=0.5):
        """Initialise new instance

        :arg V_Q: velocity function space
        :arg V_p: pressure function space
        :arg forcing: forcing, needs to be "exponential" or "constant"
        :arg kappa: decay parameter
        """
        super().__init__(V_Q, V_p)
        self.kappa = kappa
        assert forcing in (
            "exponential",
            "constant",
        ), "Forcing must be 'constant' or 'exponential'"
        self.forcing = forcing
        x, y = SpatialCoordinate(self.V_Q.mesh())
        self.Q_stationary = as_vector(
            [
                -cos((x - 1 / 2) * pi) * sin((y - 1 / 2) * pi),
                sin((x - 1 / 2) * pi) * cos((y - 1 / 2) * pi),
            ]
        )
        self.p_stationary = (
            sin((x - 1 / 2) * pi) ** 2 + sin((y - 1 / 2) * pi) ** 2
        ) / 2

    def initial_condition(self):
        """Return initial condition"""
        return self.Q_stationary, self.p_stationary

    def f_rhs(self):
        """Return expression for right hand side forcing"""
        if self.kappa == 0:
            f_rhs = 0
        else:
            if self.forcing == "exponential":
                f_rhs = lambda t: -self.kappa * exp(-self.kappa * t) * self.Q_stationary
            elif self.forcing == "constant":
                f_rhs = lambda t: -self.kappa * self.Q_stationary
        return f_rhs

    def solution(self, t):
        """Return the solution at some finite time t

        :arg t: time at which to evaluate the solution
        """
        if self.forcing == "exponential":
            Q_exact = assemble(
                exp(-self.kappa * t) * Function(self.V_Q).interpolate(self.Q_stationary)
            )
            p_exact = assemble(
                exp(-2 * self.kappa * t)
                * Function(self.V_p).interpolate(self.p_stationary)
            )
        elif self.forcing == "constant":
            Q_exact = assemble(
                (1.0 - self.kappa * t)
                * Function(self.V_Q).interpolate(self.Q_stationary)
            )
            p_exact = assemble(
                (1 - self.kappa * t) ** 2
                * Function(self.V_p).interpolate(self.p_stationary)
            )
        p_exact -= assemble(p_exact * dx)
        return Q_exact, p_exact
