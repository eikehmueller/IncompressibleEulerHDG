# pylint: disable=wildcard-import,unused-wildcard-import

import tqdm
from firedrake import *
from timesteppers.common import IncompressibleEuler

__all__ = ["IncompressibleEulerDGImplicit"]


class IncompressibleEulerDGImplicit(IncompressibleEuler):
    """Solver for incompressible Euler equations based on implicit DG method

    For details see Section 2.2 of Guzman et al. (2016).
    """

    def __init__(self, mesh, degree, dt, callbacks=None):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg degree: polynomial degree of pressure space
        :arg dt: timestep size
        :arg callbacks: callbacks to invoke at the end of each timestep
        """
        super().__init__(mesh, degree, dt, label="DG Implicit")
        # penalty parameter
        self.alpha = 1
        self.callbacks = [] if callbacks is None else callbacks

        # function spaces for velocity, pressure and trace variables
        self._V_Q = VectorFunctionSpace(self._mesh, "DG", self.degree + 1)
        self._V_p = FunctionSpace(self._mesh, "DG", self.degree)
        self._V = self._V_Q * self._V_p

    def solve(self, Q_initial, p_initial, f_rhs, T_final, warmup=False):
        """Propagate solution forward in time for a given initial velocity and pressure

        The solution is computed to the final time to T_final with nt timesteps; returns
        the final velocity and pressure

        :arg Q_initial: initial velocity, provided as an expression
        :arg p_initial: initial pressure, provided as an expression
        :arg f_rhs: function which returns an expression for a given time
        :arg T_final: final time
        :arg warmup: perform warmup run (1 timestep only)
        """
        nt = self.get_timesteps(T_final, warmup)
        # Initial conditions
        Q = Function(self._V_Q, name="velocity").interpolate(Q_initial)
        p = Function(self._V_p, name="pressure").interpolate(p_initial)
        p -= assemble(p * dx)

        n = FacetNormal(self._mesh)
        v, phi = TrialFunctions(self._V)
        w, psi = TestFunctions(self._V)
        for callback in self.callbacks:
            callback.reset()
            callback(Q, p, 0)

        # timestepping
        for k in tqdm.tqdm(range(nt)):
            n = FacetNormal(self._mesh)
            # Star-velocity as projection to BDM
            Q_star = self.project_bdm(Q)

            # Assemble weak form for the n+1's step
            # Eq. (3.16) in Guzman et al. (2016), first equation
            momentum_eq_lhs = inner(v, w) * dx + self._dt * (
                inner(outer(w, Q_star), grad(v)) * dx
                + self.alpha
                * (
                    2.0
                    * avg(self._hF_inv)
                    * avg(inner(v, n))
                    * 2.0
                    * avg(inner(w, n))
                    * dS
                    + self._hF_inv * inner(v, n) * inner(w, n) * ds
                )
                - inner(Q_star("+"), n("+")) * inner(jump(v), avg(w)) * dS
                - phi * div(w) * dx
                + 2.0 * avg(inner(w, n)) * avg(phi) * dS
                + inner(n, w) * phi * ds
            )

            continuity_eq_lhs = (
                psi * div(v) * dx
                - 2 * avg(inner(v, n)) * avg(psi) * dS
                - inner(v, n) * psi * ds
            )

            lhs = momentum_eq_lhs + continuity_eq_lhs

            f = Function(self._V_Q).interpolate(f_rhs(Constant(k * self._dt)))
            rhs = inner(Q, w) * dx + self._dt * inner(w, f) * dx

            # Solve
            Q_p = Function(self._V)
            solve(lhs == rhs, Q_p)

            Q.assign(Q_p.subfunctions[0])
            p.assign(Q_p.subfunctions[1])
            p -= assemble(p * dx)
            for callback in self.callbacks:
                callback(Q, p, (k + 1) * self._dt)

        return Q, p
