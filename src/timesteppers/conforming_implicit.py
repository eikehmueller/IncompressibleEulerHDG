# pylint: disable=wildcard-import,unused-wildcard-import

import tqdm
from firedrake import *
from timesteppers.common import IncompressibleEuler

__all__ = ["IncompressibleEulerConformingImplicit"]


class IncompressibleEulerConformingImplicit(IncompressibleEuler):
    """Solver for incompressible Euler equations based on implicit conforming finite element method

    For details see Section 2.1 of Guzman et al. (2016).
    """

    def __init__(self, mesh, dt, flux="upwind", callbacks=None):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg dt: timestep size
        :arg flux: numerical flux (upwind or centered)
        :arg callbacks: callbacks to invoke at the end of each timestep
        """
        super().__init__(mesh, 1, dt, label="Conforming Implicit")
        self.flux = flux
        assert self.flux in ["upwind", "centered"]
        self.callbacks = [] if callbacks is None else callbacks

        # function spaces for velocity, pressure and trace variables
        self._V_Q = FunctionSpace(self._mesh, "RT", 1)
        self._V_p = FunctionSpace(self._mesh, "DG", 0)
        self._V = self._V_Q * self._V_p
        self._Q_hat = Function(self._V_Q)

        v, phi = TrialFunctions(self._V)
        w, psi = TestFunctions(self._V)

        a_mixed = inner(v, w) * dx + div(w) * phi * dx + div(v) * psi * dx
        b_rhs_mixed = 1 / self._dt * div(self._Q_hat) * psi * dx
        self._dQp = Function(self._V)
        nullspace = MixedVectorSpaceBasis(
            self._V,
            [
                self._V.sub(0),
                VectorSpaceBasis(constant=True, comm=COMM_WORLD),
            ],
        )
        # homogeneous Dirichlet boundary conditions (zero normal derivative)
        bcs_mixed = [DirichletBC(self._V.sub(0), as_vector([0, 0]), "on_boundary")]
        lvp_mixed = LinearVariationalProblem(
            a_mixed, b_rhs_mixed, self._dQp, bcs=bcs_mixed
        )
        self.lvs_mixed = LinearVariationalSolver(lvp_mixed, nullspace=nullspace)

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
            # Stage 1: tentative velocity

            n = FacetNormal(self._mesh)
            a_mass = inner(v, w) * dx + phi * psi * dx
            if self.flux == "upwind":
                advective_facet_flux = (
                    inner(Q("+"), n("+")) * inner(Q("+") - Q("-"), avg(w)) * dS
                    - 1 / 2 * abs(inner(Q("+"), n("+"))) * inner(jump(Q), jump(w)) * dS
                )
            else:
                advective_facet_flux = inner(2 * avg(inner(n, Q) * Q), avg(w)) * dS

            f = Function(self._V_Q).interpolate(f_rhs(Constant(k * self._dt)))
            b_rhs_mass = inner(Q, w) * dx + self._dt * (
                inner(w, f) * dx
                + p * div(w) * dx
                - inner(outer(w, Q), grad(Q)) * dx
                + advective_facet_flux
            )

            bcs = [DirichletBC(self._V.sub(0), as_vector([0, 0]), "on_boundary")]

            Q_p_hat = Function(self._V)

            solve(a_mass == b_rhs_mass, Q_p_hat, bcs=bcs)

            # Stage 2: pressure correction

            self._Q_hat.assign(Q_p_hat.subfunctions[0])
            self.lvs_mixed.solve()

            # update velocity and pressure
            Q.assign(assemble(self._Q_hat - self._dt * self._dQp.subfunctions[0]))
            p += self._dQp.subfunctions[1]
            p -= assemble(p * dx)
            for callback in self.callbacks:
                callback(Q, p, (k + 1) * self._dt)
        return Q, p
