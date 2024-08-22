# pylint: disable=wildcard-import,unused-wildcard-import

import tqdm
from firedrake import *
from timesteppers.common import IncompressibleEuler

__all__ = ["IncompressibleEulerConformingImplicit"]


class IncompressibleEulerConformingImplicit(IncompressibleEuler):
    """Solver for incompressible Euler equations based on implicit conforming finite element method

    For details see Section 2.1 of Guzman et al. (2016).
    """

    def __init__(
        self, mesh, dt, flux="upwind", use_projection_method=True, callbacks=None
    ):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg dt: timestep size
        :arg flux: numerical flux (upwind or centered)
        :arg callbacks: callbacks to invoke at the end of each timestep
        """
        super().__init__(mesh, 1, dt, label="Conforming Implicit")
        self.flux = flux
        assert self.flux in ["upwind", "centered"]
        self._use_projection_method = use_projection_method
        self.callbacks = [] if callbacks is None else callbacks

        # function spaces for velocity, pressure and trace variables
        self._V_Q = FunctionSpace(self._mesh, "RT", 1)
        self._V_p = FunctionSpace(self._mesh, "DG", 0)
        self._V_q = FunctionSpace(self._mesh, "DG", 0)
        self._V = self._V_Q * self._V_p

        # mass solve (advection)
        v, phi = TrialFunctions(self._V)
        w, psi = TestFunctions(self._V)

        self._Q = Function(self._V_Q, name="velocity")
        self._p = Function(self._V_p, name="pressure")
        self._f = Function(self._V_Q)
        self._Q_p_hat = Function(self._V)
        n = FacetNormal(self._mesh)
        a_mass = inner(v, w) * dx + phi * psi * dx
        if self.flux == "upwind":
            advective_facet_flux = (
                inner(self._Q("+"), n("+"))
                * inner(self._Q("+") - self._Q("-"), avg(w))
                * dS
                - 1
                / 2
                * abs(inner(self._Q("+"), n("+")))
                * inner(jump(self._Q), jump(w))
                * dS
            )
        else:
            advective_facet_flux = (
                inner(2 * avg(inner(n, self._Q) * self._Q), avg(w)) * dS
            )

        b_rhs_mass = inner(self._Q, w) * dx + Constant(self._dt) * (
            inner(w, self._f) * dx
            + self._p * div(w) * dx
            - inner(outer(w, self._Q), grad(self._Q)) * dx
            + advective_facet_flux
        )
        bcs = [DirichletBC(self._V.sub(0), as_vector([0, 0]), "on_boundary")]

        lvp_mass = LinearVariationalProblem(a_mass, b_rhs_mass, self._Q_p_hat, bcs=bcs)
        self.lvs_mass = LinearVariationalSolver(lvp_mass)

        # mixed pressure solve

        a_mixed = inner(v, w) * dx + div(w) * phi * dx + div(v) * psi * dx
        b_rhs_mixed = (
            Constant(1 / self._dt) * div(self._Q_p_hat.subfunctions[0]) * psi * dx
        )
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

        # full solve
        self._Qp = Function(self._V)
        advective_facet_flux = (
            inner(self._Q("+"), n("+")) * inner(v("+") - v("-"), avg(w)) * dS
        )
        if self.flux == "upwind":
            advective_facet_flux -= (
                abs(inner(self._Q("+"), n("+")))
                * inner(v("+") - v("-"), w("+") - w("-"))
                * dS
            )
        a_monolithic = (
            inner(v, w) * dx
            + Constant(self._dt)
            * (
                inner(grad(self._Q), outer(v, w)) * dx
                - advective_facet_flux
                - phi * div(w) * dx
            )
            + psi * div(v) * dx
        )
        b_rhs_monolithic = (
            inner(self._Q, w) * dx + Constant(self._dt) * inner(self._f, w) * dx
        )
        lvp_monolithic = LinearVariationalProblem(
            a_monolithic, b_rhs_monolithic, self._Qp, bcs=bcs_mixed
        )
        self.lvs_monolithic = LinearVariationalSolver(
            lvp_monolithic, nullspace=nullspace
        )

    def solve(self, Q_initial, p_initial, q_initial, f_rhs, T_final, warmup=False):
        """Propagate solution forward in time for a given initial velocity and pressure

        The solution is computed to the final time to T_final with nt timesteps; returns
        the final velocity and pressure

        :arg Q_initial: initial velocity, provided as an expression
        :arg p_initial: initial pressure, provided as an expression
        :arg q_initial: initial tracer field, provided as an expression.
                        Set to none to advect no tracer
        :arg f_rhs: function which returns an expression for a given time
        :arg T_final: final time
        :arg warmup: perform warmup run (1 timestep only)
        """
        nt = self.get_timesteps(T_final, warmup)
        # Initial conditions
        self._Q.interpolate(Q_initial)
        self._p.interpolate(p_initial)
        self._p -= assemble(self._p * dx) / self.domain_volume
        if q_initial:
            q_tracer = Function(self._V_q, name="tracer").interpolate(q_initial)
            chi = TestFunction(self._V_q)
            sigma = TrialFunction(self._V_q)
            a_tracer = chi * sigma * dx
        else:
            q_tracer = None
        for callback in self.callbacks:
            callback.reset()
            callback(self._Q, self._p, 0, q_tracer=q_tracer)
        # timestepping
        for k in tqdm.tqdm(range(nt)):
            if q_tracer:
                b_tracer = chi * q_tracer * dx + Constant(
                    self._dt / 2
                ) * self._tracer_advection(chi, q_tracer, self._Q, project_onto_cg=True)

            # Stage 1: tentative velocity
            self._f.interpolate(f_rhs(Constant(k * self._dt)))
            if self._use_projection_method:
                self.lvs_mass.solve()

                # Stage 2: pressure correction
                self.lvs_mixed.solve()

                # update velocity and pressure
                self._Q.assign(
                    assemble(
                        self._Q_p_hat.subfunctions[0]
                        - self._dt * self._dQp.subfunctions[0]
                    )
                )
                self._p += self._dQp.subfunctions[1]

            else:
                # Do monolithic solve
                self.lvs_monolithic.solve()
                self._Q.assign(self._Qp.subfunctions[0])
                self._p.assign(self._Qp.subfunctions[1])
            self._p -= assemble(self._p * dx) / self.domain_volume
            if q_tracer:
                b_tracer += Constant(self._dt / 2) * self._tracer_advection(
                    chi, q_tracer, self._Q, project_onto_cg=True
                )
                solve(a_tracer == b_tracer, q_tracer)
            for callback in self.callbacks:
                callback(self._Q, self._p, (k + 1) * self._dt, q_tracer=q_tracer)
        return self._Q, self._p
