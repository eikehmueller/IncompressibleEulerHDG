# pylint: disable=wildcard-import,unused-wildcard-import

from abc import abstractmethod
from functools import cached_property
import tqdm

from firedrake import *
from timesteppers.common import *
from auxilliary.utils import Averager
from auxilliary.logging import PerformanceLog

__all__ = [
    "IncompressibleEulerHDGIMEX",
    "IncompressibleEulerHDGIMEXImplicit",
    "IncompressibleEulerHDGIMEXARS2_232",
    "IncompressibleEulerHDGIMEXARS3_443",
    "IncompressibleEulerHDGIMEXSSP2_332",
    "IncompressibleEulerHDGIMEXSSP3_433",
]


class IncompressibleEulerHDGIMEX(IncompressibleEuler):
    """Abstract base class for IMEX timesteppers of incompressible Euler equation

    At each stage, the update is either done fully implicitly or with a Richardson iteration
    that is preconditioned by a two-stage update defined by a projection method.
    """

    def __init__(
        self,
        mesh,
        degree,
        dt,
        flux="upwind",
        use_projection_method=True,
        label=None,
        callbacks=None,
    ):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg degree: polynomial degree of pressure space
        :arg dt: timestep size
        :arg flux: numerical flux to use, either "upwind" or "centered"
        :arg use_projection_method: use projection method instead of monolithic solve
        :arg label: name of timestepping method
        :arg callbacks: callbacks to invoke at the end of each timestep
        """
        super().__init__(mesh, degree, dt, label)
        self.flux = flux
        self.use_projection_method = use_projection_method
        assert self.flux in ["upwind", "centered"]
        # penalty parameter
        self.alpha_penalty = 1
        # stabilisation parameter
        self.tau = 1
        # number of Richardson iterations
        self.n_richardson = 2
        # callbacks class
        self.callbacks = [] if callbacks is None else callbacks

        # function spaces for velocity, pressure and trace variables
        self._V_Q = VectorFunctionSpace(self._mesh, "DG", self.degree + 1)
        self._V_p = FunctionSpace(self._mesh, "DG", self.degree)
        self._V_q = FunctionSpace(self._mesh, "DG", self.degree)
        self._V_trace = FunctionSpace(self._mesh, "DGT", self.degree)
        self._V = self._V_Q * self._V_p * self._V_trace
        self._V_BDM = FunctionSpace(self._mesh, "BDM", self.degree + 1)

        # state variable: (Q_i,p_i,lambda_i) at each stage i=0,1,...,s-1
        self._stage_state = []
        # BDM projected velocity Q*_i for stage i=0,1,...,s-2
        self._Qstar = []
        # tentative velocity
        self._Q_tentative = []
        # RHS evaluated at intermediate stages
        self._b_rhs = []
        # passive tracer
        self._q = []
        for k in range(self.nstages):
            self._stage_state.append(Function(self._V))
            self._b_rhs.append(Function(self._V_Q))
            self._q.append(Function(self._V_q))
            self._Q_tentative.append(Function(self._V_Q))
            if k < self.nstages - 1:
                self._Qstar.append(Function(self._V_BDM))

        self.niter_tentative = Averager()
        self.niter_pressure = Averager()
        self.niter_final_pressure = Averager()
        self.niter_pressure_reconstruction = Averager()

        # Set up pressure solvers

        def get_coarse_space():
            """Return coarse space, which is the lowest order conforming space P1"""
            return FunctionSpace(self._mesh, "CG", 1)

        def get_coarse_operator():
            """Return operator on coarse space which is the weak Laplace operator"""
            V_coarse = get_coarse_space()
            phi = TrialFunction(V_coarse)
            psi = TestFunction(V_coarse)
            return -inner(grad(phi), grad(psi)) * dx

        def get_coarse_op_nullspace():
            """Nullspace of coarse operator"""
            return VectorSpaceBasis(constant=True, comm=COMM_WORLD)

        # Application context that controls the GTMG preconditioner
        appctx = {
            "get_coarse_operator": get_coarse_operator,
            "get_coarse_space": get_coarse_space,
            "get_coarse_op_nullspace": get_coarse_op_nullspace,
            "interpolation_matrix": self.interpolation_matrix,
        }

        u, phi, lmbda = TrialFunctions(self._V)
        w, psi, mu = TestFunctions(self._V)

        a_mixed_poisson = (
            inner(w, u) * dx
            - self._pressure_gradient(w, phi, lmbda)
            + self._Gamma(psi, mu, u, phi, lmbda)
        )
        solver_parameters = {
            "mat_type": "matfree",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.SCPC",
            "pc_sc_eliminate_fields": "0, 1",
            "condensed_field": {
                "mat_type": "aij",
                "ksp_type": "gmres",
                "ksp_rtol": 1.0e-12,
                "pc_type": "python",
                "pc_python_type": "firedrake.GTMGPC",
                "pc_mg_log": None,
                "gt": {
                    "mat_type": "aij",
                    "mg_levels": {
                        "ksp_type": "chebyshev",
                        "pc_type": "python",
                        "pc_python_type": "firedrake.ASMStarPC",
                        "pc_star_construct_dim": 1,
                        "pc_star_patch_local_type": "additive",
                        "pc_star_patch_sub_ksp_type": "preonly",
                        "pc_star_patch_sub_pc_type": "lu",
                        "ksp_max_it": 2,
                    },
                    "mg_coarse": {
                        "ksp_type": "preonly",
                        "pc_type": "gamg",
                        "pc_mg_cycles": "v",
                        "mg_levels": {
                            "ksp_type": "chebyshev",
                            "ksp_max_it": 2,
                            "sub_pc_type": "sor",
                        },
                        "mg_coarse": {
                            "ksp_type": "chebyshev",
                            "ksp_max_it": 2,
                            "sub_pc_type": "sor",
                        },
                    },
                },
            },
        }

        self._solver_mixed_poisson = dict()
        # intermediate stages
        self._update = Function(self._V)
        self._current_state = Function(self._V)
        for i in range(1, self.nstages):
            b_rhs_mixed_poisson = Constant(
                -1 / (self._a_impl[i, i] * self._dt)
            ) * self._weak_divergence(psi, self._Q_tentative[i])
            problem_mixed_poisson = LinearVariationalProblem(
                a_mixed_poisson, b_rhs_mixed_poisson, self._update
            )
            self._solver_mixed_poisson[f"stage_{i:d}"] = LinearVariationalSolver(
                problem_mixed_poisson,
                solver_parameters=solver_parameters,
                nullspace=self.nullspace,
                appctx=appctx,
            )
        # final stage
        problem_mixed_poisson = LinearVariationalProblem(
            a_mixed_poisson, self._final_residual(w), self._current_state
        )
        self._solver_mixed_poisson[f"final_stage"] = LinearVariationalSolver(
            problem_mixed_poisson,
            solver_parameters=solver_parameters,
            nullspace=self.nullspace,
            appctx=appctx,
        )

        # pressure reconstruction
        Q_new = self._current_state.subfunctions[0]
        self._b_new = Function(self._V_Q)
        n = FacetNormal(self._mesh)
        b_rhs_pressure_reconstruction = (
            self._weak_divergence(psi, -self._b_new + dot(grad(Q_new), Q_new))
            - mu * inner(n, self._b_new) * ds
        )
        self._pressure_reconstruction = Function(self._V)
        problem_mixed_poisson = LinearVariationalProblem(
            a_mixed_poisson,
            b_rhs_pressure_reconstruction,
            self._pressure_reconstruction,
        )
        self._solver_mixed_poisson[f"pressure_reconstruction"] = (
            LinearVariationalSolver(
                problem_mixed_poisson,
                solver_parameters=solver_parameters,
                nullspace=self.nullspace,
                appctx=appctx,
            )
        )

        # tentative velocity solve
        solver_parameters = {
            "ksp_type": "gmres",
            "pc_type": "ilu",
        }
        u_Q = TrialFunction(self._V_Q)
        w_Q = TestFunction(self._V_Q)
        self._solver_tentative_velocity = dict()
        for i in range(1, self.nstages):
            a_tentative = inner(u_Q, w_Q) * dx - Constant(
                self._a_impl[i, i] * self._dt
            ) * self._f_impl(w_Q, u_Q, self._Qstar[i - 1])
            Q_i = self._stage_state[i].subfunctions[0]
            p_i = self._stage_state[i].subfunctions[1]
            lambda_i = self._stage_state[i].subfunctions[2]
            b_rhs_tentative = (
                self._residual(w_Q, i)
                - inner(w_Q, Q_i) * dx
                + Constant(self._a_impl[i, i] * self._dt)
                * (
                    self._f_impl(w_Q, Q_i, self._Qstar[i - 1])
                    + self._pressure_gradient(w_Q, p_i, lambda_i)
                )
            )
            problem_tentative = LinearVariationalProblem(
                a_tentative, b_rhs_tentative, self._Q_tentative[i]
            )

            self._solver_tentative_velocity[f"stage_{i:d}"] = LinearVariationalSolver(
                problem_tentative,
                solver_parameters=solver_parameters,
            )

    @PerformanceLog("pressure_solve")
    def pressure_solve(self, key):
        """Solve pressure correction equation

        :arg key: the particular solve to perform; can be "stage_i",
                  "final_stage" or "pressure_reconstruction"
        """
        self._solver_mixed_poisson[key].solve()
        its = (
            self._solver_mixed_poisson[key]
            .snes.getKSP()
            .getPC()
            .getPythonContext()
            .condensed_ksp.getIterationNumber()
        )
        return its

    @PerformanceLog("tentative_velocity_solve")
    def tentative_velocity_solve(self, key):
        """Compute tentative velocity

        :arg key: the particular solve to perform; needs to be of the form "stage_i",
        """
        self._solver_tentative_velocity[key].solve()
        return self._solver_tentative_velocity[key].snes.getKSP().getIterationNumber()

    @property
    @abstractmethod
    def nstages(self):
        """number of stages s"""

    @property
    @abstractmethod
    def _a_expl(self):
        """s x s matrix with explicit coefficients for intermediate stages"""

    @property
    @abstractmethod
    def _a_impl(self):
        """s x s matrix with implicit coefficients for intermediate stages"""

    @property
    @abstractmethod
    def _b_expl(self):
        """vector of length s with explicit coefficients for final stage"""

    @property
    @abstractmethod
    def _b_impl(self):
        """vector of length s with implicit coefficients for final stage"""

    @property
    @abstractmethod
    def _c_expl(self):
        """vector of length s with fractional times at which explicit term is evaluated"""

    def _f_impl(self, w, Q, Q_star):
        """implicit form f^{im}(w,Q,Q*)"""
        n = FacetNormal(self._mesh)
        a_form = (
            inner(Q_star("+"), n("+")) * inner(Q("+") - Q("-"), avg(w)) * dS
            - inner(outer(w, Q_star), grad(Q)) * dx
            - self.alpha_penalty
            * (
                4 * avg(self._hF_inv) * avg(inner(Q, n)) * avg(inner(w, n)) * dS
                + self._hF_inv * inner(Q, n) * inner(w, n) * ds
            )
        )
        if self.flux == "upwind":
            a_form -= (
                abs(inner(Q_star("+"), n("+")))
                * inner(Q("+") - Q("-"), w("+") - w("-"))
                * dS
            )
        return a_form

    def _pressure_gradient(self, w, p, lmbda):
        """pressure gradient g(w,p,lambda)"""
        n = FacetNormal(self._mesh)
        return (
            p * div(w) * dx
            - 2 * avg(inner(n, w) * lmbda) * dS
            - inner(n, w) * lmbda * ds
        )

    def _Gamma(self, psi, mu, Q, p, lmbda):
        """Constraint form Gamma(psi,mu,Q,p,lambda;tau)"""
        n = FacetNormal(self._mesh)
        return (
            psi * div(Q) * dx
            + 2 * avg(self.tau * (p - lmbda) * psi) * dS
            + self.tau * (p - lmbda) * psi * ds
            + 2 * avg((inner(Q, n) + self.tau * (p - lmbda)) * mu) * dS
            + (inner(Q, n) + self.tau * (p - lmbda)) * mu * ds
        )

    def _weak_divergence(self, psi, Q):
        """Compute weak divergence

        :arg psi: test function in pressure space
        :arg Q: velocity function to take the weak divergence of
        """
        n = FacetNormal(self._mesh)
        return (
            psi * div(Q) * dx
            - 2 * avg(psi * inner(n, Q)) * dS
            + inner(2 * avg(psi * n), avg(Q)) * dS
            - psi * inner(n, Q) * ds
        )

    def _residual(self, w, i):
        """Compute residual r_i(w) at stage i

        :arg w: velocity test function
        :arg i: stage index, must be in range 1,2,...,s-1
        """
        assert (0 < i) and (i < self.nstages)
        Q_n, _, __ = split(self._stage_state[0])
        r_form = inner(Q_n, w) * dx
        # implicit contributions
        for j in range(1, i):
            if self._a_impl[i, j] != 0:
                Q_j, _, __ = split(self._stage_state[j])
                r_form += Constant(self._a_impl[i, j] / self._a_impl[j, j]) * (
                    inner(Q_j, w) * dx - self._residual(w, j)
                )
        # explicit contributions
        for j in range(i):
            if self._a_expl[i, j] != 0:
                r_form += (
                    Constant(self._dt * self._a_expl[i, j])
                    * inner(w, self._b_rhs[j])
                    * dx
                )
        return r_form

    def _final_residual(self, w):
        """Compute final residual r^{n+1} in each timestep

        :arg w: velocity test function
        """
        Q_n, _, __ = split(self._stage_state[0])
        r_form = inner(Q_n, w) * dx
        # implicit contributions
        for i in range(1, self.nstages):
            if self._b_impl[i] != 0:
                Q_i, _, __ = split(self._stage_state[i])
                r_form += Constant(self._b_impl[i] / self._a_impl[i, i]) * (
                    inner(Q_i, w) * dx - self._residual(w, i)
                )
        # explicit contributions
        for i in range(self.nstages):
            if self._b_expl[i] != 0:
                r_form += (
                    Constant(self._dt * self._b_expl[i]) * inner(w, self._b_rhs[i]) * dx
                )
        return r_form

    def _tracer_residual(self, chi, i):
        """Compute passive tracer residual at stage i

        :arg chi: test function in tracer space
        :arg i: stage index
        """
        r_form = chi * self._q[0] * dx
        for j in range(i):
            if self._a_expl[i, j] != 0:
                r_form += Constant(
                    self._dt * self._a_expl[i, j]
                ) * self._tracer_advection(
                    chi,
                    self._q[j],
                    self._stage_state[i].subfunctions[0],
                    project_onto_cg=True,
                )
        return r_form

    def _tracer_final_residual(self, chi):
        """Compute passive tracer residual at final stage

        :arg chi: test function in tracer space
        """
        r_form = chi * self._q[0] * dx
        for i in range(self.nstages):
            if self._b_expl[i] != 0:
                r_form += Constant(self._dt * self._b_expl[i]) * self._tracer_advection(
                    chi,
                    self._q[i],
                    self._stage_state[i].subfunctions[0],
                    project_onto_cg=True,
                )
        return r_form

    def _reconstruct_trace(self, state):
        """Reconstruct trace variable

        This is required at the first timestep where we are only given the initial
        velocity and pressure.

        :arg state: state consisting of velocity, pressure and trace
        """
        Q, p, _ = state.subfunctions
        lmbda = TrialFunction(self._V_trace)
        mu = TestFunction(self._V_trace)
        n = FacetNormal(self._mesh)
        a_trace = 2 * self.tau * lmbda("+") * mu("+") * dS + self.tau * lmbda * mu * ds
        b_rhs_trace = (
            2 * avg((inner(Q, n) + self.tau * p) * mu) * dS
            + (inner(Q, n) + self.tau * p) * mu * ds
        )
        lmbda_reconstructed = Function(self._V_trace)
        solve(a_trace == b_rhs_trace, lmbda_reconstructed)
        state.subfunctions[2].assign(lmbda_reconstructed)

    def _shift_pressure(self, state):
        """Shift the pressure and trace variable such that pressure integrates to zero

        :arg state: state consisting of velocity, pressure and trace
        """
        p_shift = assemble(state.subfunctions[1] * dx) / self.domain_volume
        state.subfunctions[1].assign(state.subfunctions[1] - p_shift)
        state.subfunctions[2].assign(state.subfunctions[2] - p_shift)

    @cached_property
    def nullspace(self):
        """Nullspace for pressure solve"""
        z = Function(self._V)
        z.subfunctions[0].interpolate(as_vector([Constant(0), Constant(0)]))
        z.subfunctions[1].interpolate(Constant(1))
        z.subfunctions[2].interpolate(Constant(1))
        _nullspace = VectorSpaceBasis([z])
        _nullspace.orthonormalize()
        return _nullspace

    @cached_property
    def interpolation_matrix(self):
        """Interpolation matrix for GTMG"""
        V_coarse = FunctionSpace(self._mesh, "CG", 1)
        u = TrialFunction(self._V_trace)
        u_coarse = TrialFunction(V_coarse)
        w = TestFunction(self._V_trace)
        a_mass = u("+") * w("+") * dS + u * w * ds
        a_proj = 0.5 * avg(u_coarse) * w("+") * dS + u_coarse * w * ds
        a_proj_mat = assemble(a_proj, mat_type="aij").M.handle
        a_mass_inv = assemble(Tensor(a_mass).inv, mat_type="aij")
        a_mass_inv_mat = a_mass_inv.M.handle
        return a_mass_inv_mat.matMult(a_proj_mat)

    def test_pressure_reconstruction(self, Q, f_rhs, t):
        """Test the pressure reconstruction from velocity

        For a given velocity Q (given as an analytical expression) and forcing f_rhs (given as a function),
        reconstruct the corresponding pressure at the same time.

        :arg Q: velocity
        :arg f_rhs: right hand side forcing function
        :arg t: time at which to perform the test
        """
        self._current_state.subfunctions[0].interpolate(Q)
        self._b_new.interpolate(f_rhs(Constant(t)))
        self.pressure_solve("pressure_reconstruction")
        for idx in (1, 2):
            self._current_state.subfunctions[idx].assign(
                self._pressure_reconstruction.subfunctions[idx]
            )
        self._shift_pressure(self._current_state)
        return self._current_state.subfunctions[0], self._current_state.subfunctions[1]

    def solve(self, Q_initial, p_initial, q_initial, f_rhs, T_final, warmup=False):
        """Propagate solution forward in time for a given initial velocity and pressure

        The solution is computed to the final time to T_final with nt timesteps; returns
        the final velocity and pressure

        :arg Q_initial: initial velocity, provided as an expression
        :arg p_initial: initial pressure, provided as an expression
        :arg q_initial: initial tracer field, provided as expression
                        Set to "None" to disable tracer advection
        :arg f_rhs: function which returns an expression for a given time
        :arg T_final: final time
        :arg warmup: only perform a single timestep
        """
        nt = self.get_timesteps(T_final, warmup)
        Q_0 = Function(self._V_Q).interpolate(Q_initial)
        p_0 = Function(self._V_p).interpolate(p_initial)
        p_0 -= assemble(p_0 * dx) / self.domain_volume
        if q_initial:
            q_tracer = Function(self._V_q, name="tracer").interpolate(q_initial)
            chi = TestFunction(self._V_q)
            sigma = TrialFunction(self._V_q)
            a_tracer = chi * sigma * dx
        else:
            q_tracer = None
        self._current_state.subfunctions[0].assign(Q_0)
        self._current_state.subfunctions[0].rename("Q")
        self._current_state.subfunctions[1].assign(p_0)
        self._current_state.subfunctions[1].rename("p")
        self._reconstruct_trace(self._current_state)
        u, phi, lmbda = TrialFunctions(self._V)
        w, psi, mu = TestFunctions(self._V)
        self.niter_tentative.reset()
        self.niter_pressure.reset()
        self.niter_final_pressure.reset()
        self.niter_pressure_reconstruction.reset()
        for callback in self.callbacks:
            callback.reset()
            callback(
                self._current_state.subfunctions[0],
                self._current_state.subfunctions[1],
                0,
                q_tracer=q_tracer,
            )
        # loop over all timesteps
        for k in tqdm.tqdm(range(nt)):
            with PerformanceLog("timestep"):
                # evaluate RHS at intermediate stages
                tn = k * self._dt  # current time
                for i in range(self.nstages):
                    self._b_rhs[i].interpolate(
                        f_rhs(Constant(tn + self._c_expl[i] * self._dt))
                    )
                self._stage_state[0].assign(self._current_state)
                if q_tracer:
                    self._q[0].assign(q_tracer)
                # loop over stages 1,2,...,s-1
                for i in range(1, self.nstages):
                    # compute Q*_{i-1}
                    with PerformanceLog("bdm_projection"):
                        self._Qstar[i - 1].assign(
                            self.project_bdm(self._stage_state[i - 1].subfunctions[0])
                        )
                    if self.use_projection_method:
                        # Richardson iteration
                        for _ in range(self.n_richardson):
                            # Compute residual
                            its = self.tentative_velocity_solve(f"stage_{i:d}")
                            self.niter_tentative.update(its)
                            # step 2: compute (hybridised) pressure and velocity increment
                            its = self.pressure_solve(f"stage_{i:d}")
                            # The solution of the mixed problem is stored in self._update
                            self.niter_pressure.update(its)
                            # step 3: update velocity at current stage
                            self._shift_pressure(self._update)
                            self._stage_state[i].subfunctions[0].assign(
                                assemble(
                                    self._stage_state[i].subfunctions[0]
                                    + self._Q_tentative[i]
                                    + Constant(self._a_impl[i, i] * self._dt)
                                    * self._update.subfunctions[0]
                                )
                            )
                            self._stage_state[i].subfunctions[1].assign(
                                assemble(
                                    self._stage_state[i].subfunctions[1]
                                    + self._update.subfunctions[1]
                                )
                            )
                            self._stage_state[i].subfunctions[2].assign(
                                assemble(
                                    self._stage_state[i].subfunctions[2]
                                    + self._update.subfunctions[2]
                                )
                            )
                    else:
                        with PerformanceLog("unsplit_solve"):
                            a_implicit = (
                                inner(w, u) * dx
                                - Constant(self._a_impl[i, i] * self._dt)
                                * (
                                    self._pressure_gradient(w, phi, lmbda)
                                    + self._f_impl(w, u, self._Qstar[i - 1])
                                )
                                + self._Gamma(psi, mu, u, phi, lmbda)
                            )
                            solve(
                                a_implicit == self._residual(w, i),
                                self._stage_state[i],
                                solver_parameters={
                                    "ksp_type": "gmres",
                                    "pc_type": "lu",
                                    "pc_factor_mat_solver_type": "mumps",
                                },
                                nullspace=self.nullspace,
                            )
                    self._shift_pressure(self._stage_state[i])
                    if q_tracer:
                        solve(a_tracer == self._tracer_residual(chi, i), self._q[i])
                its = self.pressure_solve("final_stage")
                # The solution of the mixed problem is stored in self._current_state
                self.niter_final_pressure.update(its)

                # Reconstruct pressure from velocity
                self._b_new.interpolate(f_rhs(Constant(tn + self._dt)))
                its = self.pressure_solve("pressure_reconstruction")
                # The solution of the mixed problem is stored in self._pressure_reconstruction
                self.niter_pressure_reconstruction.update(its)
                for idx in (1, 2):
                    self._current_state.subfunctions[idx].assign(
                        self._pressure_reconstruction.subfunctions[idx]
                    )
                self._shift_pressure(self._current_state)
                if q_tracer:
                    solve(a_tracer == self._tracer_final_residual(chi), q_tracer)
            for callback in self.callbacks:
                callback(
                    self._current_state.subfunctions[0],
                    self._current_state.subfunctions[1],
                    tn + self._dt,
                    q_tracer=q_tracer,
                )

        print("average number of solver iterations")
        print(40 * "-")
        print(f"  tentative velocity its      : {self.niter_tentative.value:8.2f}")
        if self.use_projection_method:
            print(f"  pressure its                : {self.niter_pressure.value:8.2f}")
            print(
                f"  final pressure its          : {self.niter_final_pressure.value:8.2f}"
            )
        print(
            f"  pressure reconstruction its : {self.niter_pressure_reconstruction.value:8.2f}"
        )
        print()
        return self._current_state.subfunctions[0], self._current_state.subfunctions[1]


#######################################################################################
#       S P E C I F I C     I M E X     T I M E S T E P P E R S                       #
#######################################################################################


class IncompressibleEulerHDGIMEXImplicit(IncompressibleEulerHDGIMEX):
    """IMEX implementation of the first order implicit method"""

    def __init__(
        self,
        mesh,
        degree,
        dt,
        flux="upwind",
        use_projection_method=True,
        callbacks=None,
    ):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg degree: polynomial degree of pressure space
        :arg dt: timestep size
        :arg flux: numerical flux to use, either "upwind" or "centered"
        :arg use_projection_method: use projection method instead of monolithic solve
        :arg callbacks: callbacks to invoke at the end of each timestep
        """
        super().__init__(
            mesh,
            degree,
            dt,
            flux,
            use_projection_method,
            label="HDG IMEX Implicit",
            callbacks=callbacks,
        )

    @property
    def nstages(self):
        return 2

    @property
    def _a_expl(self):
        """2 x 2 matrix with explicit coefficients for intermediate stages"""
        return np.asarray([[0, 0], [1, 0]])

    @property
    def _a_impl(self):
        """2 x 2 matrix with implicit coefficients for intermediate stages"""
        return np.asarray([[0, 0], [0, 1]])

    @property
    def _b_expl(self):
        """vector of length 2 with explicit coefficients for final stage"""
        return np.asarray([1, 0])

    @property
    def _b_impl(self):
        """vector of length 2 with implicit coefficients for final stage"""
        return np.asarray([0, 1])

    @property
    def _c_expl(self):
        """vector of length 2 with fractional times at which explicit term is evaluated"""
        return np.asarray([0, 1])


class IncompressibleEulerHDGIMEXARS2_232(IncompressibleEulerHDGIMEX):
    """IMEX ARS2(2,3,2) timestepper for the incompressible Euler equations"""

    def __init__(
        self,
        mesh,
        degree,
        dt,
        flux="upwind",
        use_projection_method=True,
        callbacks=None,
    ):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg degree: polynomial degree of pressure space
        :arg dt: timestep size
        :arg flux: numerical flux to use, either "upwind" or "centered"
        :arg use_projection_method: use projection method instead of monolithic solve
        :arg callbacks: callbacks to invoke at the end of each timestep
        """
        super().__init__(
            mesh,
            degree,
            dt,
            flux,
            use_projection_method,
            label="HDG IMEX ARS2(2,3,2)",
            callbacks=callbacks,
        )

    @property
    def nstages(self):
        return 3

    @property
    def _a_expl(self):
        """3 x 3 matrix with explicit coefficients for intermediate stages"""
        gamma = 1 - 1 / np.sqrt(2)
        delta = -2 / 3 * np.sqrt(2)
        return np.asarray([[0, 0, 0], [gamma, 0, 0], [delta, 1 - delta, 0]])

    @property
    def _a_impl(self):
        """3 x 3 matrix with implicit coefficients for intermediate stages"""
        gamma = 1 - 1 / np.sqrt(2)
        return np.asarray([[0, 0, 0], [0, gamma, 0], [0, 1 - gamma, gamma]])

    @property
    def _b_expl(self):
        """vector of length 3 with explicit coefficients for final stage"""
        gamma = 1 - 1 / np.sqrt(2)
        return np.asarray([0, 1 - gamma, gamma])

    @property
    def _b_impl(self):
        """vector of length 3 with implicit coefficients for final stage"""
        gamma = 1 - 1 / np.sqrt(2)
        return np.asarray([0, 1 - gamma, gamma])

    @property
    def _c_expl(self):
        """vector of length 3 with fractional times at which explicit term is evaluated"""
        gamma = 1 - 1 / np.sqrt(2)
        return np.asarray([0, gamma, 1])


class IncompressibleEulerHDGIMEXARS3_443(IncompressibleEulerHDGIMEX):
    """IMEX ARS3(4,4,3) timestepper for the incompressible Euler equations"""

    def __init__(
        self,
        mesh,
        degree,
        dt,
        flux="upwind",
        use_projection_method=True,
        callbacks=None,
    ):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg degree: polynomial degree of pressure space
        :arg dt: timestep size
        :arg flux: numerical flux to use, either "upwind" or "centered"
        :arg use_projection_method: use projection method instead of monolithic solve
        :arg callbacks: callbacks to invoke at the end of each timestep
        """
        super().__init__(
            mesh,
            degree,
            dt,
            flux,
            use_projection_method,
            label="HDG IMEX ARS3(4,4,3)",
            callbacks=callbacks,
        )

    @property
    def nstages(self):
        return 5

    @property
    def _a_expl(self):
        """5 x 5 matrix with explicit coefficients for intermediate stages"""
        return np.asarray(
            [
                [0, 0, 0, 0, 0],
                [1 / 2, 0, 0, 0, 0],
                [11 / 18, 1 / 18, 0, 0, 0],
                [5 / 6, -5 / 6, 1 / 2, 0, 0],
                [1 / 4, 7 / 4, 3 / 4, -7 / 4, 0],
            ]
        )

    @property
    def _a_impl(self):
        """5 x 5 matrix with implicit coefficients for intermediate stages"""
        return np.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 1 / 2, 0, 0, 0],
                [0, 1 / 6, 1 / 2, 0, 0],
                [0, -1 / 2, 1 / 2, 1 / 2, 0],
                [0, 3 / 2, -3 / 2, 1 / 2, 1 / 2],
            ]
        )

    @property
    def _b_expl(self):
        """vector of length 5 with explicit coefficients for final stage"""
        return np.asarray([1 / 4, 7 / 4, 3 / 4, -7 / 4, 0])

    @property
    def _b_impl(self):
        """vector of length 5 with implicit coefficients for final stage"""
        return np.asarray([0, 3 / 2, -3, 2, 1 / 2, 1 / 2])

    @property
    def _c_expl(self):
        """vector of length 5 with fractional times at which explicit term is evaluated"""
        return np.asarray([0, 1 / 2, 2 / 3, 1 / 2, 1])


class IncompressibleEulerHDGIMEXSSP2_332(IncompressibleEulerHDGIMEX):
    """IMEX SSP2(3,3,2) timestepper for the incompressible Euler equations"""

    def __init__(
        self,
        mesh,
        degree,
        dt,
        flux="upwind",
        use_projection_method=True,
        callbacks=None,
    ):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg degree: polynomial degree of pressure space
        :arg dt: timestep size
        :arg flux: numerical flux to use, either "upwind" or "centered"
        :arg use_projection_method: use projection method instead of monolithic solve
        :arg callbacks: callbacks to invoke at the end of each timestep
        """
        super().__init__(
            mesh,
            degree,
            dt,
            flux,
            use_projection_method,
            label="HDG IMEX SSP2(3,3,2)",
            callbacks=callbacks,
        )

    @property
    def nstages(self):
        return 3

    @property
    def _a_expl(self):
        """3 x 3 matrix with explicit coefficients for intermediate stages"""
        return np.asarray([[0, 0, 0], [1 / 2, 0, 0], [1 / 2, 1 / 2, 0]])

    @property
    def _a_impl(self):
        """3 x 3 matrix with implicit coefficients for intermediate stages"""
        return np.asarray(
            [
                [1 / 4, 0, 0],
                [0, 1 / 4, 0],
                [1 / 3, 1 / 3, 1 / 3],
            ]
        )

    @property
    def _b_expl(self):
        """vector of length 3 with explicit coefficients for final stage"""
        return np.asarray([1 / 3, 1 / 3, 1 / 3])

    @property
    def _b_impl(self):
        """vector of length 3 with implicit coefficients for final stage"""
        return np.asarray([1 / 3, 1 / 3, 1 / 3])

    @property
    def _c_expl(self):
        """vector of length 3 with fractional times at which explicit term is evaluated"""
        return np.asarray([0, 1, 1 / 2])


class IncompressibleEulerHDGIMEXSSP3_433(IncompressibleEulerHDGIMEX):
    """IMEX SSP3(4,3,3) timestepper for the incompressible Euler equations"""

    def __init__(
        self,
        mesh,
        degree,
        dt,
        flux="upwind",
        use_projection_method=True,
        callbacks=None,
    ):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg degree: polynomial degree of pressure space
        :arg dt: timestep size
        :arg flux: numerical flux to use, either "upwind" or "centered"
        :arg use_projection_method: use projection method instead of monolithic solve
        :arg callbacks: callbacks to invoke at the end of each timestep
        """
        super().__init__(
            mesh,
            degree,
            dt,
            flux,
            use_projection_method,
            label="HDG IMEX SSP3(4,3,3)",
            callbacks=callbacks,
        )

    @property
    def nstages(self):
        return 4

    @property
    def _a_expl(self):
        """4 x 4 matrix with explicit coefficients for intermediate stages"""
        return np.asarray(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1 / 4, 1 / 4, 0],
            ]
        )

    @property
    def _a_impl(self):
        """4 x 4 matrix with implicit coefficients for intermediate stages

        Constants for a_impl are taken from from:

            Pareschi, L. and Russo, G., 2005. "Implicit–explicit Runge–Kutta schemes
            and applications to hyperbolic systems with relaxation."
            Journal of Scientific computing, 25, pp.129-155.
        """
        alpha = 0.24169426078821
        beta = 0.06042356519705
        eta = 0.12915286960590
        delta = 1 / 2 - alpha - beta - eta
        return np.asarray(
            [
                [alpha, 0, 0, 0],
                [-alpha, alpha, 0, 0],
                [0, 1 - alpha, alpha, 0],
                [beta, eta, delta, alpha],
            ]
        )

    @property
    def _b_expl(self):
        """vector of length 4 with explicit coefficients for final stage"""
        return np.asarray([0, 1 / 6, 1 / 6, 2 / 3])

    @property
    def _b_impl(self):
        """vector of length 4 with implicit coefficients for final stage"""
        return np.asarray([0, 1 / 6, 1 / 6, 2 / 3])

    @property
    def _c_expl(self):
        """vector of length 4 with fractional times at which explicit term is evaluated"""
        return np.asarray([0, 0, 1, 1 / 2])
