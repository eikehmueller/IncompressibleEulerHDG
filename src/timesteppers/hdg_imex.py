# pylint: disable=wildcard-import,unused-wildcard-import

from abc import abstractmethod
import tqdm

from firedrake import *
from timesteppers.common import IncompressibleEuler

__all__ = [
    "IncompressibleEulerHDGIMEX",
    "IncompressibleEulerHDGEuler",
    "IncompressibleEulerHDGARS232",
]


class IncompressibleEulerHDGIMEX(IncompressibleEuler):
    """Abstract base class for IMEX timesteppers of incompressible Euler equation

    At each stage, the update is either done fully implicitly or with a Richardson iteration
    that is preconditioned by a two-stage update defined by a projection method.
    """

    def __init__(
        self, mesh, degree, dt, flux="upwind", use_projection_method=True, label=None
    ):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg degree: polynomial degree of pressure space
        :arg dt: timestep size
        :arg flux: numerical flux to use, either "upwind" or "centered"
        :arg use_projection_method: use projection method instead of monolithic solve
        :arg label: name of timestepping method
        """
        super().__init__(mesh, degree, dt, label)
        self.flux = flux
        self.use_projection_method = use_projection_method
        assert self.flux in ["upwind", "centered"]
        # penalty parameter
        self.alpha = 1
        # stabilisation parameter
        self.tau = 1
        # number pf Richardson iterations
        self._n_richardson = 2

        # function spaces for velocity, pressure and trace variables
        self._V_Q = VectorFunctionSpace(self._mesh, "DG", self.degree + 1)
        self._V_p = FunctionSpace(self._mesh, "DG", self.degree)
        self._V_trace = FunctionSpace(self._mesh, "DGT", self.degree)
        self._V = self._V_Q * self._V_p * self._V_trace

        # state variable: (Q_i,p_i,lambda_i) at each stage i=0,1,...,s-1
        self._stage_state = []
        # BDM projected velocity Q*_i for stage i=0,1,...,s-2
        self._Qstar = []
        for k in range(self.nstages):
            self._stage_state.append(Function(self._V))
            if k < self.nstages - 1:
                self._Qstar.append(Function(self._V_Q))

        # solver parameters
        self._tentative_velocity_solver_parameters = {
            "ksp_type": "gmres",
            "pc_type": "ilu",
        }
        if self.use_projection_method:
            self._pressure_solver_parameters = {
                "mat_type": "matfree",
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.SCPC",
                "pc_sc_eliminate_fields": "0, 1",
                "condensed_field": {
                    "mat_type": "aij",
                    "ksp_type": "cg",
                    "ksp_rtol": 1.0e-12,
                    "pc_type": "gamg",
                    "pc_mg_log": None,
                    "mg_levels": {
                        "ksp_type": "chebyshev",
                        "ksp_max_it": 2,
                        "pc_type": "bjacobi",
                        "sub_pc_type": "sor",
                    },
                },
            }
        else:
            self._pressure_solver_parameters = {
                "ksp_type": "gmres",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }

    @property
    def label(self):
        """Label of method"""
        return self._label

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
            - self.alpha
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

    def _residual(self, w, f_rhs, i, tn):
        """Compute residual r_i(w) at stage i

        :arg w: velocity test function
        :arg f_rhs: function which returns a function for a given time
        :arg i: stage index, must be in range 1,2,...,s-1
        :arg tn: time at beginning of timestep
        """
        assert (0 < i) and (i < self.nstages)
        Q_n, _, __ = split(self._stage_state[0])
        r_form = inner(Q_n, w) * dx
        # implicit contributions
        for j in range(1, i):
            if self._a_impl[i, j] != 0:
                Q_j, _, __ = split(self._stage_state[j])
                r_form += Constant(self._a_impl[i, j] / self._a_impl[j, j]) * (
                    inner(Q_j, w) * dx - self._residual(w, f_rhs, j, tn)
                )
        # explicit contributions
        for j in range(i):
            if self._a_expl[i, j] != 0:
                r_form += (
                    Constant(self._dt * self._a_expl[i, j])
                    * inner(
                        w,
                        Function(self._V_Q).interpolate(
                            f_rhs(Constant(tn + self._c_expl[j] * self._dt))
                        ),
                    )
                    * dx
                )
        return r_form

    def _final_residual(self, w, f_rhs, tn):
        """Compute final residual r^{n+1} in each timestep

        :arg w: velocity test function
        :arg f_rhs: function which returns a function for a given time
        :arg tn: time at beginning of timestep
        """
        Q_n, _, __ = split(self._stage_state[0])
        r_form = inner(Q_n, w) * dx
        # implicit contributions
        for i in range(1, self.nstages):
            if self._b_impl[i] != 0:
                Q_i, _, __ = split(self._stage_state[i])
                r_form += Constant(self._b_impl[i] / self._a_impl[i, i]) * (
                    inner(Q_i, w) * dx - self._residual(w, f_rhs, i, tn)
                )
        # explicit contributions
        for i in range(self.nstages):
            if self._b_expl[i] != 0:
                r_form += (
                    Constant(self._dt * self._b_expl[i])
                    * inner(
                        w,
                        Function(self._V_Q).interpolate(
                            f_rhs(Constant(tn + self._c_expl[i] * self._dt))
                        ),
                    )
                    * dx
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
        p_shift = assemble(state.subfunctions[1] * dx)
        state.subfunctions[1].assign(state.subfunctions[1] - p_shift)
        state.subfunctions[2].assign(state.subfunctions[2] - p_shift)

    def solve(self, Q_initial, p_initial, f_rhs, T_final):
        """Propagate solution forward in time for a given initial velocity and pressure

        The solution is computed to the final time to T_final with nt timesteps; returns
        the final velocity and pressure

        :arg Q_initial: initial velocity, provided as an expression
        :arg p_initial: initial pressure, provided as an expression
        :arg f_rhs: function which returns an expression for a given time
        :arg T_final: final time
        """
        nt = int(T_final / self._dt)  # number of timesteps
        assert nt * self._dt - T_final < 1.0e-12  # check that dt divides the final time
        current_state = Function(self._V)
        Q_0 = Function(self._V_Q).interpolate(Q_initial)
        p_0 = Function(self._V_p).interpolate(p_initial)
        p_0 -= assemble(p_0 * dx)
        Q_i = Function(self._V_Q)
        p_i = Function(self._V_p)
        lambda_i = Function(self._V_trace)
        current_state.subfunctions[0].assign(Q_0)
        current_state.subfunctions[1].assign(p_0)
        self._reconstruct_trace(current_state)
        u_Q = TrialFunction(self._V_Q)
        w_Q = TestFunction(self._V_Q)
        u, phi, lmbda = TrialFunctions(self._V)
        w, psi, mu = TestFunctions(self._V)
        z = Function(self._V)
        z.subfunctions[0].interpolate(as_vector([Constant(0), Constant(0)]))
        z.subfunctions[1].interpolate(Constant(1))
        z.subfunctions[2].interpolate(Constant(1))
        nullspace = VectorSpaceBasis([z])
        nullspace.orthonormalize()
        # loop over all timesteps
        for n in tqdm.tqdm(range(nt)):
            self._stage_state[0].assign(current_state)
            tn = n * self._dt  # current time
            # loop over stages 1,2,...,s-1
            for i in range(1, self.nstages):
                # compute Q*_{i-1}
                self._Qstar[i - 1].assign(
                    self.project_bdm(self._stage_state[i - 1].subfunctions[0])
                )
                if self.use_projection_method:
                    rhs_i = self._residual(w_Q, f_rhs, i, tn)
                    Q_i.assign(self._stage_state[i].subfunctions[0])
                    p_i.assign(self._stage_state[i].subfunctions[1])
                    lambda_i.assign(self._stage_state[i].subfunctions[2])
                    # Richardson iteration
                    for _ in range(self._n_richardson):
                        # Compute residual
                        b_rhs_tentative = (
                            rhs_i
                            - inner(w_Q, Q_i) * dx
                            + Constant(self._a_impl[i, i] * self._dt)
                            * (
                                self._f_impl(w_Q, Q_i, self._Qstar[i - 1])
                                + self._pressure_gradient(w_Q, p_i, lambda_i)
                            )
                        )
                        # step 1: compute tentative velocity
                        a_tentative = inner(u_Q, w_Q) * dx - Constant(
                            self._a_impl[i, i] * self._dt
                        ) * self._f_impl(w_Q, u_Q, self._Qstar[i - 1])
                        Q_tentative = Function(self._V_Q)
                        solve(
                            a_tentative == b_rhs_tentative,
                            Q_tentative,
                            solver_parameters=self._tentative_velocity_solver_parameters,
                        )
                        # step 2: compute (hybridised) pressure and velocity increment
                        a_mixed_poisson = (
                            inner(w, u) * dx
                            - self._pressure_gradient(w, phi, lmbda)
                            + self._Gamma(psi, mu, u, phi, lmbda)
                        )
                        n_ = FacetNormal(self._mesh)
                        b_rhs_mixed_poisson = Constant(
                            -1 / (self._a_impl[i, i] * self._dt)
                        ) * (
                            psi * div(Q_tentative) * dx
                            - 2 * avg(psi * inner(n_, Q_tentative)) * dS
                            + inner(2 * avg(psi * n_), avg(Q_tentative)) * dS
                            - psi * inner(n_, Q_tentative) * ds
                        )
                        update = Function(self._V)
                        solve(
                            a_mixed_poisson == b_rhs_mixed_poisson,
                            update,
                            solver_parameters=self._pressure_solver_parameters,
                            nullspace=nullspace,
                        )
                        # step 3: update velocity at current stage
                        self._shift_pressure(update)
                        Q_i.assign(
                            assemble(
                                Q_i
                                + Q_tentative
                                + Constant(self._a_impl[i, i] * self._dt)
                                * update.subfunctions[0]
                            )
                        )
                        p_i.assign(assemble(p_i + update.subfunctions[1]))
                        lambda_i.assign(assemble(lambda_i + update.subfunctions[2]))
                    self._stage_state[i].subfunctions[0].assign(Q_i)
                    self._stage_state[i].subfunctions[1].assign(p_i)
                    self._stage_state[i].subfunctions[2].assign(lambda_i)
                else:
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
                        a_implicit == self._residual(w, f_rhs, i, tn),
                        self._stage_state[i],
                        solver_parameters=self._pressure_solver_parameters,
                        nullspace=nullspace,
                    )

            # compute velocity and (hybridised) pressure at next time step
            a_final = (
                inner(w, u) * dx
                - self._pressure_gradient(w, phi, lmbda)
                + self._Gamma(psi, mu, u, phi, lmbda)
            )
            solve(
                a_final == self._final_residual(w, f_rhs, tn),
                current_state,
            )

            # Add pressures from stages
            for idx in (1, 2):
                current_state.subfunctions[idx].assign(0)
                for i in range(1, self.nstages):
                    current_state.subfunctions[idx].assign(
                        current_state.subfunctions[idx]
                        + Constant(self._b_impl[i])
                        * self._stage_state[i].subfunctions[idx]
                    )

        return current_state.subfunctions[0], current_state.subfunctions[1]


class IncompressibleEulerHDGEuler(IncompressibleEulerHDGIMEX):
    """IMEX implementation of the first order implicit method"""

    def __init__(self, mesh, degree, dt, flux="upwind", use_projection_method=True):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg degree: polynomial degree of pressure space
        :arg dt: timestep size
        :arg flux: numerical flux to use, either "upwind" or "centered"
        :arg use_projection_method: use projection method instead of monolithic solve
        """
        super().__init__(
            mesh, degree, dt, flux, use_projection_method, label="IMEX Euler"
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


class IncompressibleEulerHDGARS232(IncompressibleEulerHDGIMEX):
    """IMEX ARS(2,3,2) timestepper for the incompressible Euler equations"""

    def __init__(self, mesh, degree, dt, flux="upwind", use_projection_method=True):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg degree: polynomial degree of pressure space
        :arg dt: timestep size
        :arg flux: numerical flux to use, either "upwind" or "centered"
        :arg use_projection_method: use projection method instead of monolithic solve
        """
        super().__init__(
            mesh, degree, dt, flux, use_projection_method, label="IMEX ARS(2,3,2)"
        )
        self.gamma = 1 - 1 / np.sqrt(2)
        self.delta = -2 / 3 * np.sqrt(2)

    @property
    def nstages(self):
        return 3

    @property
    def _a_expl(self):
        """3 x 3 matrix with explicit coefficients for intermediate stages"""
        return np.asarray(
            [[0, 0, 0], [self.gamma, 0, 0], [self.delta, 1 - self.delta, 0]]
        )

    @property
    def _a_impl(self):
        """3 x 3 matrix with implicit coefficients for intermediate stages"""
        return np.asarray(
            [[0, 0, 0], [0, self.gamma, 0], [0, 1 - self.gamma, self.gamma]]
        )

    @property
    def _b_expl(self):
        """vector of length 3 with explicit coefficients for final stage"""
        return np.asarray([0, 1 - self.gamma, self.gamma])

    @property
    def _b_impl(self):
        """vector of length 3 with implicit coefficients for final stage"""
        return np.asarray([0, 1 - self.gamma, self.gamma])

    @property
    def _c_expl(self):
        """vector of length 3 with fractional times at which explicit term is evaluated"""
        return np.asarray([0, self.gamma, 1])
