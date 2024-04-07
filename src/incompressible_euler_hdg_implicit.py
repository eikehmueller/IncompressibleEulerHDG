import tqdm
import numpy as np
from firedrake import *
from incompressible_euler import IncompressibleEuler


class IncompressibleEulerHDGImplicit(IncompressibleEuler):
    """Solver for incompressible Euler equations based on HDG splitting method"""

    def __init__(self, mesh, degree, dt, flux="upwind", use_projection_method=True):
        """Initialise new instance

        :arg mesh: underlying mesh
        :arg degree: polynomial degree of pressure space
        :arg dt: timestep size
        :arg flux: numerical flux to use, either "upwind" or "centered"
        :arg use_projection_method: use projection method instead of monolithic solve
        """
        super().__init__(mesh, degree, dt, label="HDGImplicit")
        self.flux = flux
        assert self.flux in ["upwind", "centered"]
        self.use_projection_method = use_projection_method
        # penalty parameter
        self.alpha = 1
        # stabilisation parameter
        self.tau = 1

        # function spaces for velocity, pressure and trace variables
        self._V_Q = VectorFunctionSpace(self._mesh, "DG", self.degree + 1)
        self._V_p = FunctionSpace(self._mesh, "DG", self.degree)
        self._V_trace = FunctionSpace(self._mesh, "DGT", self.degree)
        self._V = self._V_Q * self._V_p * self._V_trace

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

        u, phi, lmbda = TrialFunctions(self._V)
        w, psi, mu = TestFunctions(self._V)

        u_Q = TrialFunction(self._V_Q)
        w_Q = TestFunction(self._V_Q)

        # Initial conditions
        Q = Function(self._V_Q).interpolate(Q_initial)
        p = Function(self._V_p).interpolate(p_initial)
        p -= assemble(p * dx)

        n = FacetNormal(self._mesh)
        # timestepping
        for k in tqdm.tqdm(range(nt)):
            # Step 1: Compute H(div)-conforming advection velocity
            Q_star = self.project_bdm(Q)
            Q_p_trace = Function(self._V)
            f = Function(self._V_Q).interpolate(f_rhs(Constant(k * self._dt)))
            if self.use_projection_method:
                # Step 2-a: Compute tentative velocity
                a_tentative = inner(u_Q, w_Q) * dx + self._dt * (
                    inner(outer(w_Q, Q_star), grad(u_Q)) * dx
                    - inner(Q_star("+"), n("+"))
                    * inner(u_Q("+") - u_Q("-"), avg(w_Q))
                    * dS
                    + Constant(self.alpha)
                    * (
                        4
                        * avg(self._hF_inv)
                        * avg(inner(u_Q, n))
                        * avg(inner(w_Q, n))
                        * dS
                        + self._hF_inv * inner(u_Q, n) * inner(w_Q, n) * ds
                    )
                )

                if self.flux == "upwind":
                    a_tentative += (
                        self._dt
                        * abs(inner(Q_star("+"), n("+")))
                        * inner(u_Q("+") - u_Q("-"), w_Q("+") - w_Q("-"))
                        * dS
                    )
                b_rhs_tentative = inner(Q, w_Q) * dx + self._dt * (inner(f, w_Q)) * dx
                Q_tentative = Function(self._V_Q)

                solve(a_tentative == b_rhs_tentative, Q_tentative)

                # Step 2-b: Compute pressure correction

                a_poisson = (
                    inner(u, w) * dx
                    - phi * div(w) * dx
                    + 2 * avg(inner(w, n) * lmbda) * dS
                    + inner(w, n) * lmbda * ds
                    + psi * div(u) * dx
                    + 2 * avg(self.tau * (phi - lmbda) * psi) * dS
                    + self.tau * (phi - lmbda) * psi * ds
                    + 2 * avg((inner(u, n) + self.tau * (phi - lmbda)) * mu) * dS
                    + (inner(u, n) + self.tau * (phi - lmbda)) * mu * ds
                )

                b_rhs_poisson = -1 / self._dt * psi * div(Q_tentative) * dx
                solve(a_poisson == b_rhs_poisson, Q_p_trace)

                # Step 2-c: update velocity

                Q = assemble(Q_tentative + self._dt * Q_p_trace.sub(0))
            else:
                # Step 2: Compute velocity and pressure at next timestep
                a_momentum = inner(u, w) * dx + self._dt * (
                    inner(outer(w, Q_star), grad(u)) * dx
                    - inner(Q_star("+"), n("+")) * inner(u("+") - u("-"), avg(w)) * dS
                    + self.alpha
                    * (
                        4 * avg(self._hF_inv) * avg(inner(u, n)) * avg(inner(w, n)) * dS
                        + self._hF_inv * inner(u, n) * inner(w, n) * ds
                    )
                    - phi * div(w) * dx
                    + 2 * avg(inner(w, n) * lmbda) * dS
                    + inner(w, n) * lmbda * ds
                )
                if self.flux == "upwind":
                    a_momentum += (
                        self._dt
                        * abs(inner(Q_star("+"), n("+")))
                        * inner(u("+") - u("-"), w("+") - w("-"))
                        * dS
                    )
                a_continuity = (
                    psi * div(u) * dx
                    + 2 * avg(self.tau * (phi - lmbda) * psi) * dS
                    + self.tau * (phi - lmbda) * psi * ds
                )
                a_flux = (
                    2 * avg(inner(u, n) + self.tau * (phi - lmbda)) * mu("+") * dS
                    + (inner(u, n) + self.tau * (phi - lmbda)) * mu * ds
                )

                b_rhs = inner(Q, w) * dx + self._dt * inner(f, w) * dx
                a = a_momentum + a_flux + a_continuity

                solve(a == b_rhs, Q_p_trace)
                Q = Q_p_trace.sub(0)

            # Step 3: update pressure
            p = Q_p_trace.sub(1)
            p -= assemble(p * dx)
        return Q, p
