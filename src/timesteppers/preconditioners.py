"""Preconditioner for timestep update based on the projection method"""

from firedrake import *

__all__ = ["PCProjection"]

class PCProjection(PCBase):
    """Preconditioner for timestep update based on the projection method

    The problem to be solved for (Q,p,lambda) and a given RHS is:

        (Q,w)_Omega - dt * (f^{im}(w,Q,Q*)+g(w,p,lambda)) = r(w)

            subject to the incompressibility constraint

        Gamma(psi,mu,Q,p,lambda) = 0

    for all test functions w,psi,mu.

    To solve this approximately, proceed in three steps:

    1. Compute Q_tentative by solving

        (Q_tentative,w)_Omega - dt * f^{im}(w,Q_tentative,Q*) = r(w)

    for all test functions w. Note that the form on the LHS corresponds to the
    upper left corner of the full problem.

    2. Given Q_tentative, compute dQ, p, lambda by solving

        (dQ,w)_Omega - g(w,p,lambda) = -1/dt * Div(w,Q_tentative)

            subject to the incompressibility constraint

        Gamma(psi,mu,dQ,p,lambda) = 0

    3. Return (Q_tentative + dt * dQ, p, lambda)

    """

    def initialize(self, pc):
        """Initialise new instance

        :arg pc: PETSc preconditioner object
        """
        prefix = pc.getOptionsPrefix() + "projection_"
        _, P = pc.getOperators()
        self.ctx = P.getPythonContext()
        if not isinstance(self.ctx, matrix_free.operators.ImplicitMatrixContext):
            raise ValueError("The python context must be an ImplicitMatrixContext")
        test, _ = self.ctx.a.arguments()
        V = test.function_space()
        self._mesh = V.mesh()
        self._ises = V._ises
        Pmat = assemble(self.ctx.a).petscmat
        # Extract matrix for tentative velocity solve, this is nothing but
        # the upper left block of the entire system
        A_QQ = Pmat.createSubMatrix(self._ises[0], self._ises[0])
        velocity_ksp = PETSc.KSP().create(comm=pc.comm)
        velocity_ksp.incrementTabLevel(1, parent=pc)
        velocity_ksp.setOptionsPrefix(prefix + "velocity_")
        velocity_ksp.setOperators(A_QQ, A_QQ)
        velocity_ksp.setFromOptions()
        self._velocity_ksp = velocity_ksp
        # timestep size and HDG penalty parameter
        self.dt = PETSc.Options().getReal(prefix + "dt", 1.0)
        self.tau = PETSc.Options().getReal(prefix + "tau", 1.0)
        V_Q = V.sub(0)
        self.Q_tentative = Function(V_Q)
        w, psi, mu = TestFunctions(V)
        u, p, lmbda = TrialFunctions(V)
        # weak form for mixed pressure solve
        a_mixed = (
            inner(w, u) * dx
            - self._pressure_gradient(w, p, lmbda)
            + self._Gamma(psi, mu, u, p, lmbda)
        )

        b_mixed = Constant(1 / self.dt) * self._weak_divergence(psi, self.Q_tentative)
        self.update = Function(V)
        lvp_pressure = LinearVariationalProblem(a_mixed, b_mixed, self.update)
        # Construct nullspace
        z = Function(V)
        z.subfunctions[0].interpolate(as_vector([Constant(0), Constant(0)]))
        z.subfunctions[1].interpolate(Constant(1))
        z.subfunctions[2].interpolate(Constant(1))
        nullspace = VectorSpaceBasis([z])
        nullspace.orthonormalize()
        # Solver for mixed pressure problem
        self.lvs_pressure = LinearVariationalSolver(
            lvp_pressure, options_prefix=prefix + "pressure_", nullspace=nullspace
        )
        # Size of domain (required for normalisation)
        V_DG0 = FunctionSpace(self._mesh, "DG", 0)
        self.domain_volume = assemble(Function(V_DG0).interpolate(1) * dx)

    def apply(self, pc, x, y):
        """Apply the preconditioner.

        :arg pc: a Preconditioner instance.
        :arg x: A PETSc vector containing the incoming right-hand side.
        :arg y: A PETSc vector for the result.
        """
        r_Q = x.getSubVector(self._ises[0])
        Q = y.getSubVector(self._ises[0])
        p = y.getSubVector(self._ises[1])
        lmbda = y.getSubVector(self._ises[2])
        # Step 1: compute tentative velocity
        self._velocity_ksp.solve(r_Q, Q)
        self.Q_tentative.dat.data[:] = Q[:]
        self.lvs_pressure.solve()
        p_shift = assemble(self.update.subfunctions[1] * dx) / self.domain_volume
        self.update.subfunctions[1].assign(self.update.subfunctions[1] - p_shift)
        self.update.subfunctions[2].assign(self.update.subfunctions[2] - p_shift)
        Q[:] = (
            self.Q_tentative.dat.data + self.dt * self.update.subfunctions[0].dat.data
        )[:]
        p[:] = self.update.subfunctions[1].dat.data[:]
        lmbda[:] = self.update.subfunctions[2].dat.data[:]
        y.restoreSubVector(self._ises[0], Q)
        y.restoreSubVector(self._ises[1], p)
        y.restoreSubVector(self._ises[2], lmbda)

    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        """Apply the transpose of the preconditioner

        :arg pc: a Preconditioner instance.
        :arg x: A PETSc vector containing the incoming right-hand side.
        :arg y: A PETSc vector for the result.

        """
        raise NotImplementedError("Transpose application is not implemented.")

    def view(self, pc, viewer=None):
        """Viewer KSPs of tentative-velocity and mixed pressure system"""
        super().view(pc, viewer)
        viewer.printfASCII(f"KSP solver for tentative velocity:\n")
        self._velocity_ksp.view(viewer)
        viewer.printfASCII(f"KSP solver for mixed pressure:\n")
        self.lvs_pressure.snes.getKSP().view(viewer)

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