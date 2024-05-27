"""Test code for projecting a DG function to BDM^0 using two methods:

1. The method described in Section 2.2 of Guzmán, J., Shu, C.W. and Sequeira, F.A., 2017. 
    "H (div) conforming and DG methods for incompressible Euler’s equations."
    IMA Journal of Numerical Analysis, 37(4), pp.1733-1771.

2. Firedrake's native project functionality

"""

import finat
from firedrake import *
from firedrake.output import VTKFile

pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)

# Mesh size
nx = 8
# Polynomial degree
degree = 2

mesh = UnitSquareMesh(nx, nx, quadrilateral=False)

# Construct function spaces
el_trace = finat.ufl.BrokenElement(
    finat.ufl.FiniteElement("DGT", cell=mesh.ufl_cell(), degree=degree + 1)
)
el_nedelec = finat.ufl.BrokenElement(
    finat.ufl.FiniteElement("N1curl", cell=mesh.ufl_cell(), degree=degree)
)
el_trace = finat.ufl.BrokenElement(
    finat.ufl.FiniteElement("DGT", cell=mesh.ufl_cell(), degree=degree + 1)
)

el_nedelec = finat.ufl.BrokenElement(
    finat.ufl.FiniteElement("N1curl", cell=mesh.ufl_cell(), degree=degree)
)


V_DG = VectorFunctionSpace(mesh, "DG", degree + 1)
V_nedelec = FunctionSpace(mesh, el_nedelec)
V_broken_trace = FunctionSpace(mesh, el_trace)
W = V_DG * V_nedelec * V_broken_trace

# Trial- and test-functions in mixed function space
u, gamma, omega = TrialFunctions(W)
w, sigma, q = TestFunctions(W)

n = FacetNormal(mesh)
# Construct bilinear form
a = (
    inner(u, sigma) * dx
    + 2 * avg(inner(u, n) * q) * dS
    + inner(u, n) * q * ds
    + inner(gamma, w) * dx
    + 2 * avg(omega * inner(n, w)) * dS
    + omega * inner(n, w) * ds
)

# Random function to be projected
Q = rg.normal(V_DG)

# Right hand side
b_rhs = inner(Q, sigma) * dx + 2 * inner(avg(Q), avg(q * n)) * dS

solution = Function(W)

lvp = LinearVariationalProblem(a, b_rhs, solution)
solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.ASMVankaPC",
    "pc_vanka_construct_dim": 2,
    "pc_vanka_patch_sub_ksp_type": "preonly",
    "pc_vanka_patch_sub_pc_type": "lu",
    "ksp_monitor": None,
}
lvs = LinearVariationalSolver(lvp, solver_parameters=solver_parameters)
lvs.solve()
Q_projected = solution.subfunctions[0]

# Firedrake BDM projection
V_BDM = FunctionSpace(mesh, "BDM", degree + 1)
bcs = [DirichletBC(V_BDM, (0, 0), j) for j in range(1, 5)]
Q_BDM = Function(V_BDM).project(Q, bcs=bcs, solver_parameters={"ksp_monitor": None})

# Save functions to disk
Q.rename("Q")
Q_projected.rename("Q [projected]")
Q_BDM.rename("Q [BDM]")
file = VTKFile("bdm_test.pvd")
file.write(Q, Q_projected, Q_BDM)
