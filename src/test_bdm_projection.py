"""Test code for projecting a DG function to BDM^0 using three different methods:

1. The method described in Section 2.2 of Guzmán, J., Shu, C.W. and Sequeira, F.A., 2017. 
    "H (div) conforming and DG methods for incompressible Euler’s equations."
    IMA Journal of Numerical Analysis, 37(4), pp.1733-1771.

2. Firedrake's native L2 project functionality

3. Interpolation with averaging of dofs

The first method is implemented in two different ways.

"""

import numpy as np

import finat
from firedrake import *
from firedrake.output import VTKFile

from auxilliary.logging import PerformanceLog, log_summary


def interpolate_with_avg(Q, V, bcs):
    """Interpolate velocity to BDM space, averaging the dofs on the boundary.

    Original implementation by Pablo Brubeck, Oxford

    :arg Q: velocity to interpolate
    :arg V: BDM function space to interpolate into
    :arg bcs: list of boundary conditions
    """
    from firedrake.__future__ import interpolate

    # Compute (inverse) multiplicity of function space.

    V
    shapes = (V.finat_element.space_dimension(), np.prod(V.shape))
    domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
    instructions = """
    for i, j
        w[i,j] = w[i,j] + 1
    end
    """
    inverse_multiplicity = Function(V)
    par_loop((domain, instructions), dx, {"w": (inverse_multiplicity, INC)})
    with inverse_multiplicity.dat.vec as wv:
        wv.reciprocal()

    # interpolate with incremental access
    Q_interpolate = assemble(interpolate(Q, V, access=INC))
    with Q_interpolate.dat.vec as uv, inverse_multiplicity.dat.vec_ro as wv:
        uv.pointwiseMult(uv, wv)
    for bc in bcs:
        bc.apply(Q_interpolate)
    return Q_interpolate


def project_local_brute_force(Q, V, degree):
    """Project to BDM space as in Guzman et al., using the brute force approach

    :arg Q: velocity to interpolate
    :arg V: BDM function space to interpolate into
    :arg degree: polynomial degree of velocity space
    """
    mesh = V.mesh()
    el_broken_trace = finat.ufl.BrokenElement(
        finat.ufl.FiniteElement("DGT", cell=mesh.ufl_cell(), degree=degree)
    )
    el_nedelec = finat.ufl.BrokenElement(
        finat.ufl.FiniteElement("N1curl", cell=mesh.ufl_cell(), degree=degree - 1)
    )
    # Construct mixed function space
    V_nedelec = FunctionSpace(mesh, el_nedelec)
    V_broken_trace = FunctionSpace(mesh, el_broken_trace)
    W = V * V_nedelec * V_broken_trace

    # Trial- and test-functions in mixed function space
    u, gamma, omega = TrialFunctions(W)
    w, sigma, q = TestFunctions(W)

    n = FacetNormal(mesh)

    # Construct bilinear form
    a_mixed = (
        inner(u, sigma) * dx
        + 2 * avg(inner(u, n) * q) * dS
        + inner(u, n) * q * ds
        + inner(gamma, w) * dx
        + 2 * avg(omega * inner(n, w)) * dS
        + omega * inner(n, w) * ds
    )
    b_rhs_mixed = inner(Q, sigma) * dx + 2 * inner(avg(Q), avg(q * n)) * dS

    solution = Function(W)

    lvp = LinearVariationalProblem(a_mixed, b_rhs_mixed, solution)
    solver_parameters = {
        "ksp_type": "gmres",
        "pc_type": "jacobi",
        "ksp_monitor": None,
    }
    lvs = LinearVariationalSolver(lvp, solver_parameters=solver_parameters)
    lvs.solve()
    Q_projected = solution.subfunctions[0]
    return Q_projected


def project_local(Q, V, degree):
    """Project to BDM space as in Guzman et al., using Slate

    :arg Q: velocity to interpolate
    :arg V: BDM function space to interpolate into
    :arg degree: polynomial degree of velocity space
    """
    mesh = V.mesh()
    V_nedelec = FunctionSpace(mesh, "N1curl", degree - 1)
    V_trace = FunctionSpace(mesh, "DGT", degree)
    W = V_nedelec * V_trace

    # Construct bilinear form
    n = FacetNormal(mesh)
    u = TrialFunction(V_BDM)
    sigma, q = TestFunctions(W)
    a_mixed = (
        inner(u, sigma) * dx + 2 * avg(inner(u, q * n)) * dS + inner(u, n) * q * ds
    )

    # Right hand side
    b_rhs_mixed = inner(Q, sigma) * dx + 2 * inner(avg(Q), avg(q * n)) * dS

    A = Tensor(a_mixed).blocks
    f = Tensor(b_rhs_mixed).blocks
    S = (A[0, 0].T * A[0, 0] + A[1, 0].T * A[1, 0]).inv
    solution = S * (A[0, 0].T * f[0] + A[1, 0].T * f[1])
    Q_projected = Function(V)
    assemble(solution, tensor=Q_projected)
    return Q_projected


# Random number generation
pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)

# Mesh size
nx = 64
# Polynomial degree of vector function spaces
degree = 4

mesh = UnitSquareMesh(nx, nx, quadrilateral=False)

# Construct function spaces
V_DG = VectorFunctionSpace(mesh, "DG", degree)
V_BDM = FunctionSpace(mesh, "BDM", degree)

# Boundary conditions
bcs = [DirichletBC(V_BDM, (0, 0), j) for j in range(1, 5)]

# Random function to be projected
Q_original = rg.normal(V_DG)
Q_original.rename("original")

labels = [
    "original",
    "global projection",
    "interpolation",
    "local projection [brute force]",
    "local projection",
]
Q = {label: Function(V_BDM, name=label) for label in labels}

Q["original"] = Q_original

# Method 1: global L2 projection
with PerformanceLog("global projection"):
    Q["global projection"].project(Q_original, bcs=bcs)

# Method 2: interpolation
with PerformanceLog("interpolation"):
    Q["interpolation"].assign(interpolate_with_avg(Q_original, V_BDM, bcs))

# Method 3: brute force local projection
with PerformanceLog("local projection [brute force]"):
    Q["local projection [brute force]"].assign(
        project_local_brute_force(Q_original, V_BDM, degree)
    )

# Method 4: local projection with Slate
with PerformanceLog("local projection"):
    Q["local projection"].assign(project_local(Q_original, V_BDM, degree))

# Compute differences
print()
n = len(Q)
diff_matrix = np.empty((n, n))

short_keys = {
    "original": "original",
    "global projection": "P [global]",
    "interpolation": "interp",
    "local projection [brute force]": "P [loc, BF]",
    "local projection": "P [loc]",
}

s = 12 * " "
for i, key_i in enumerate(Q.keys()):
    s += f"{short_keys[key_i]:12s} "
print(s)
for i, key_i in enumerate(Q.keys()):
    s = f"{short_keys[key_i]:12s}"
    for j, key_j in enumerate(Q.keys()):
        diff_nrm = norm(Q[key_i] - Q[key_j])
        if i == j:
            s += f"   ---       "
        else:
            s += f"{diff_nrm:8.3e}    "
    print(s)

# Performance log
print()
log_summary()

file = VTKFile("bdm_test.pvd")
file.write(*Q.values())
