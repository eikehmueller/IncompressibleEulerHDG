import argparse
from firedrake import *

from incompressible_euler_conforming_implicit import (
    IncompressibleEulerConformingImplicit,
)
from incompressible_euler_dg_implicit import IncompressibleEulerDGImplicit
from incompressible_euler_hdg_implicit import IncompressibleEulerHDGImplicit
from incompressible_euler_hdg_imex import (
    IncompressibleEulerHDGARS232,
    IncompressibleEulerHDGEuler,
)

#######################################################################################
##                                M A I N                                            ##
#######################################################################################

if __name__ == "__main__":
    # Parse command line parameters

    parser = argparse.ArgumentParser("Mesh specifications and polynomial degree")
    parser.add_argument(
        "--nx",
        metavar="nx",
        type=int,
        action="store",
        default=8,
        help="number of grid cells in x-direction",
    )

    parser.add_argument(
        "--degree",
        metavar="degree",
        type=int,
        action="store",
        default=1,
        help="polynomial degree",
    )

    parser.add_argument(
        "--discretisation",
        choices=["conforming", "dg", "hdg"],
        type=str,
        action="store",
        default="hdg",
        help="discretisation method",
    )

    parser.add_argument(
        "--use_projection_method",
        action="store_true",
        default=False,
        help="use projection method for timestepping",
    )

    parser.add_argument(
        "--flux",
        choices=["upwind", "centered"],
        type=str,
        action="store",
        default="upwind",
        help="numerical flux",
    )

    parser.add_argument(
        "--timestepper",
        choices=["implicit_euler", "imex_implicit_euler", "imex_ars232"],
        type=str,
        action="store",
        default="imex_ars232",
        help="timestepper",
    )

    args = parser.parse_args()

    # final time
    t_final = 1.0
    # number of timesteps
    nt = args.nx
    # resulting timestep size
    dt = t_final / nt
    # decay constant, kappa=0 corresponds to stationary vortex
    kappa = 1

    mesh = UnitSquareMesh(args.nx, args.nx, quadrilateral=False)

    if args.discretisation == "conforming":
        # conforming discretisation
        print("Warning: ignoring degree for conforming method")
        assert (
            not args.use_projection_method
        ), "Can not use projection method with conforming discretsation"
        if args.timestepper == "implicit_euler":
            timestepper = IncompressibleEulerConformingImplicit(mesh, dt, args.flux)
        raise RuntimeError(
            "Invalid timestepping method for conforming discretisation: '{args.timestepper}'"
        )
    elif args.discretisation == "dg":
        # DG discretisation
        assert (
            not args.use_projection_method
        ), "Can not use projection method with DG discretsation"
        if args.timestepper == "implicit_euler":
            timestepper = IncompressibleEulerDGImplicit(mesh, args.degree, dt)
        else:
            raise RuntimeError(
                "Invalid timestepping method for DG discretisation: '{args.timestepper}'"
            )
    elif args.discretisation == "hdg":
        # HDG discretisation
        if args.timestepper == "implicit_euler":
            timestepper = IncompressibleEulerHDGImplicit(
                mesh, args.degree, dt, args.flux, args.use_projection_method
            )
        elif args.timestepper == "imex_ars232":
            timestepper = IncompressibleEulerHDGARS232(
                mesh,
                args.degree,
                dt,
                flux=args.flux,
                use_projection_method=args.use_projection_method,
            )
        elif args.timestepper == "imex_implicit_euler":
            timestepper = IncompressibleEulerHDGEuler(
                mesh,
                args.degree,
                dt,
                flux=args.flux,
                use_projection_method=args.use_projection_method,
            )
        else:
            raise RuntimeError(
                "Invalid timestepping method for HDG discretisation: '{args.timestepper}'"
            )

    print(f"+---------------------------------------------+")
    print(f"! IMEX-HDG for incompressible Euler equations !")
    print(f"+---------------------------------------------+")
    print()
    print(f"mesh size = {args.nx} x {args.nx}")
    print(f"polynomial degree = {args.degree}")
    print(f"final time = {t_final}")
    print(f"number of timesteps = {nt}")
    print(f"timestep size = {dt}")
    print(f"discretisation = {args.discretisation}")
    print(f"numerical flux = {args.flux}")
    print(f"use projection method = {args.use_projection_method}")
    print(f"timestepping method = {timestepper.label}")
    print()

    # initial conditions for velocity and pressure
    x, y = SpatialCoordinate(mesh)
    Q_stationary = as_vector(
        [
            -cos((x - 1 / 2) * pi) * sin((y - 1 / 2) * pi),
            sin((x - 1 / 2) * pi) * cos((y - 1 / 2) * pi),
        ]
    )
    p_stationary = (sin((x - 1 / 2) * pi) ** 2 + sin((y - 1 / 2) * pi) ** 2) / 2
    f_rhs = lambda t: -kappa * exp(-kappa * t) * Q_stationary

    Q, p = timestepper.solve(Q_stationary, p_stationary, f_rhs, t_final)

    V_Q = timestepper._V_Q
    V_p = timestepper._V_p

    Q.rename("velocity")
    p.rename("pressure")

    Q_exact = assemble(exp(-kappa * t_final) * Function(V_Q).interpolate(Q_stationary))
    Q_exact.rename("velocity_exact")
    p_exact = assemble(
        exp(-2 * kappa * t_final) * Function(V_p).interpolate(p_stationary)
    )
    p_exact -= assemble(p_exact * dx)
    p_exact.rename("pressure_exact")

    Q_error = assemble(Q - Q_exact)
    Q_error.rename("velocity_error")
    p_error = assemble(p - p_exact)
    p_error.rename("pressure_error")

    Q_error_nrm = np.sqrt(assemble(inner(Q_error, Q_error) * dx))
    p_error_nrm = np.sqrt(assemble((p_error) ** 2 * dx))
    print()
    print(f"Velocity error = {Q_error_nrm}")
    print(f"Pressure error = {p_error_nrm}")
    print()

    divQ = Function(V_p, name="divergence")
    phi = TrialFunction(V_p)
    psi = TestFunction(V_p)

    a_mass = phi * psi * dx
    b_hdiv_projection = div(Q) * psi * dx
    solve(a_mass == b_hdiv_projection, divQ)

    outfile = File("solution.pvd")
    outfile.write(Q, Q_exact, Q_error, p, p_exact, p_error, divQ)
