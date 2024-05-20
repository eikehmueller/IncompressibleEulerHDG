# pylint: disable=wildcard-import,unused-wildcard-import

import sys
import time
import argparse
from firedrake import *
from firedrake.output import VTKFile

from timesteppers.conforming_implicit import *
from timesteppers.dg_implicit import *
from timesteppers.hdg_implicit import *
from timesteppers.hdg_imex import *

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
        choices=[
            "implicit",
            "imex_implicit",
            "imex_ars2_232",
            "imex_ars3_443",
            "imex_ssp2_332",
            "imex_ssp3_433",
        ],
        type=str,
        action="store",
        default="imex_ssp2_332",
        help="timestepper",
    )

    parser.add_argument(
        "--forcing",
        choices=[
            "exponential",
            "constant",
        ],
        type=str,
        action="store",
        default="exponential",
        help="forcing",
    )

    parser.add_argument(
        "--test_pressure_solver",
        action="store_true",
        default=False,
        help="carry out a single solve with the pressure solver for testing",
    )

    args = parser.parse_args()

    # final time
    t_final = 1.0
    # number of timesteps
    nt = args.nx
    # resulting timestep size
    dt = t_final / nt
    # decay constant, kappa=0 corresponds to stationary vortex
    kappa = 0.5

    mesh = UnitSquareMesh(args.nx, args.nx, quadrilateral=False)

    if args.discretisation == "conforming":
        # conforming discretisation
        print("Warning: ignoring degree for conforming method")
        assert (
            not args.use_projection_method
        ), "Can not use projection method with conforming discretsation"
        if args.timestepper == "implicit":
            timestepper = IncompressibleEulerConformingImplicit(mesh, dt, args.flux)
        raise RuntimeError(
            "Invalid timestepping method for conforming discretisation: '{args.timestepper}'"
        )
    elif args.discretisation == "dg":
        # DG discretisation
        assert (
            not args.use_projection_method
        ), "Can not use projection method with DG discretsation"
        if args.timestepper == "implicit":
            timestepper = IncompressibleEulerDGImplicit(mesh, args.degree, dt)
        else:
            raise RuntimeError(
                "Invalid timestepping method for DG discretisation: '{args.timestepper}'"
            )
    elif args.discretisation == "hdg":
        # HDG discretisation
        if args.timestepper == "implicit":
            timestepper = IncompressibleEulerHDGIMEXImplicit(
                mesh, args.degree, dt, args.flux, args.use_projection_method
            )
        elif args.timestepper == "imex_ars2_232":
            timestepper = IncompressibleEulerHDGIMEXARS2_232(
                mesh,
                args.degree,
                dt,
                flux=args.flux,
                use_projection_method=args.use_projection_method,
            )
        elif args.timestepper == "imex_ars3_443":
            timestepper = IncompressibleEulerHDGIMEXARS3_443(
                mesh,
                args.degree,
                dt,
                flux=args.flux,
                use_projection_method=args.use_projection_method,
            )
        elif args.timestepper == "imex_ssp2_332":
            timestepper = IncompressibleEulerHDGIMEXSSP2_332(
                mesh,
                args.degree,
                dt,
                flux=args.flux,
                use_projection_method=args.use_projection_method,
            )
        elif args.timestepper == "imex_ssp3_433":
            timestepper = IncompressibleEulerHDGIMEXSSP3_433(
                mesh,
                args.degree,
                dt,
                flux=args.flux,
                use_projection_method=args.use_projection_method,
            )
        elif args.timestepper == "imex_implicit":
            timestepper = IncompressibleEulerHDGIMEXImplicit(
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

    print("+-------------------------------------------------+")
    print("! timesteppers for incompressible Euler equations !")
    print("+-------------------------------------------------+")
    print()
    print(f"mesh size = {args.nx} x {args.nx}")
    print(f"polynomial degree = {args.degree}")
    print(f"final time = {t_final}")
    print(f"number of timesteps = {nt}")
    print(f"timestep size = {dt}")
    print(f"discretisation = {args.discretisation}")
    print(f"numerical flux = {args.flux}")
    print(f"forcing = {args.forcing}")
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
    if kappa == 0:
        f_rhs = 0
    else:
        if args.forcing == "exponential":
            f_rhs = lambda t: -kappa * exp(-kappa * t) * Q_stationary
        elif args.forcing == "constant":
            f_rhs = lambda t: -kappa * Q_stationary
        else:
            raise NotImplementedError(f"Unknown forcing function: {args.forcing}")

    if args.test_pressure_solver:
        pcg = PCG64(seed=123456789)
        rg = RandomGenerator(pcg)
        f_Q = rg.normal(timestepper._V_Q, 0.0, 1.0)
        w, _, __ = TestFunctions(timestepper._V)
        state = Function(timestepper._V)
        b_rhs = inner(f_Q, w) * dx
        print("=== Testing pressure solver")
        print()
        _ = timestepper.pressure_solve(state, b_rhs)
        state.assign(0)
        t_start = time.perf_counter()
        its = timestepper.pressure_solve(state, b_rhs)
        t_finish = time.perf_counter()
        print(f"    solve time           = {t_finish-t_start:12.4f} s")
        print(f"    number of iterations = {its}")
        sys.exit()

    Q, p = timestepper.solve(Q_stationary, p_stationary, f_rhs, t_final)

    V_Q = timestepper._V_Q
    V_p = timestepper._V_p

    Q.rename("velocity")
    p.rename("pressure")

    if args.forcing == "exponential":
        Q_exact = assemble(
            exp(-kappa * t_final) * Function(V_Q).interpolate(Q_stationary)
        )
        p_exact = assemble(
            exp(-2 * kappa * t_final) * Function(V_p).interpolate(p_stationary)
        )
    elif args.forcing == "constant":
        Q_exact = assemble(
            (1.0 - kappa * t_final) * Function(V_Q).interpolate(Q_stationary)
        )
        p_exact = assemble(
            (1 - kappa * t_final) ** 2 * Function(V_p).interpolate(p_stationary)
        )
    p_exact -= assemble(p_exact * dx)
    Q_exact.rename("velocity_exact")
    p_exact.rename("pressure_exact")

    Q_error = assemble(Q - Q_exact)
    Q_error.rename("velocity_error")
    p_error = assemble(p - p_exact)
    p_error.rename("pressure_error")

    Q_error_nrm = np.sqrt(assemble(inner(Q_error, Q_error) * dx))
    p_error_nrm = np.sqrt(assemble((p_error) ** 2 * dx))
    print()
    print(f"velocity error = {Q_error_nrm}")
    print(f"pressure error = {p_error_nrm}")
    print()

    divQ = Function(V_p, name="divergence")
    phi = TrialFunction(V_p)
    psi = TestFunction(V_p)

    a_mass = phi * psi * dx
    b_hdiv_projection = div(Q) * psi * dx
    solve(a_mass == b_hdiv_projection, divQ)

    outfile = VTKFile("solution.pvd")
    outfile.write(Q, Q_exact, Q_error, p, p_exact, p_error, divQ)
