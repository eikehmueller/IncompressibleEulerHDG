# pylint: disable=wildcard-import,unused-wildcard-import

import sys
import time
import argparse
from firedrake import *
from firedrake.output import VTKFile

from auxilliary.utils import gridspacing
from auxilliary.logging import log_summary
from auxilliary.callbacks import AnimationCallback

from timesteppers.conforming_implicit import *
from timesteppers.dg_implicit import *
from timesteppers.hdg_implicit import *
from timesteppers.hdg_imex import *
from model_problems import TaylorGreen, KelvinHelmholtz, DoubleLayerShearFlow

#######################################################################################
##                                M A I N                                            ##
#######################################################################################

if __name__ == "__main__":
    # Parse command line parameters

    parser = argparse.ArgumentParser("Mesh specifications and polynomial degree")
    parser.add_argument(
        "--problem",
        choices=["taylorgreen", "kelvinhelmholtz", "shear"],
        type=str,
        action="store",
        default="taylorgreen",
        help="model problem to solve",
    )

    parser.add_argument(
        "--nx",
        metavar="nx",
        type=int,
        action="store",
        default=8,
        help="number of grid cells in x-direction",
    )

    parser.add_argument(
        "--refinement",
        metavar="refinement",
        type=int,
        action="store",
        default=2,
        help="refinement level for unit disk mesh",
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
        "--tfinal",
        metavar="tfinal",
        type=float,
        action="store",
        default=1.0,
        help="final time",
    )

    parser.add_argument(
        "--kappa",
        type=float,
        action="store",
        default=0.5,
        help="exponential decay factor",
    )

    parser.add_argument(
        "--dt",
        type=float,
        action="store",
        default=0.04,
        help="timestep size",
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
        "--richardson",
        metavar="richardson",
        type=int,
        action="store",
        default=2,
        help="number of Richardson iterations",
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

    parser.add_argument(
        "--warmup",
        action="store_true",
        default=False,
        help="only perform one timestep",
    )

    parser.add_argument(
        "--animation",
        action="store_true",
        default=False,
        help="save velocity and pressure fields at the end of each timestep as an animation",
    )

    parser.add_argument(
        "--tracer_advection",
        action="store_true",
        default=False,
        help="advect tracer field",
    )

    args = parser.parse_args()

    if args.problem == "taylorgreen":
        mesh = UnitSquareMesh(args.nx, args.nx, quadrilateral=False)
    elif args.problem == "shear":
        mesh = PeriodicSquareMesh(args.nx, args.nx, L=2 * np.pi, quadrilateral=False)
    elif args.problem == "kelvinhelmholtz":
        mesh = UnitDiskMesh(refinement_level=args.refinement)

    callbacks = [AnimationCallback("evolution.pvd")] if args.animation else None

    if args.discretisation == "conforming":
        # conforming discretisation
        print("Warning: ignoring degree for conforming method")
        if args.timestepper == "implicit":
            timestepper = IncompressibleEulerConformingImplicit(
                mesh,
                args.dt,
                args.flux,
                args.use_projection_method,
                callbacks=callbacks,
            )
        else:
            raise RuntimeError(
                "Invalid timestepping method for conforming discretisation: '{args.timestepper}'"
            )
    elif args.discretisation == "dg":
        # DG discretisation
        assert (
            not args.use_projection_method
        ), "Can not use projection method with DG discretsation"
        if args.timestepper == "implicit":
            timestepper = IncompressibleEulerDGImplicit(
                mesh, args.degree, args.dt, flux=args.flux, callbacks=callbacks
            )
        else:
            raise RuntimeError(
                "Invalid timestepping method for DG discretisation: '{args.timestepper}'"
            )
    elif args.discretisation == "hdg":
        # HDG discretisation
        if args.timestepper == "implicit":
            timestepper = IncompressibleEulerHDGImplicit(
                mesh,
                args.degree,
                args.dt,
                flux=args.flux,
                use_projection_method=args.use_projection_method,
                n_richardson=args.richardson,
                callbacks=callbacks,
            )
        elif args.timestepper == "imex_ars2_232":
            timestepper = IncompressibleEulerHDGIMEXARS2_232(
                mesh,
                args.degree,
                args.dt,
                flux=args.flux,
                use_projection_method=args.use_projection_method,
                n_richardson=args.richardson,
                callbacks=callbacks,
            )
        elif args.timestepper == "imex_ars3_443":
            timestepper = IncompressibleEulerHDGIMEXARS3_443(
                mesh,
                args.degree,
                args.dt,
                flux=args.flux,
                use_projection_method=args.use_projection_method,
                n_richardson=args.richardson,
                callbacks=callbacks,
            )
        elif args.timestepper == "imex_ssp2_332":
            timestepper = IncompressibleEulerHDGIMEXSSP2_332(
                mesh,
                args.degree,
                args.dt,
                flux=args.flux,
                use_projection_method=args.use_projection_method,
                n_richardson=args.richardson,
                callbacks=callbacks,
            )
        elif args.timestepper == "imex_ssp3_433":
            timestepper = IncompressibleEulerHDGIMEXSSP3_433(
                mesh,
                args.degree,
                args.dt,
                flux=args.flux,
                use_projection_method=args.use_projection_method,
                n_richardson=args.richardson,
                callbacks=callbacks,
            )
        elif args.timestepper == "imex_implicit":
            timestepper = IncompressibleEulerHDGIMEXImplicit(
                mesh,
                args.degree,
                args.dt,
                flux=args.flux,
                use_projection_method=args.use_projection_method,
                n_richardson=args.richardson,
                callbacks=callbacks,
            )
        else:
            raise RuntimeError(
                "Invalid timestepping method for HDG discretisation: '{args.timestepper}'"
            )

    print("+-------------------------------------------------+")
    print("! timesteppers for incompressible Euler equations !")
    print("+-------------------------------------------------+")
    print()
    print(f"model problem = {args.problem}")
    if args.problem == "taylorgreen":
        print(f"mesh size = {args.nx} x {args.nx}")
        print(f"forcing = {args.forcing}")
        print(f"kappa = {args.kappa}")
    elif args.problem == "shear":
        print(f"mesh size = {args.nx} x {args.nx}")
    elif args.problem == "kelvinhelmholtz":
        print(f"mesh refinement = {args.refinement}")
    print(f"polynomial degree = {args.degree}")
    print(f"final time = {args.tfinal}")
    print(f"timestep size = {args.dt}")
    print(f"discretisation = {args.discretisation}")
    print(f"numerical flux = {args.flux}")
    if type(timestepper) is IncompressibleEulerHDGIMEX and args.use_projection_method:
        print(f"number of Richardson iterations = {args.richardson}")
    print(f"use projection method = {args.use_projection_method}")
    print(f"advect tracer = {args.tracer_advection}")
    print(f"timestepping method = {timestepper.label}")
    print()

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

    if args.warmup:
        print("WARNING: performing a single timestep only!")
        print()

    if args.problem == "taylorgreen":
        model_problem = TaylorGreen(
            timestepper._V_Q, timestepper._V_p, args.forcing, args.kappa
        )
    elif args.problem == "shear":
        model_problem = DoubleLayerShearFlow(timestepper._V_Q, timestepper._V_p)
    elif args.problem == "kelvinhelmholtz":
        model_problem = KelvinHelmholtz(timestepper._V_Q, timestepper._V_p)

    Q_0, p_0 = model_problem.initial_condition()
    if args.tracer_advection:
        x, y = SpatialCoordinate(mesh)
        q_0 = sin(2 * pi * x) * sin(2 * pi * y)
    else:
        q_0 = None
    Q, p = timestepper.solve(
        Q_0, p_0, q_0, model_problem.f_rhs(), args.tfinal, warmup=args.warmup
    )

    log_summary()

    if not args.warmup:

        Q.rename("velocity")
        p.rename("pressure")
        V_p = timestepper._V_p
        divQ = Function(V_p, name="divergence")
        phi = TrialFunction(V_p)
        psi = TestFunction(V_p)

        a_mass = phi * psi * dx
        b_hdiv_projection = div(Q) * psi * dx
        solve(a_mass == b_hdiv_projection, divQ)

        output_fields = [Q, p, divQ]
        exact_solution = model_problem.solution(args.tfinal)
        if exact_solution is not None:
            Q_exact, p_exact = exact_solution
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
            output_fields += [Q_exact, Q_error, p_exact, p_error]

        outfile = VTKFile("solution.pvd")
        outfile.write(*output_fields)
