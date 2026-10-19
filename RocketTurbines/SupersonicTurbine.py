from IdealGas import IdealGas
from Turbine1D import Turbine1D
from TraupelLossModel import TraupelLossModel
import SupersonicTurbineBlading as blading
from RelaxedAitken import solve_with_relaxed_aitken
import numpy as np
from scipy.optimize import brentq, least_squares
import warnings

class SupersonicTurbine:
    def __init__(self, gas: IdealGas, loss_model: TraupelLossModel, Turbine1D_arguments,
                 SupersonicStatorNozzle_arguments, SupersonicRotorBlades_arguments, separation_margin,
                 *, t_min_stator, t_min_rotor, stator_nozzle_chord_scale=1.):
        """SupersonicTurbine is the class to represent a supersonic turbine, with consistent meanline design and blading
         shape. It ensures that chord/pitch ratios and other variables used in Turbine1D agrees with the values obtained from
         SupersonicTurbineBlading objects, which in turn also use Turbine1D meanline results. Thus, there is a coupling
          between meanline and blading design, which this class solves numerically. This object also checks if the
           stator nozzles exit can fit optimally in the turbine annulus by producing continuous coefficients than can be
            either checked by the engineer or constrained by outer optimizer.

        :param IdealGas gas: Calorically perfect working gas, shared by both design libraries.
        :param TraupelLossModel loss_model: Meanline loss model.
        :param dict Turbine1D_arguments: Arguments to size_turbine, including initial coupling estimates.
         gas and loss_model are supplied by this object. For one nozzle, admission_fraction is its initial estimate
         (default 0.1); its final value is solved from the exit geometry.
        :param dict SupersonicStatorNozzle_arguments: Nozzle controls, including contour_method ("moc" or "conical").
         Flow conditions, reference_length and trailing_edge_thickness are supplied by the coupling.
        :param dict SupersonicRotorBlades_arguments: Rotor controls. Flow conditions, surface Mach numbers,
         blade count, radius and speed are supplied by the coupling. leading_edge_thickness_over_total_pitch is
         an optional initial LE estimate; the coupling solves LE and geometric TE thickness independently.
        :param float separation_margin: Fraction of the separation-limited Mach interval held in reserve,
         strictly between zero and one. A value of 0.05 reproduces Landergren's equation (23).
        :param float t_min_stator: Minimum true stator metal thickness, m; nonnegative.
        :param float t_min_rotor: Minimum true rotor LE/TE metal thickness, m; nonnegative.
        :param float stator_nozzle_chord_scale: Positive multiplier of throat-centre to exit-centre chord,
         to account for the upstream turning section. Default 1.
        :ivar numpy.ndarray solution: Converged [stator chord/pitch, rotor chord/pitch, rotor TE,
         stator TE (or single-nozzle admission), rotor LE]; thicknesses are circumferential projections in m.
        :raises ValueError: If inputs or the local annulus/nozzle construction are outside their domains.
        :raises RuntimeError: If coupling fails to satisfy the final normalized residual tolerance of 1e-6.
        """

        # Store objects
        self.gas = gas
        self.loss_model = loss_model

        # Store dictionaries with Turbine1D, SupersonicRotorBlade and SupersonicStatorNozzle
        # (both from SupersonicTurbineBlading library) input arguments as dictionaries. These can be used to call
        # any method or function by passing **dictionary to them
        self.Turbine1D_arguments = Turbine1D_arguments.copy()
        self.SupersonicStatorNozzle_arguments = SupersonicStatorNozzle_arguments.copy()
        self.SupersonicRotorBlades_arguments = SupersonicRotorBlades_arguments.copy()
        # Supply shared objects explicitly without modifying the caller's dictionaries.
        self.Turbine1D_arguments.update(gas=gas, loss_model=loss_model)

        # Store separation margin which will be used to calculate surface Mach numbers for upper and lower rotor surface
        # in relative reference frame
        self.separation_margin = separation_margin
        # Validate manufacturing inputs and store the optional upstream chord allowance.
        if not np.isfinite(separation_margin) or not 0 < separation_margin < 1:
            raise ValueError("separation_margin must be strictly between zero and one.")
        if not np.all(np.isfinite([t_min_stator, t_min_rotor])) or min(t_min_stator, t_min_rotor) < 0:
            raise ValueError("Minimum metal thicknesses must be finite and nonnegative.")
        if not np.isfinite(stator_nozzle_chord_scale) or stator_nozzle_chord_scale <= 0:
            raise ValueError("stator_nozzle_chord_scale must be finite and positive.")
        self.t_min_stator, self.t_min_rotor = t_min_stator, t_min_rotor
        self.stator_nozzle_chord_scale = stator_nozzle_chord_scale

        # Create and store Fluid object for SupersonicTurbineBlading objects.
        self.SupersonicTurbineBlading_Fluid = blading.Fluid(ideal_gas=gas)

        # Determine number of stator nozzles, since this will determine what is the solution vector to achieve
        # consistent design. For more details, see comments in evaluate_design.
        no_nozzles = self.Turbine1D_arguments["no_blades_stator"]
        # If there is just one nozzle, admission fraction is not a free design variable anymore. In such case,
        # retain its input only as an initial estimate and force zero stator trailing-edge blockage.
        if isinstance(no_nozzles, (bool, np.bool_)) or not isinstance(no_nozzles, (int, np.integer)) or no_nozzles < 1:
            raise ValueError("no_blades_stator must be a positive integer.")
        if no_nozzles == 1:
            self.Turbine1D_arguments["t_TE_stator"] = 0.
            # A local single nozzle starts with partial admission rather than an annulus-spanning meanline chord.
            self.Turbine1D_arguments.setdefault("admission_fraction", 0.1)
        # Rotor and stator objects from SupersonicTurbineBlading can adjust the blading to achieve flow angles from
        # meanline design. However, in this case, they cannot adjust the blading to achieve some real Mach number after
        # mixing. This is because isentropic values from meanline design used to construct the blading,
        # already correspond to real values to be achieved when mixing and losses occur. Consequently, if the blading
        # was to be adjusted by SupersonicTurbineBlading based on boundary layer and
        # mixing calculations to achieve real Mach number from the meanline code, this would double count the losses.
        # Consequently, if the user specifies this option in the arguments to SupersonicTurbineBlading objects, it
        # should be removed here. Raise a user warning about it too.
        for arguments, option in ((self.SupersonicStatorNozzle_arguments, "match_real_outlet_absolute_flow_mach"),
                                  (self.SupersonicRotorBlades_arguments, "match_real_outlet_mach")):
            # Remove each matching control independently; the two APIs use different names.
            if arguments.pop(option, False):
                warnings.warn(f"{option} is disabled: matching mixed Mach would count meanline losses twice.",
                              UserWarning, stacklevel=2)

        # Define a function that calculates new values of design variables from evaluate_design based on assumed ones
        self.coupling_evaluations = 0
        def calculate_new_solution(x_0):
            """Return the five obtained design variables and their associated design objects."""
            # Count full meanline/blading evaluations, including fallback finite differences.
            self.coupling_evaluations += 1
            # If there are multiple nozzles, one of such variables is t_TE_stator
            if no_nozzles != 1:
                # Unpack input arguments
                chord_over_pitch_stator, chord_over_pitch_rotor, t_TE_rotor, t_TE_stator, t_LE_rotor = x_0
                # Copy Turbine1D arguments dictionary so that only internal copy changes during iterations
                meanline_arguments = self.Turbine1D_arguments.copy()
                meanline_arguments.update({"chord_over_pitch_stator": chord_over_pitch_stator,
                                           "chord_over_pitch_rotor": chord_over_pitch_rotor,
                                           "t_TE_rotor": t_TE_rotor,
                                           "t_TE_stator": t_TE_stator})
                # Get all residuals, Turbine1D object, stator & rotor shapes and optimality coefficients & thickness
                # margins
                residuals, turbine_meanline, stator_shape, rotor_shape, coefficients =\
                    self.evaluate_design(meanline_arguments, t_LE_rotor=t_LE_rotor)
                # Obtain new estimate of design values from residuals
                x_1 = np.asarray(x_0) + residuals
            # If there is just one nozzle, t_TE_stator is always zero. However, admission fraction must be now
            # variable to adjust
            else:
                # Unpack input arguments
                chord_over_pitch_stator, chord_over_pitch_rotor, t_TE_rotor, admission_fraction, t_LE_rotor = x_0
                # Copy Turbine1D arguments dictionary so that only internal copy changes during iterations
                meanline_arguments = self.Turbine1D_arguments.copy()
                meanline_arguments.update({"chord_over_pitch_stator": chord_over_pitch_stator,
                                           "chord_over_pitch_rotor": chord_over_pitch_rotor,
                                           "t_TE_rotor": t_TE_rotor,
                                           "admission_fraction": admission_fraction})
                # Get all residuals, Turbine1D object, stator & rotor shapes and optimality coefficients & thickness
                # margins
                residuals, turbine_meanline, stator_shape, rotor_shape, coefficients = \
                    self.evaluate_design(meanline_arguments, t_LE_rotor=t_LE_rotor)
                # Obtain new estimate of design values from residuals
                x_1 = np.asarray(x_0) + residuals
            # Retain the best valid trial for a bounded least-squares fallback and final diagnostics.
            normalized_residual = (x_1 - x_0) / solution_scales
            if np.linalg.norm(normalized_residual) < best_trial[0]:
                best_trial[:] = [np.linalg.norm(normalized_residual), np.array(x_0, copy=True)]
            # Return new solution estimate together with remaining results
            return x_1, turbine_meanline, stator_shape, rotor_shape, coefficients

        # Create initial solution array if there are multiple nozzles
        if no_nozzles != 1:
            x_initial = np.array([self.Turbine1D_arguments["chord_over_pitch_stator"],
                                  self.Turbine1D_arguments.get("chord_over_pitch_rotor", 2.5),
                                  self.Turbine1D_arguments["t_TE_rotor"],
                                  self.Turbine1D_arguments["t_TE_stator"]], dtype=float)
        # If there is just one nozzle
        else:
            x_initial = np.array([self.Turbine1D_arguments["chord_over_pitch_stator"],
                                  self.Turbine1D_arguments.get("chord_over_pitch_rotor", 2.5),
                                  self.Turbine1D_arguments["t_TE_rotor"],
                                  self.Turbine1D_arguments.get("admission_fraction", 1.)], dtype=float)

        # Use numerical solver to that will create a consistent design, in which values assumed for sizing
        # in Turbine1D are equivalent to those obtained from SupersonicTurbineBlading. Use Aitken's scheme.
        # Use fixed Euler-radius pitches to bound thicknesses and normalize length and dimensionless variables.
        arguments = self.Turbine1D_arguments
        radius = np.sqrt(arguments["shaft_power"] / arguments["mdot"] / arguments["work_coefficient_tt"]) \
                 / (arguments["RPM"] * np.pi / 30.)
        rotor_pitch = 2 * np.pi * radius / arguments["no_blades_rotor"]
        # LE controls the shape scale independently of the TE thickness used by the meanline loss model.
        initial_LE = self.SupersonicRotorBlades_arguments.get(
            "leading_edge_thickness_over_total_pitch", x_initial[2] / rotor_pitch) * rotor_pitch
        x_initial = np.append(x_initial, initial_LE)
        lower = np.array([1e-8, 1e-8, 0., 0. if no_nozzles > 1 else 1e-6, 0.])
        upper = np.array([np.inf, np.inf, 0.95 * rotor_pitch,
                          0.95 * arguments.get("admission_fraction", 1.) * 2 * np.pi * radius / no_nozzles
                          if no_nozzles > 1 else 1., 0.95 * rotor_pitch])
        solution_scales = np.array([max(x_initial[0], 1.), max(x_initial[1], 1.), rotor_pitch,
                                    upper[3] if no_nozzles > 1 else max(x_initial[3], 0.01), rotor_pitch])
        if not np.all(np.isfinite(x_initial)) or np.any(x_initial < lower) or np.any(x_initial > upper):
            raise ValueError("Initial coupling estimates must be finite and within positive pitch/thickness bounds.")
        best_trial = [np.inf, x_initial.copy()]

        # Solve in normalized coordinates so all five equations receive the same residual tolerance.
        def normalized_mapping(values):
            """Evaluate the fixed-point map using fixed component scales."""
            return calculate_new_solution(values * solution_scales)[0] / solution_scales

        try:
            x_solution = solution_scales * solve_with_relaxed_aitken(
                normalized_mapping, x_initial / solution_scales, initial_relaxation=0.3,
                min_relaxation=0.05, max_relaxation=0.8, xtol=1e-7, maxiter=15,
                bounds=(lower / solution_scales, upper / solution_scales))
            self.coupling_solver = "relaxed_aitken"
        except (ValueError, RuntimeError, ArithmeticError) as aitken_error:
            # Require one valid design before fallback: an invalid initial geometry is not a convergence problem.
            if not np.isfinite(best_trial[0]):
                try:
                    calculate_new_solution(x_initial)
                except (ValueError, RuntimeError, ArithmeticError) as domain_error:
                    raise ValueError(f"Initial supersonic design is outside the model domain: {domain_error}") from domain_error

            # Keep the domain-rejection cost fixed and above the best valid starting cost.
            rejection_cost = 10 * (1 + best_trial[0])
            def normalized_residual(values):
                """Return physical residuals, or a finite rejection cost for an out-of-domain trial."""
                try:
                    return normalized_mapping(values) - values
                except (ValueError, RuntimeError, ArithmeticError):
                    # A cost above every retained valid trial makes SciPy shrink the step at a domain boundary.
                    # This barrier is never accepted as a converged design: final residuals are evaluated directly.
                    return np.full(values.shape, rejection_cost)

            # Restart from the best valid Aitken iterate, using bounds and the same normalization.
            try:
                solution = least_squares(normalized_residual, best_trial[1] / solution_scales,
                                         bounds=(lower / solution_scales, upper / solution_scales),
                                         method="trf", jac="3-point",
                                         xtol=1e-9, ftol=1e-9, gtol=1e-13, max_nfev=30)
            except (ValueError, RuntimeError, ArithmeticError) as fallback_error:
                raise RuntimeError("Supersonic coupling fallback encountered an invalid trial; revise initial "
                                   f"estimates or design inputs. Aitken: {aitken_error}. "
                                   f"Fallback: {fallback_error}.") from fallback_error
            x_solution = solution.x * solution_scales
            self.coupling_solver = "least_squares"

        # Obtain consistent design objects after solution was found
        obtained, self.turbine_meanline, self.stator_shape, self.rotor_shape, coefficients = \
            calculate_new_solution(x_solution)
        # Check actual residuals even if a solver reports success or stops with a small step.
        self.coupling_residuals = obtained - x_solution
        self.normalized_coupling_residuals = self.coupling_residuals / solution_scales
        if not np.all(np.isfinite(obtained)) or np.max(np.abs(self.normalized_coupling_residuals)) >= 1e-6:
            raise RuntimeError(f"Supersonic turbine coupling did not converge: {self.normalized_coupling_residuals}.")
        self.solution = x_solution.copy()
        self.geometry = coefficients
        # Expose the independent projected LE and the contour-derived TE in metres.
        self.t_LE_rotor = coefficients["rotor_contour_LE_thickness"]
        self.t_TE_rotor = coefficients["rotor_contour_TE_thickness"]
        # Save optimality coefficients
        # Coefficient for height is valid for all nozzles
        self.height_optimality = coefficients["height_optimality"]
        # Coefficient for width is valid only for da Laval nozzle, as for MOC nozzle width of the exit will always
        # match the assigned nozzle space. For MOC nozzle, it is consequently always 0, which means it is always
        # optimal.
        self.width_optimality = coefficients["width_optimality"]
        # Margin on the minimum thickness for the stator is valid only for da Laval nozzle. For MOC nozzle, minimum
        # thickness is always satisfied explicitly during calculating width of the nozzle exit. Consequently, in that case
        # this coefficient will be always 0.
        self.t_TE_stator_min_margin = coefficients["t_TE_stator_min_margin"]
        # Warn once about infeasible exit packing while retaining signed diagnostics for an outer optimizer.
        if self.height_optimality > 1e-8 or self.t_TE_stator_min_margin < -1e-8:
            warnings.warn("Consistent design has infeasible stator exit packing; inspect height_optimality and "
                          "t_TE_stator_min_margin (m).", UserWarning, stacklevel=2)
    def evaluate_design(self, meanline_arguments, *, t_LE_rotor=None):
        """This method allows to calculate the residuals between assumed values for 1D meanline calculations in
        Turbine 1D and obtained values from blade shaping with SupersonicTurbineBlading. These residuals can be driven
         to zero by a numerical solver in the constructor of this class or the engineer to obtain consistent design.
          This method also returns some remaining residuals and optimality coefficients that can be constrained by outer
           optimizer or which can inform the engineer about manufacturability & optimality of the supersonic turbine.

        :param dict meanline_arguments: Complete size_turbine inputs or overrides of the constructor inputs.
        :param float t_LE_rotor: Independent circumferential LE thickness (m), excluding BL correction.
         Defaults to the converged LE, or the input LE estimate before convergence.
        :return: Five residuals, meanline object, stator object, rotor object and geometry dict.
         Residuals 1/2 are dimensionless chord/pitch differences. Residual 3 is geometric-minus-meanline rotor
         TE thickness (m). Residual 4 is stator arc thickness (m), or admission fraction for one nozzle.
         Residual 5 is the LE correction enforcing the minimum true metal thickness at both rotor edges (m).
         height_optimality and width_optimality are dimensional excesses (m); positive values indicate overfilling.
         t_TE_stator_min_margin is signed projected clearance minus the required projected metal thickness (m).
         Multiplying this margin by cos(alpha_1) gives the approximate true metal-thickness margin.
         All chords, exit intersections and dimensional geometry diagnostics use uncorrected shapes.
        :rtype: tuple[numpy.ndarray, Turbine1D, blading.SupersonicStatorNozzle, blading.SupersonicRotorBlade, dict]
        :raises ValueError: If the flow is not supersonic or the local exit/annulus intersection is outside the model.
        """

        # The automatic coupling between Turbine1D (meanline code) and SupersonicTurbineBlading (blade shaping) is
        # available for matching chord/pitch ratios of the stator and the rotor by driving difference between assumed
        # and obtained values to zero. These residuals are chord_to_pitch_ratio_stator and chord_to_pitch_ratio_rotor.
        # However, the code cannot automatically ensure stator nozzles match the annulus of the turbine, since it
        # requires varying nozzle number or flow coefficients, which are preferentially things adjusted by the user or
        # some outer optimizer. Fortunately, the code can determine if stator nozzles are placed optimally by using a
        # few simple & continuous coefficients, which can be constrained by outer optimizer or which can give user
        # indication of the feasibility/optimality of the stator. These calculations are required to calculate
        # chord_to_pitch_ratio_stator, so they are performed already in the numerical solution loop. They also yield
        # arc length t_TE_stator between edges of adjacent nozzle exits, which must match the assumed value for
        # meanline calculations and blade shaping as well. Consequently, another design consistency
        # condition arise, t_TE_stator_residual, which the numerical solver will attempt to make zero.

        # Furthermore, a word about nomenclature and definition of trailing edge thickness. In Turbine1D, TraupelLossModel,
        # in this class or in SupersonicRotorBlading library, t_TE or t_LE means the length of the projection of
        # unwrapped trailing/leading edge on the circumferential axis. This allows to easily perform loss or MOC
        # calculations. However, while often abbreviated as thickness elsewhere, this is not the same as true metal thickness
        # - the blade/nozzles inlets and outlets are angled, so actual metal thicknesses there are smaller. The minimal
        # allowed values of this true metal thickness are given as t_min_rotor and t_min_stator as inputs to this object.
        # Stator blade outlet angle is then used to change t_min_stator to t_TE_stator_min. MOC nozzle shaping is done
        # such that t_TE_stator_min_obtained is never below t_TE_stator_min. For da Laval nozzle this is not possible,
        # since nozzle exit height and widths are coupled and there is one less degree of freedom in shaping it.
        # Consequently, t_TE_stator_min_margin = t_TE_stator_min_obtained - t_TE_stator_min is calculated for
        # it, which the outer optimizer or designed must always keep above 0 by changing number of nozzles, admission
        # fraction, etc. Rotor blade inlet and outlet angles also convert actual t_LE_rotor and t_TE_rotor
        # to normal metal thicknesses; their minimum is t_TE_rotor_min_obtained. It is used to calculate
        # t_TE_rotor_optimality = t_TE_rotor_min_obtained - t_min_rotor. To ensure fair comparison,
        # the engineer or outer optimizer should always compare rotors with the same metal thickness. Consequently,
        # the numerical solver in this class will also ensure that t_TE_rotor_optimality is zero by varying t_LE_rotor.
        # An independent residual matches the meanline t_TE_rotor to the actual uncorrected contour TE thickness.
        # To convert stator TE and rotor LE/TE projections to normal metal thicknesses, flow angles
        # from meanline design are always used. These are approximately the same as nozzle/rotor blade metal angles.
        # Blade metal angles can be a few degrees different if the user specifies in input arguments to
        # SupersonicTurbineBlading objects that they should be adjusted to obtain flow angles equal to those from
        # meanline design, however this has a negligible influence. Finally, boundary layer correction for stator and
        # rotor shapes can lead to blade thickness for BL corrected shapes to be different from t_TE_stator_min_obtained
        # or t_TE_rotor_min_obtained. SupersonicRotorBlading takes an LE thickness input; prescribed endpoint
        # states need not close the inviscid passage pitch, so the actual contour TE is measured independently.

        # Shaping of nozzles will be now discussed.
        # The optimal placement of nozzles means that:
        # 1. the area a single nozzle takes in the annulus sector dedicated to it is maximized for given constraints;
        # this ensures the most uniform flow, which is the basic assumption of all 1D meanline codes.
        # 2. the arc length at Euler diameter between edges of neighbouring nozzles equals trailing edge thickness used
        # in meanline analysis
        # 3. the minimum spacing between edges of the neighbouring nozzles is above some specified minimum value of
        # manufacturable metal thickness
        # 4. the nozzles must allow for specified massflow at design point
        # Stator nozzles can be either da Laval or MOC nozzles and there can be either many or one of them, which
        # require slightly different treatment of conditions above.

        # Starting with MOC nozzles. First, width of the rectangular exit, w_nozzle, can be calculated such that point 3 is
        # satisfied for given annulus dimensions, admission fraction, number of nozzles and t_TE_stator_min. Then,
        # maximum possible height, H_max, for the rectangular nozzle can be calculated. w_nozzle dimensionalizes MOC shape in
        # SupersonicStatorNozzle, which in turn settles throat width. Throat area can be calculated such that point 4 is
        # satisfied. Throat area and width are known, so its height can be calculated too. It is the same as height of
        # the nozzle exit. This height, H_nozzle, must be smaller or equal to already calculated H_max - otherwise the
        # nozzle cannot fit in the annulus without violating point 3. Consequently, nozzle optimality factor,
        # height_optimality is H_nozzle - H_max. If it is positive, geometry is not feasible. If it is zero,
        # the geometry is optimal as nozzle exit has the largest possible area and matches annulus well. This satisfies
        # point 1. If it is negative, the geometry is feasible, but may be not optimal, especially for large negative
        # values. Finally, the arc length at Euler diameter between edges of adjacent nozzles, t_TE_stator_obtained,
        # can be calculated. It must be equal to the arc length t_TE_stator assumed during meanline calculations.
        # Their difference is t_TE_stator_residual and the numerical solver must hold it at zero - it means that assumed
        # and obtained trailing edge thicknesses are consistent, and that meanline code and nozzle shapes agree with
        # each other. This satisfies point 2.

        # This was a procedure when multiple MOC nozzles are used. If there is a single MOC nozzle, a few things change.
        # Firstly, points 2 and 3 go away, as t_TE_stator for the meanline code is always zero and there is no adjecent
        # nozzle for the engineer to worry about metal thickness. Point 1 sets the dimensions of the rectangle with
        # greatest area for given annulus dimensions, with its centre constrained to Euler radius.
        # Its width and height are w_nozzle and H_max; the actual exit retains the same centre.
        # w_nozzle can be used to calculate throat width and height that satisfy point 4, and which also settles
        # H_nozzle. Consequently, height_optimality can be again calculated, while t_TE_stator_residual is always
        # zero. However, a single nozzle also constraints admission fraction of the stator. admission_fraction_obtained
        # can be thus calculated. admission_fraction_residual can be thus defined as the difference between
        # admission_fraction_obtained and the admission fraction assumed for meanline calculations.

        # Regarding da Laval nozzle, the idea is similar, but the difference is that throat is circular and the exit
        # nozzle has elliptical shape from axial perspective. Consequently, massflow settles throat area and diameter.
        # This satisfies point 4. From throat diameter D_throat and required expansion ratio (settled by stator
        # isentropic exit Mach number), exit diameter D_nozzle can be calculated. It is s the same as nozzle exit
        # ellipse minor diameter, which preferentially matches annulus height, H_annulus. This is the first condition to
        # satisfy point 1. Thus, height_optimality factor for da Laval nozzle is D_nozzle - H_annulus. Its positive,
        # zero and negative values have the same meaning as height_optimality for MOC nozzle. Elliptical shape of the
        # nozzle exit is just a cross-section of a cylinder with diameter D_nozzle with the annulus plane at the stator
        # nozzle angle. Therefore, major diameter of the nozzle exit ellipse, D_ellipse, is also set by D_nozzle
        # (D_ellipse be also obtained from da Laval nozzle shape from SupersonicStatorNozzle scaled with throat diameter).
        # Ellipse centre is placed at mean diameter of the annulus, thus the arc distance between edges of adjacent
        # nozzles at Euler diameter, t_TE_stator_obtained, can be calculated. Just like for MOC nozzles, it must be
        # equal to the arc distance t_TE_stator assumed during meanline calculations. Their difference is
        # again t_TE_stator_residual, which must be held at zero. This satisfies point 2 and is the second condition
        # to satisfy point 1. It may happen that t_TE_stator_obtained will be nagative, because stator parameters cause
        # adjacent nozzles to overlap. In such case, to still ensure consistent design can be created without code
        # failure, t_TE_stator_obtained is clipped at zero. The issue of overlapping nozzles must be then solved by the
        # engineer or outer optimizer. To do so, the minimum distance between edges of adjacent nozzles,
        # t_TE_stator_min_obtained, can be calculated. It is very close to t_TE_stator_obtained, but still
        # different. Consequently, t_TE_stator_min_margin can be calculated, which is t_TE_stator_min_obtained
        # - t_TE_stator_min. This factor must be zero or positive for geometry to be manufacturable and for point 3 to
        # be satisfied. If the engineer or outer optimizer hold t_TE_stator_min_margin at least at zero, it means
        # that nozzles are not overlapping and that t_TE_stator_obtained is positive, since it is always greater than
        # t_TE_stator_min_margin. Consequently, the design is then consistent and the nozzles are feasible.

        # This was a procedure when multiple da Laval nozzles are used. If there is a single nozzle, analogous
        # procedure to singular MOC nozzle is followed. Points 2 and 3 go away. height_optimality is again calculated
        # the same way. D_ellipse is known, so admission_fraction_obtained can be calculated.
        # admission_fraction_residual can be defined, which the numerical solver must hold at zero.

        # First, obtain meanline analysis results from Turbine1D
        meanline_arguments = {**self.Turbine1D_arguments, **meanline_arguments, "gas": self.gas, "loss_model": self.loss_model}
        # One nozzle has no adjacent trailing-edge wall, including when this method is called independently.
        if meanline_arguments["no_blades_stator"] == 1:
            meanline_arguments["t_TE_stator"] = 0.
        turbine_meanline = Turbine1D()
        turbine_meanline.size_turbine(**meanline_arguments)
        # Read the actual inlet and isentropic rotor outlet reference states.
        results = turbine_meanline.analysis_results_at_design_point
        # Shape the blade row using its outlet direction; the stage angle includes losses outside the passage.
        rotor_outlet_angle = turbine_meanline.blade_row_results_at_design_point["beta_2_blade"]
        M_in, M_out = results["M_r1"], results["M_r2_ideal_r_real_s"]
        if min(M_in, M_out, results["M_s1_ideal"]) <= 1:
            raise ValueError("Supersonic blading requires M_r1, M_r2_ideal_r_real_s and M_s1_ideal > 1.")

        # Now calculate rotor surface Mach numbers based on separation margins
        # First lower Mach number
        M_lower_min, M_upper_max = self._separation_mach_limits(M_in, M_out)
        M_lower = M_in - (1 - self.separation_margin) * (M_in - M_lower_min)
        # Now upper Mach number
        M_upper = M_out + (1 - self.separation_margin) * (M_upper_max - M_out)
        # An accelerating/decelerating rotor must also bracket both endpoint Mach numbers.
        if not 1 <= M_lower < min(M_in, M_out) or M_upper <= max(M_in, M_out):
            raise ValueError("Separation-margin surface Mach numbers do not bracket both rotor endpoint Mach numbers.")
        # Now calculate rotor shape and dimensions. Use real Mach number at station 1 from Turbine 1D and Mach number
        # at station 2 that assumes isentropic expansion through the rotor
        # Independent evaluations use the converged LE unless an explicit trial thickness is supplied.
        if t_LE_rotor is None:
            t_LE_rotor = getattr(self, "t_LE_rotor", self.SupersonicRotorBlades_arguments.get(
                "leading_edge_thickness_over_total_pitch", turbine_meanline.t_TE_rotor / turbine_meanline.p_rotor)
                * turbine_meanline.p_rotor)
        if not np.isfinite(t_LE_rotor) or not 0 <= t_LE_rotor < turbine_meanline.p_rotor:
            raise ValueError("t_LE_rotor must be finite and in [0, rotor pitch).")
        rotor_arguments = {**self.SupersonicRotorBlades_arguments,
                           "ideal_inlet_relative_flow_mach": M_in,
                           "ideal_inlet_relative_flow_angle": results["beta_1_deg"],
                           "requested_outlet_relative_flow_mach": M_out,
                           "requested_outlet_relative_flow_angle": np.degrees(rotor_outlet_angle),
                           "lower_surface_relative_flow_mach": M_lower,
                           "upper_surface_relative_flow_mach": M_upper,
                           "blade_count": turbine_meanline.no_blades_rotor,
                           "mean_radius": turbine_meanline.R_Euler,
                           "rotational_speed_rpm": results["RPM"],
                           "fluid": self.SupersonicTurbineBlading_Fluid,
                           "inlet_total_temperature": results["h_t1"] / self.gas.Cp,
                           "inlet_total_pressure": results["p_t1"],
                           "leading_edge_thickness_over_total_pitch":
                               t_LE_rotor / turbine_meanline.p_rotor}
        rotor_shape = blading.SupersonicRotorBlade(**rotor_arguments)

        # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the rotor
        # The two stored surfaces bound a passage. Their endpoint averages give the blade centreline displacement
        # after translating the suction side by one total pitch; the constant translation cancels in the difference.
        rotor_ideal = rotor_shape.dimensionalize().uncorrected
        pressure, suction = rotor_ideal.pressure_surface, rotor_ideal.suction_surface
        rotor_chord = np.hypot((pressure.x[-1] + suction.x[-1] - pressure.x[0] - suction.x[0]) / 2,
                               (pressure.y[-1] + suction.y[-1] - pressure.y[0] - suction.y[0]) / 2)
        rotor_ratio_residual = rotor_chord / turbine_meanline.p_rotor - meanline_arguments.get("chord_over_pitch_rotor", 2.5)

        # Calculate t_TE_rotor_optimality. t_TE_rotor_optimality is part of residuals, since it will always be
        # decreased to zero by the optimizer.
        # Subtract open uncorrected passages from total Euler-radius pitch to obtain actual metal projections.
        rotor_LE = turbine_meanline.p_rotor - rotor_ideal.inlet_passage_pitch
        rotor_TE = turbine_meanline.p_rotor - rotor_ideal.outlet_passage_pitch
        rotor_thickness_residual = rotor_TE - turbine_meanline.t_TE_rotor
        # Constrain the thinner true edge, allowing the two geometric thicknesses to differ.
        rotor_cosines = np.cos([results["beta_1"], rotor_outlet_angle])
        if np.min(rotor_cosines) <= 0:
            raise ValueError("Rotor inlet/outlet angles must have positive axial components.")
        rotor_metal_thicknesses = np.array([rotor_LE, rotor_TE]) * rotor_cosines
        limiting_edge = np.argmin(rotor_metal_thicknesses)
        t_TE_rotor_optimality = rotor_metal_thicknesses[limiting_edge] - self.t_min_rotor
        rotor_LE_residual = -t_TE_rotor_optimality / rotor_cosines[limiting_edge]

        # Calculate total throat area needed to satisfy massflow requirements
        gamma = self.gas.gamma
        throat_mass_flux = results["p_0"] / np.sqrt(results["T_0"]) * np.sqrt(gamma / self.gas.R) \
                           * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        total_throat_area = results["mdot_total"] / throat_mass_flux
        # Assume identical nozzles in one uniformly pitched admitted sector. Multiple rectangles are hub-tangent;
        # a single rectangle is centred at Euler radius, including when its actual height differs from H_max.
        # Ellipses are centred at the annulus mean radius. Normal metal thickness uses the meanline outlet angle.
        count = turbine_meanline.no_blades_stator
        R_hub, R_tip, R_Euler = turbine_meanline.R_hub, turbine_meanline.R_tip, turbine_meanline.R_Euler
        R_mean = turbine_meanline.R_mean
        angle = results["alpha_1"]
        if not 0 < angle < np.pi / 2:
            raise ValueError("Stator outlet angle must be between zero and 90 degrees from the axial direction.")
        t_TE_stator_min = self.t_min_stator / np.cos(angle)
        delta = turbine_meanline.p_stator / R_Euler
        if count > 1 and not 0 < delta < np.pi:
            raise ValueError("The local nozzle construction requires angular pitch strictly below pi radians.")
        # Coupling quantities override any duplicate flow/scaling inputs without mutating the supplied dictionary.
        stator_arguments = {**self.SupersonicStatorNozzle_arguments,
                            "requested_outlet_absolute_flow_mach": results["M_s1_ideal"],
                            "requested_outlet_absolute_flow_angle": results["alpha_1_deg"],
                            "fluid": self.SupersonicTurbineBlading_Fluid,
                            "upstream_total_temperature": results["T_0"],
                            "upstream_total_pressure": results["p_0"],
                            "trailing_edge_thickness": turbine_meanline.t_TE_stator if count > 1 else 0.}
        contour = stator_arguments.get("contour_method", "moc")
        if contour not in ("moc", "conical"):
            raise ValueError("contour_method must be 'moc' or 'conical' (de Laval).")

        # Now separate procedures outlined at the top of this method can be followed to shape nozzles in the stator and
        # determines all remaining margins, residuals and optimality coefficients
        # If MOC stator nozzles are used
        if contour == "moc" and count > 1:
            # First calculate reference width for the MOC nozzle such that minimum thickness is always satisfied.
            # Adjacent inner corners are separated by 2*R_hub*sin(delta/2) - w*cos(delta/2).
            width = (2 * R_hub * np.sin(delta / 2) - t_TE_stator_min) / np.cos(delta / 2)
            maximum_height = self._rectangle_maximum_height(width, R_hub, R_tip)
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**{**stator_arguments, "reference_length": width})

            # Get stator blade chord (from shape uncorrected for BL)
            stator_chord = self._nozzle_chord(stator_shape) * self.stator_nozzle_chord_scale
            # Calculate throat height from total throat area, reference width and the number of nozzles
            height = total_throat_area / (count * stator_shape.dimensional_shapes.uncorrected.throat_width)
            # Calculate height optimality factor for MOC nozzle exit
            height_optimality = height - maximum_height
            # Width optimality factor and thickness margin are always zero
            width_optimality = t_TE_stator_min_margin = 0.
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the stator
            stator_ratio_residual = stator_chord / turbine_meanline.p_stator - meanline_arguments["chord_over_pitch_stator"]
            # Calculate trailing edge thickness residuals between meanline design and blade shapes for the stator
            exit_arc = self._rectangle_exit_arc(width, height, R_hub, R_Euler)
            signed_arc_gap = turbine_meanline.p_stator - exit_arc
            fourth_residual = signed_arc_gap - turbine_meanline.t_TE_stator

        # Follow the procedure if a single MOC stator nozzle is used
        elif contour == "moc":
            # Calculate width and height of the optimal nozzle exit shape to maximize area taken up by it
            # Maximize rectangle area subject to hub clearance, tip corners and centre fixed at Euler radius.
            width, maximum_height = self._single_moc_exit_dimensions(R_hub, R_tip, R_Euler)
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**{**stator_arguments, "reference_length": width})

            # Get stator blade chord (from shape uncorrected for BL)
            stator_chord = self._nozzle_chord(stator_shape) * self.stator_nozzle_chord_scale
            # From total throat area, calculate throat height
            height = total_throat_area / stator_shape.dimensional_shapes.uncorrected.throat_width
            # Calculate height optimality factor for MOC nozzle exit
            height_optimality = height - maximum_height
            # Also calculate admission fraction residual
            exit_arc = self._rectangle_exit_arc(width, height, R_Euler - height / 2, R_Euler)
            fourth_residual = exit_arc / (2 * np.pi * R_Euler) - turbine_meanline.admission_fraction
            # Width optimality factor and thickness margin are always zero
            width_optimality = t_TE_stator_min_margin = 0.
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the stator
            stator_ratio_residual = stator_chord / turbine_meanline.p_stator - meanline_arguments["chord_over_pitch_stator"]
            # Calculate trailing edge thickness residuals between meanline design and blade shapes for the stator
            signed_arc_gap = 0.

        # Follow the procedure if multiple da Laval nozzles are used
        elif count > 1:
            # First calculate throat diameter based on total throat area and number of nozzles
            throat_diameter = np.sqrt(4 * total_throat_area / (count * np.pi))
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**{**stator_arguments, "reference_length": throat_diameter})

            # Get stator blade chord (from shape uncorrected for BL)
            stator_chord = self._nozzle_chord(stator_shape) * self.stator_nozzle_chord_scale
            # Get nozzle exit width and height from stator_shape. Use stator shape uncorrected for the boundary layer
            ideal = stator_shape.dimensional_shapes.uncorrected
            width, height = ideal.nozzle_passage_pitch, ideal.nozzle_exit_width
            # Calculate height and width optimality factors
            maximum_height = R_tip - R_hub
            height_optimality = height - maximum_height
            # The ellipse support in the adjacent-centre direction gives its exact signed clearance in this plane.
            half_pitch = delta / 2
            support = np.hypot(width / 2 * np.cos(half_pitch), height / 2 * np.sin(half_pitch))
            projected_clearance = 2 * (R_mean * np.sin(half_pitch) - support)
            available_half_width_squared = ((R_mean * np.sin(half_pitch) - t_TE_stator_min / 2)**2
                                            - (height / 2 * np.sin(half_pitch))**2)
            available_width = (2 * np.sqrt(max(0., available_half_width_squared)) / np.cos(half_pitch)
                               if 2 * R_mean * np.sin(half_pitch) >= t_TE_stator_min else 0.)
            width_optimality = width - available_width
            # Calculate trailing edge thickness margin
            # Do not clip negative clearance: it measures overlap and remains useful to an outer optimizer.
            t_TE_stator_min_margin = projected_clearance - t_TE_stator_min
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the stator
            stator_ratio_residual = stator_chord / turbine_meanline.p_stator - meanline_arguments["chord_over_pitch_stator"]
            # Calculate trailing edge thickness residuals between meanline design and blade shapes for the stator
            exit_arc = self._ellipse_exit_arc(width, height, R_mean, R_Euler)
            signed_arc_gap = turbine_meanline.p_stator - exit_arc
            fourth_residual = max(0., signed_arc_gap) - turbine_meanline.t_TE_stator

        # Now follow the procedure if a single da Laval nozzle is used
        else:
            # First calculate throat diameter based on total throat area
            throat_diameter = np.sqrt(4 * total_throat_area / np.pi)
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**{**stator_arguments, "reference_length": throat_diameter})

            # Get stator blade chord (from shape uncorrected for BL)
            stator_chord = self._nozzle_chord(stator_shape) * self.stator_nozzle_chord_scale
            # Calculate height optimality factor for nozzle exit
            ideal = stator_shape.dimensional_shapes.uncorrected
            width, height = ideal.nozzle_passage_pitch, ideal.nozzle_exit_width
            maximum_height = R_tip - R_hub
            height_optimality = height - maximum_height
            # Also calculate admission fraction residual
            exit_arc = self._ellipse_exit_arc(width, height, R_mean, R_Euler)
            fourth_residual = exit_arc / (2 * np.pi * R_Euler) - turbine_meanline.admission_fraction
            # Width optimality factor and thickness margin are always zero
            width_optimality = t_TE_stator_min_margin = 0.
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the stator
            stator_ratio_residual = stator_chord / turbine_meanline.p_stator - meanline_arguments["chord_over_pitch_stator"]
            # Calculate trailing edge thickness residuals between meanline design and blade shapes for the stator
            signed_arc_gap = 0.

        # Now pack all the results and return them
        # Components 1-4 are obtained minus assumed; component 5 corrects LE to enforce the metal-thickness minimum.
        residuals = np.array([stator_ratio_residual, rotor_ratio_residual, rotor_thickness_residual,
                              fourth_residual, rotor_LE_residual])
        coefficients = {"height_optimality": height_optimality, "width_optimality": width_optimality,
                        "t_TE_stator_min_margin": t_TE_stator_min_margin,
                        "t_TE_rotor_optimality": t_TE_rotor_optimality,
                        "nozzle_exit_width": width, "nozzle_exit_height": height,
                        "nozzle_maximum_height": maximum_height, "nozzle_exit_arc": exit_arc,
                        "nozzle_pitch": turbine_meanline.p_stator, "rotor_pitch": turbine_meanline.p_rotor,
                        "stator_chord": stator_chord, "rotor_chord": rotor_chord,
                        "total_throat_area": total_throat_area,
                        "nozzle_radial_centre": (R_Euler if count == 1 else R_hub + height / 2)
                                                if contour == "moc" else R_mean,
                        "t_TE_stator_signed_arc": signed_arc_gap,
                        "t_TE_stator_obtained": max(0., signed_arc_gap),
                        "rotor_passage_pitch_mismatch": rotor_ideal.outlet_passage_pitch - rotor_ideal.inlet_passage_pitch,
                        "rotor_contour_LE_thickness": rotor_LE, "rotor_contour_TE_thickness": rotor_TE,
                        "rotor_LE_metal_thickness": rotor_metal_thicknesses[0],
                        "rotor_TE_metal_thickness": rotor_metal_thicknesses[1],
                        "M_lower_min": M_lower_min, "M_upper_max": M_upper_max}
        if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(list(coefficients.values()))):
            raise ValueError("Supersonic turbine evaluation produced nonfinite residuals or geometry.")
        return residuals, turbine_meanline, stator_shape, rotor_shape, coefficients

    def _separation_mach_limits(self, M_in, M_out):
        """Return pressure/suction separation limits from NASA TN D-4421, equations (36)-(40).

        The pressure-recovery coefficient is 1/2: (p_down-p_up)/(gamma*p_up*M_up**2/2).
        The lower surface compresses inlet flow; the upper surface compresses to outlet flow. The empirical
        criterion comes from incompressible experiments and is an approximate supersonic design limit.

        :param float M_in: Actual relative inlet Mach, greater than one.
        :param float M_out: Isentropic relative outlet Mach, greater than one.
        :return: Minimum lower-surface Mach (floored at sonic) and maximum upper-surface Mach.
        :rtype: tuple[float, float]

        Reference: https://ntrs.nasa.gov/api/citations/19680009151/downloads/19680009151.pdf
        """
        # Invert the isentropic pressure ratio after the limiting inlet-to-pressure-side compression.
        gamma = self.gas.gamma
        exponent = (gamma - 1) / gamma
        lower_squared = 2 / (gamma - 1) * ((1 + (gamma - 1) / 2 * M_in**2)
                         / (1 + gamma * M_in**2 / 4)**exponent - 1)
        # Solve the same criterion for the unknown upstream suction-surface Mach. Logs avoid high-Mach overflow.
        def upper_residual(mach):
            """Return the logarithmic upper-surface pressure-recovery residual."""
            return gamma / (gamma - 1) * (np.log1p((gamma - 1) / 2 * mach**2)
                    - np.log1p((gamma - 1) / 2 * M_out**2)) - np.log1p(gamma * mach**2 / 4)

        upper_bound = 2 * M_out
        while upper_residual(upper_bound) < 0:
            upper_bound *= 2
        return np.sqrt(max(1., lower_squared)), brentq(upper_residual, M_out, upper_bound, xtol=1e-12)

    @staticmethod
    def _nozzle_chord(nozzle):
        """Return uncorrected throat-centre to exit-centre straight distance, m.

        Stored nozzle coordinates follow the nozzle axis. The exit centre averages the two wall endpoints,
        including the suction-side straight extension needed to terminate the exit in the machine axial plane.
        """
        # Average the two endpoints of each opening before measuring the actual chord.
        ideal = nozzle.dimensional_shapes.uncorrected
        pressure, suction = ideal.pressure_surface, ideal.suction_surface
        return np.hypot((pressure.x[-1] + suction.x[-1] - pressure.x[0] - suction.x[0]) / 2,
                        (pressure.y[-1] + suction.y[-1] - pressure.y[0] - suction.y[0]) / 2)

    @staticmethod
    def _rectangle_maximum_height(width, R_hub, R_tip):
        """Return maximum radial height of a hub-tangent rectangle whose outer corners touch the tip, m."""
        # The radial inner side is tangent to the hub; the tip circle limits both outer corners.
        if not np.isfinite(width) or width <= 0 or width / 2 >= np.sqrt(R_tip**2 - R_hub**2):
            raise ValueError("No positive hub-tangent MOC exit fits this width; reduce nozzle pitch or metal thickness.")
        return np.sqrt(R_tip**2 - width**2 / 4) - R_hub

    @staticmethod
    def _single_moc_exit_dimensions(R_hub, R_tip, R_Euler):
        """Return width and maximum height of the largest Euler-centred rectangular exit, m."""
        # For half-height h, tip contact gives w=2*sqrt(R_tip**2-(R_Euler+h)**2).
        # The area derivative gives 2*h**2+3*R_Euler*h+R_Euler**2-R_tip**2=0; the hub caps h.
        if not np.all(np.isfinite([R_hub, R_tip, R_Euler])) or not 0 < R_hub < R_Euler < R_tip:
            raise ValueError("Single-MOC radii must satisfy 0 < R_hub < R_Euler < R_tip.")
        half_height = min(R_Euler - R_hub, 2 * (R_tip**2 - R_Euler**2)
                          / (np.sqrt(R_Euler**2 + 8 * R_tip**2) + 3 * R_Euler))
        return 2 * np.sqrt(R_tip**2 - (R_Euler + half_height)**2), 2 * half_height

    @staticmethod
    def _rectangle_exit_arc(width, height, R_hub, R_Euler):
        """Return Euler-circle arc inside a rectangular exit, m, requiring crossings on both radial sides."""
        # Exclude large-nozzle cases in which the circle crosses a tangential side or misses the rectangle.
        if width <= 0 or width >= 2 * R_Euler:
            raise ValueError("MOC exit width must be positive and smaller than the Euler diameter.")
        intersection_radius = np.sqrt(R_Euler**2 - width**2 / 4)
        if not R_hub < intersection_radius < R_hub + height:
            raise ValueError("Euler circle must cross the radial sides of the MOC exit; adjust admission or nozzle count.")
        return 2 * R_Euler * np.arcsin(width / (2 * R_Euler))

    @staticmethod
    def _ellipse_exit_arc(width, height, R_mean, R_Euler):
        """Return Euler-circle arc inside a mean-radius-centred ellipse, m, for a local nozzle."""
        # Require the Euler circle to pass through the central portion and cross the flanks of the exit.
        a, b = width / 2, height / 2
        if min(a, b) <= 0 or a >= R_Euler or abs(R_Euler - R_mean) >= b:
            raise ValueError("Euler circle must cross a local de Laval exit centred at mean radius.")
        # Height-only radial fitting assumes the outermost ellipse point is its radial top, not a wide flank.
        if a**2 - b**2 > R_mean * b:
            raise ValueError("De Laval exit is too wide for the local annulus model: a**2-b**2 must not exceed R_mean*b.")
        # Intersect x=R_Euler*sin(theta), y=R_Euler*cos(theta) with x**2/a**2+(y-R_mean)**2/b**2=1.
        def intersection(theta):
            """Return the ellipse equation evaluated on the Euler circle."""
            return (R_Euler * np.sin(theta) / a)**2 + ((R_Euler * np.cos(theta) - R_mean) / b)**2 - 1

        return 2 * R_Euler * brentq(intersection, 0., np.arcsin(a / R_Euler), xtol=1e-14)
