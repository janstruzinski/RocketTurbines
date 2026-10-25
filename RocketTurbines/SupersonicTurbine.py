from IdealGas import IdealGas
from Turbine1D import Turbine1D
from TraupelLossModel import TraupelLossModel
import SupersonicTurbineBlading as blading
from RelaxedAitken import solve_with_relaxed_aitken
import numpy as np
from scipy.optimize import brentq
import warnings
from typing import Any, cast

class SupersonicTurbine:
    def __init__(self, gas: IdealGas, loss_model: TraupelLossModel, Turbine1D_arguments,
                 SupersonicStatorNozzle_arguments, SupersonicRotorBlades_arguments, separation_margin,
                 t_min_stator, t_min_rotor, stator_nozzle_chord_scale=1.0, print_user_warnings=True):
        """SupersonicTurbine is the class to represent a supersonic turbine, with consistent meanline design and blading
         shape. It ensures that chord/pitch ratios and other variables used in Turbine1D agrees with the values obtained from
         SupersonicTurbineBlading objects, which in turn also use Turbine1D meanline results. Thus, there is a coupling
          between meanline and blading design, which couple_meanline_and_blading solves numerically when called.
           Construction only prepares inputs and leaves the design results as None. This object also checks if the
           stator nozzles exit can fit optimally in the turbine annulus by producing continuous coefficients than can be
            either checked by the engineer or constrained by outer optimizer.

        :param IdealGas gas: Object representing calorically perfect working gas.
        :param TraupelLossModel loss_model: Object representing meanline loss model.
        :param dict Turbine1D_arguments: Arguments to Turbine1D.size_turbine, including explicit chord/pitch ratios,
         admission_fraction and initial estimates for design variables calculated in this class. In case a single
          stator nozzle is used, admission_fraction is its initial estimate; its final value is solved from the exit
           geometry like other design variables.
        :param dict SupersonicStatorNozzle_arguments: Arguments for SupersonicStatorNozzle, including contour_method.
         Exit Mach number and flow angle, reference_length and trailing_edge_thickness are calculated in this class.
        :param dict SupersonicRotorBlades_arguments: Arguments for SupersonicRotorBlades. Inlet & outlet Mach numbers
         and flow angles, surface Mach numbers, blade count, radius and speed are supplied by this.
          leading_edge_thickness_over_total_pitch is a required initial LE thickness estimate.
        :param float separation_margin: Flow separation margin, strictly between zero and one. A value of 0.05 reserves
         5% of the Mach interval bounded by Goldman and Scullin's separation limits
          (NASA TN D-4421, equations (36)-(40)).
        :param float t_min_stator: Minimum true stator metal thickness, m; nonnegative.
        :param float t_min_rotor: Minimum true rotor LE/TE metal thickness, m; nonnegative.
        :param float stator_nozzle_chord_scale: Positive multiplier of throat-centre to exit-centre chord,
         to account for the upstream turning section, which SupersonicStatorNozzle does not calculate. Default 1.
        :param bool print_user_warnings: Whether to emit this object's user warnings. Default True.
         Traupel extrapolation warnings are controlled separately by loss_model.warn_on_extrapolation.
        :ivar numpy.ndarray x_solution: None until coupling converges; then [stator chord/pitch, rotor chord/pitch, rotor TE thickness,
         stator TE thickness (or single-nozzle admission), rotor LE thickness]; thicknesses are circumferential projections in m.
        :ivar dict coefficients: None until coupling converges; then the optimality coefficients and thickness margins.
        :raises ValueError: If construction inputs are invalid.
        :raises KeyError: If a required input is missing from the argument dictionaries.
        """

        # Meanline - nozzle/blade shapes coupling logic diagram:
        #  SupersonicTurbine.__init__
        #    |
        #    v
        #  Copy input dictionaries; validate inputs and create the shared Fluid object.
        #  Disable blading Mach matching to avoid counting meanline losses twice.
        #  Initialize design results to None; return without solving.
        #    |
        #    v  (when the user calls couple_meanline_and_blading())
        #  SupersonicTurbine.couple_meanline_and_blading
        #    |
        #    v
        #  Assemble five unknowns x and their initial estimates:
        #    [stator chord/pitch, rotor chord/pitch, rotor TE thickness, stator TE thickness or admission, rotor LE thickness].
        #    Multiple nozzles: solve stator TE thickness; one nozzle: fix stator TE thickness = 0 and solve admission.
        #  Calculate Euler-radius pitches, positive component scales s and solution bounds.
        #    |
        #    +--> COUPLING LOOP: bounded relaxed Aitken ------------------------------------------------------+
        #    |      Iterate normalized unknowns z = x/s                                                       |
        #    |      |                                                                                         |
        #    |      v                                                                                         |
        #    |    normalized_mapping(z)                                                                       |
        #    |      |  Convert z to dimensional/mixed-unit unknowns x = s*z.                                  |
        #    |      v                                                                                         |
        #    |    calculate_new_solution(x)                                                                   |
        #    |      |  Copy meanline arguments and insert chord/pitch ratios, TE thicknesses and admission.   |
        #    |      |  Pass the independent rotor LE thickness to evaluate_design.                            |
        #    |      v                                                                                         |
        #    |    SupersonicTurbine.evaluate_design                                                           |
        #    |      |                                                                                         |
        #    |      +--> Turbine1D.size_turbine                                                               |
        #    |      |      Solve its nested entropy-rise and enthalpy-ratio loops.                            |
        #    |      |      Obtain flow states, blade-row angles, annulus radii and Euler-radius pitches.      |
        #    |      |                                                                                         |
        #    |      +--> Rotor shaping                                                                        |
        #    |      |      Apply Goldman--Scullin separation limits and separation_margin.                    |
        #    |      |      Construct SupersonicRotorBlade using the blade-row outlet angle.                   |
        #    |      |      Pass rotor LE thickness/total pitch; pitch includes metal at Euler radius.         |
        #    |      |      Measure LE/TE thickness projections and the actual LE-to-TE chord.                 |
        #    |      |      Store the obtained LE thickness in the meanline object.                            |
        #    |      |      Use flow angles to estimate minimum metal thickness at both rotor edges.           |
        #    |      |                                                                                         |
        #    |      +--> Stator shaping and placement                                                         |
        #    |      |      Calculate the total choked throat area required by mass flow.                      |
        #    |      |      |                                                                                  |
        #    |      |      +--> Multiple MOC nozzles                                                          |
        #    |      |      |      Determine nozzle placement.                                                 |
        #    |      |      |      Calculate MOC nozzle exit width based on minimum thickness.                 |
        #    |      |      |      Mass flow sets nozzle height.                                               |
        #    |      |      |                                                                                  |
        #    |      |      +--> Single MOC nozzle                                                             |
        #    |      |      |      Determine nozzle placement.                                                 |
        #    |      |      |      Calculate MOC nozzle exit width to maximize area.                           |
        #    |      |      |      Mass flow sets nozzle height.                                               |
        #    |      |      |                                                                                  |
        #    |      |      +--> Multiple de Laval nozzles                                                     |
        #    |      |      |      Throat area per nozzle sets throat diameter and nozzle reference scale.     |
        #    |      |      |      Place elliptical exits at mean radius; calculate metal thickness.           |
        #    |      |      |                                                                                  |
        #    |      |      +--> Single de Laval nozzle                                                        |
        #    |      |             Total throat area sets throat diameter and nozzle reference scale.          |
        #    |      |             Place the elliptical exit at mean radius                                    |
        #    |      |                                                                                         |
        #    |      +--> Shared stator measurements                                                           |
        #    |             Measure the Euler-circle arc through the nozzle exit.                              |
        #    |             Match stator TE thickness arc for multiple nozzles or admission for one nozzle.    |
        #    |             Measure throat-centre to exit-centre chord; apply stator_nozzle_chord_scale.       |
        #    |             Calculate height/width excesses and minimum stator-thickness margin.               |
        #    |             Store the actual exit-centre radius as turbine_meanline.R_nozzle.                 |
        #    |      |                                                                                         |
        #    |      v                                                                                         |
        #    |    Return five residuals r(x), meanline/blading objects and geometric coefficients:            |
        #    |      r1/r2: obtained minus assumed stator/rotor chord-to-pitch ratios.                         |
        #    |      r3: uncorrected contour rotor TE thickness minus meanline TE thickness.                   |
        #    |      r4: stator TE thickness mismatch (multiple nozzles) or admission mismatch (one nozzle).   |
        #    |      r5: rotor LE thickness correction enforcing the minimum normal LE/TE metal thickness.     |
        #    |      |                                                                                         |
        #    |      v                                                                                         |
        #    |    calculate_new_solution: F(x) = x + r(x).                                                    |
        #    |    normalized_mapping: G(z) = F(s*z)/s.                                                        |
        #    |      |                                                                                         |
        #    |      v                                                                                         |
        #    |    Aitken checks every finite normalized residual G(z)-z against xtol.                         |
        #    |    If not converged: adapt the shared relaxation factor and clip the next z to bounds.         |
        #    |      |                                                                                         |
        #    |      +--> Repeat with updated z ---------------------------------------------------------------+
        #    |
        #    v  (after coupling converges; Aitken raises an error if it cannot converge)
        #  Evaluate the solved design again to obtain the final objects and coefficients.
        #  Store turbine_meanline, stator_shape and rotor_shape.
        #  Store x_solution and the coefficients dictionary with optimality coefficients and thickness margins.
        #    |
        #    v
        #  Engineer / outer optimizer checks nozzle packing:
        #    height_optimality <= 0; width_optimality <= 0; t_TE_stator_min_margin >= 0.
        #  Adjust nozzle count, flow coefficients or other free inputs and repeat as required.
        #  Coupling convergence establishes meanline/blading consistency; packing margins can remain infeasible.

        # Store objects
        self.gas = gas
        self.loss_model = loss_model
        self.print_user_warnings = print_user_warnings

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

        # Validate manufacturing inputs and store the optional upstream chord scale that accounts for flow turning
        # upstream of the throat.
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
            if option in arguments and arguments.pop(option):
                if self.print_user_warnings:
                    warnings.warn(f"{option} is disabled: matching mixed Mach would count meanline losses twice.",
                                  UserWarning, stacklevel=2)

        # Initialize results that will be assigned after the meanline and blading coupling converges.
        self.x_solution = None
        self.turbine_meanline = None
        self.stator_shape = None
        self.rotor_shape = None
        self.coefficients = None

    def couple_meanline_and_blading(self):
        """Solve the meanline/blading coupling and store the converged design objects and coefficients.

        Use the stored input dictionaries for the initial estimates. Results remain unset until the solve succeeds;
        if a later solve fails, the previously stored results are retained.

        :raises ValueError: If an evaluated design violates the flow or construction assumptions.
        :raises KeyError: If a required input is missing from the argument dictionaries.
        :raises RuntimeError: If the bounded Aitken solver does not converge to its normalized tolerance of 1e-3.
        """

        # Determine which stator variable is coupled (trailing edge thickness for multiple nozzles, admission for one)
        # based on the number of stator nozzles
        no_nozzles = self.Turbine1D_arguments["no_blades_stator"]

        # Define a function that calculates new values of design variables from evaluate_design based on assumed ones
        def calculate_new_solution(x_0):
            """Return the five obtained design variables and their associated design objects."""

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
                x_1 = np.add(x_0, residuals)

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
                x_1 = np.add(x_0, residuals)

            # Return new solution estimate together with remaining results
            return x_1, turbine_meanline, stator_shape, rotor_shape, coefficients

        # Create initial solution array if there are multiple nozzles
        x_initial: Any
        if no_nozzles != 1:
            x_initial = np.array([self.Turbine1D_arguments["chord_over_pitch_stator"],
                                  self.Turbine1D_arguments["chord_over_pitch_rotor"],
                                  self.Turbine1D_arguments["t_TE_rotor"],
                                  self.Turbine1D_arguments["t_TE_stator"]], dtype=float)
        # If there is just one nozzle
        else:
            x_initial = np.array([self.Turbine1D_arguments["chord_over_pitch_stator"],
                                  self.Turbine1D_arguments["chord_over_pitch_rotor"],
                                  self.Turbine1D_arguments["t_TE_rotor"],
                                  self.Turbine1D_arguments["admission_fraction"]], dtype=float)

        # Numerical solver is used that will create a consistent design, in which values assumed for sizing
        # in Turbine1D are equivalent to those obtained from SupersonicTurbineBlading. Aitken's scheme is used for this.
        # First, use fixed Euler-radius pitches to bound thicknesses and normalize length and dimensionless variables.
        arguments = self.Turbine1D_arguments
        radius = np.sqrt(arguments["shaft_power"] / arguments["mdot"] / arguments["work_coefficient_tt"]) \
                 / (arguments["RPM"] * np.pi / 30.)
        rotor_pitch = 2 * np.pi * radius / arguments["no_blades_rotor"]
        # Stator thickness uses the total pitch within the prescribed admitted sector at Euler radius.
        stator_pitch = arguments["admission_fraction"] * 2 * np.pi * radius / no_nozzles
        # Rotor LE thickness is independent of the TE thickness used by the meanline loss model.
        initial_t_LE_rotor = self.SupersonicRotorBlades_arguments[
            "leading_edge_thickness_over_total_pitch"] * rotor_pitch
        # Assemble initial solution array
        x_initial = np.append(x_initial, initial_t_LE_rotor)
        # Create upper and lower bounds for the solution; thicknesses cannot exceed 24% of their total pitch.
        lower = np.array([1e-8, 1e-8, 0., 0. if no_nozzles > 1 else 1e-6, 0.])
        upper = np.array([np.inf, np.inf, 0.24 * rotor_pitch,
                          0.24 * stator_pitch if no_nozzles > 1 else 1., 0.24 * rotor_pitch])
        # Create solution scales for normalizing it. Solution is normalized because it mixes nondimensional &
        # dimensional variables. Use 12% of total pitch for each thickness scale.
        # Use finite substitutes when a dimensionless initial estimate is needed to construct its scale.
        scale_estimates = np.nan_to_num(x_initial, nan=1., posinf=1., neginf=0.)
        solution_scales = np.array([max(scale_estimates[0], 1.), max(scale_estimates[1], 1.), 0.12 * rotor_pitch,
                                    0.12 * stator_pitch if no_nozzles > 1 else np.clip(scale_estimates[3], 0.01, 1.),
                                    0.12 * rotor_pitch])
        # Replace nonfinite or out-of-bound initial estimates with their corresponding solution scales and notify
        # the user.
        invalid_initial = ~np.isfinite(x_initial) | (x_initial < lower) | (x_initial > upper)
        if np.any(invalid_initial):
            if self.print_user_warnings:
                warnings.warn("Nonfinite or out-of-bound initial coupling estimates were replaced by their"
                              " solution scales.", UserWarning, stacklevel=2)
            x_initial[invalid_initial] = solution_scales[invalid_initial]

        # Define a function to extract new design variables from calculate_new_solution based on their assumed,
        # normalized values. These new values are also normalized by the solution scales.
        def normalized_mapping(values):
            """Normalized mapping to and from calculate_new_solution function"""
            return np.divide(calculate_new_solution(np.multiply(values, solution_scales))[0], solution_scales)

        # Solve the coupling with bounded relaxed Aitken. Bounds and initial solution are scaled. Since the solver
        # calls normalized_mapping which also operates on normalized values, x_scale argument for
        # solve_with_relaxed_aitken can be default one. However, the final solution must be scaled to obtain it in
        # desirable dimensions.
        x_solution = np.multiply(solution_scales, solve_with_relaxed_aitken(
            normalized_mapping, np.divide(x_initial, solution_scales), initial_relaxation=0.3,
            min_relaxation=0.05, max_relaxation=1.5, xtol=1e-3, maxiter=60,
            bounds=(np.divide(lower, solution_scales), np.divide(upper, solution_scales))))

        # Obtain consistent design objects after solution was found
        x_converged, self.turbine_meanline, self.stator_shape, self.rotor_shape, coefficients = \
            calculate_new_solution(x_solution)

        # Save solution vector
        self.x_solution = x_converged
        # Save optimality coefficients & thickness margins
        self.coefficients = coefficients

    def print_coefficients(self, coefficients=None):
        """Print optimality coefficients and thickness margins with their feasible sign conditions.

        :param dict coefficients: Coefficients returned by evaluate_design; use self.coefficients when omitted.
        """
        # Use converged coefficients unless a dictionary from a manual design evaluation was supplied.
        if coefficients is None:
            coefficients = self.coefficients
        # Report that no coupled result has been stored yet.
        if coefficients is None:
            print("SupersonicTurbine.coefficients is None")
            return
        # Packing excesses must be nonpositive; obtained-minus-required thickness margins must be nonnegative.
        conditions = {"height_optimality": "<= 0 (nonpositive)", "width_optimality": "<= 0 (nonpositive)",
                      "t_TE_stator_min_margin": ">= 0 (nonnegative)", "t_rotor_min_margin": ">= 0 (nonnegative)"}
        # Print each supplied coefficient and the sign condition for a feasible design.
        for name, value in coefficients.items():
            print(f"{name}: {value}; feasible: {conditions[name]}")

    def evaluate_design(self, meanline_arguments, t_LE_rotor):
        """This method allows to calculate the residuals between assumed values for 1D meanline calculations in
        Turbine 1D and obtained values from blade shaping with SupersonicTurbineBlading. These residuals can be driven
         to zero by couple_meanline_and_blading or the engineer to obtain consistent design. This method also returns
          some remaining residuals and optimality coefficients that can be constrained by outer optimizer or which can
           inform the engineer about manufacturability & optimality of the supersonic turbine.

        :param dict meanline_arguments: Dictionary with size_turbine inputs.
        :param float t_LE_rotor: Circumferential LE thickness (m), excluding BL correction.
        :return: Five residuals, meanline object, stator object, rotor object and optimality/margin coefficients.
         Residuals 1/2 are dimensionless chord/pitch differences. Residual 3 is geometric-minus-meanline rotor
         TE thickness (m). Residual 4 is stator arc thickness (m), or admission fraction for one nozzle.
         Residual 5 is the LE thickness correction enforcing the minimum true metal thickness at both rotor edges (m).
         height_optimality and width_optimality are dimensional excesses (m); positive values indicate overfilling.
         t_TE_stator_min_margin is signed projected clearance minus the required projected metal thickness (m).
         Multiplying this margin by cos(alpha_1) gives the approximate true metal-thickness margin.
         t_rotor_min_margin is the smaller rotor LE/TE normal metal thickness minus t_min_rotor (m).
         All geometric calculations use shapes without boundary layer correction.
        :rtype: tuple[numpy.ndarray, Turbine1D, blading.SupersonicStatorNozzle, blading.SupersonicRotorBlade, dict]
        :raises ValueError: If the flow is not supersonic or the construction assumptions are violated.
        :raises KeyError: If a required input is missing from the argument dictionaries.
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
        # t_rotor_min_margin = t_TE_rotor_min_obtained - t_min_rotor. To ensure fair comparison,
        # the engineer or outer optimizer should always compare rotors with the same metal thickness. Consequently,
        # the numerical solver in this class will also ensure that t_rotor_min_margin is zero by varying t_LE_rotor.
        # An independent residual, t_TE_rotor_residula matches the meanline t_TE_rotor to the actual contour TE thickness.
        # To convert stator TE thickness and rotor LE/TE thickness projections to normal metal thicknesses, flow angles
        # from meanline design are always used. These are approximately the same as nozzle/rotor blade metal angles.
        # Blade metal angles can be a few degrees different if the user specifies in input arguments to
        # SupersonicTurbineBlading objects that they should be adjusted to obtain flow angles equal to those from
        # meanline design, however this has a negligible influence. Finally, boundary layer correction for stator and
        # rotor shapes can lead to blade thickness for BL corrected shapes to be different from t_TE_stator_min_obtained
        # or t_TE_rotor_min_obtained. SupersonicRotorBlading takes an LE thickness input; prescribed endpoint
        # states need not close the inviscid passage pitch, so the actual contour TE thickness is measured independently.

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
        # satisfied for given annulus dimensions, admission fraction, number of nozzles and t_TE_stator_min when the
        # exit is tangent to the hub. Actual exit centres start at Euler radius. Then, maximum possible nozzle height,
        # H_max, places the inner side tangent to the hub and outer corners at the tip.
        # If no positive height fits this width, H_max is zero. w_nozzle dimensionalizes MOC shape in
        # SupersonicStatorNozzle, which in turn settles throat width. Throat area can be calculated such that point 4 is
        # satisfied. Throat area and width are known, so its height can be calculated too. It is the same as height of
        # the nozzle exit. This height, H_nozzle, must be smaller or equal to already calculated H_max - otherwise the
        # nozzle cannot fit in the annulus without violating point 3. Consequently, nozzle optimality factor,
        # height_optimality is H_nozzle - H_max. If it is positive, geometry is not feasible. If it is zero,
        # the geometry is optimal as nozzle exit has the largest possible area and matches annulus well. This satisfies
        # point 1. If it is negative, the geometry is feasible, but may be not optimal, especially for large negative
        # values. If the Euler-centred exit exceeds the tip, shift its centre inward until the corners touch the tip.
        # At H_max, the inner side also touches the hub. For greater heights, retain hub tangency and allow the exit to
        # cross tip radius. In such case height_optimality is positive and the engineer or outer optimizer need to adjust the design.
        # Minimum spacing uses the actual inner corners and is at least t_TE_stator_min, with zero margin at H_max.
        # Finally, the arc length at Euler diameter between edges of adjacent nozzles, t_TE_stator_obtained,
        # can be calculated. It must be equal to the arc length t_TE_stator assumed during meanline calculations.
        # Their difference is t_TE_stator_residual and the numerical solver must hold it at zero - it means that assumed
        # and obtained trailing edge thicknesses are consistent, and that meanline code and nozzle shapes agree with
        # each other. This satisfies point 2.

        # This was a procedure when multiple MOC nozzles are used. If there is a single MOC nozzle, a few things change.
        # Firstly, points 2 and 3 go away, as t_TE_stator for the meanline code is always zero and there is no adjecent
        # nozzle for the engineer to worry about metal thickness. Point 1 sets the dimensions of the rectangle with
        # greatest area within the tip circle, with its centre constrained to Euler radius, to determine w_nozzle.
        # Keeping that width, calculate H_max for a hub-tangent rectangle with its outer corners touching the tip.
        # The actual exit starts centred at Euler radius and shifts by the same rule as for multiple MOC exits.
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
        # satisfy point 1. The ellipse can reach furthest from the turbine axis on a flank rather than at its radial top.
        # Thus, height_optimality is the difference between the maximum radius that bounds the ellipse and tip radius,
        # R_max - R_tip. Its positive, zero and negative values
        # have the same meaning as height_optimality for MOC nozzle. Elliptical shape of the nozzle exit is just a
        # cross-section of a cylinder with diameter D_nozzle with the annulus plane at the stator
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

        # First, modify meanline dictionary if needed
        # One nozzle has no adjacent trailing-edge wall, so blade thickness is set to zero
        if meanline_arguments["no_blades_stator"] == 1:
            meanline_arguments["t_TE_stator"] = 0.
        # Obtain meanline analysis results from Turbine1D
        turbine_meanline = Turbine1D()
        turbine_meanline.size_turbine(**meanline_arguments)
        # Read the actual inlet and isentropic rotor outlet reference states.
        results = turbine_meanline.analysis_results_at_design_point
        M_in, M_out = results["M_r1"], results["M_r2_ideal_r_real_s"]
        if min(M_in, M_out, results["M_s1_ideal"]) <= 1:
            raise ValueError("Supersonic blading requires M_r1, M_r2_ideal_r_real_s and M_s1_ideal > 1.")
        # Shape the blade row using its flow angle at its outlet.
        rotor_outlet_angle = turbine_meanline.blade_row_results_at_design_point["beta_2_blade"]

        # Now calculate rotor surface Mach numbers based on separation margins
        # First lower Mach number
        M_lower_min, M_upper_max = self._calculate_separation_mach_limits(M_in, M_out)
        M_lower = M_in - (1 - self.separation_margin) * (M_in - M_lower_min)
        # Now upper Mach number
        M_upper = M_out + (1 - self.separation_margin) * (M_upper_max - M_out)
        # An accelerating/decelerating rotor must also bracket both endpoint Mach numbers.
        if not 1 <= M_lower < min(M_in, M_out) or M_upper <= max(M_in, M_out):
            raise ValueError("Separation-margin surface Mach numbers do not bracket both rotor endpoint Mach numbers.")

        # Now calculate rotor shape and dimensions. Use real Mach number at station 1 from Turbine 1D and Mach number
        # at station 2 that assumes isentropic expansion through the rotor
        # Validate the supplied rotor LE thickness before constructing the shape.
        if not np.isfinite(t_LE_rotor) or not 0 <= t_LE_rotor < turbine_meanline.p_rotor:
            raise ValueError("t_LE_rotor must be finite and in [0, rotor pitch).")
        # Create dictionary with arguments
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
        # Construct rotor shape
        rotor_shape = blading.SupersonicRotorBlade(**rotor_arguments)

        # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the rotor
        # The two stored surfaces bound a passage. Their endpoint averages give the vector from LE to TE
        # after translating the suction side by one total pitch; the constant translation cancels in the difference.
        # Length of that vactor is blade chord.
        rotor_ideal = rotor_shape.dimensionalize().uncorrected
        pressure_surface, suction_surface = rotor_ideal.pressure_surface, rotor_ideal.suction_surface
        rotor_chord = np.hypot(
            (pressure_surface.x[-1] + suction_surface.x[-1] - pressure_surface.x[0] - suction_surface.x[0]) / 2,
            (pressure_surface.y[-1] + suction_surface.y[-1] - pressure_surface.y[0] - suction_surface.y[0]) / 2)
        rotor_ratio_residual = rotor_chord / turbine_meanline.p_rotor - meanline_arguments["chord_over_pitch_rotor"]

        # Perform calculations related to rotor LE & TE thicknesses
        # First subtract open passages from total Euler-radius pitch to obtain blade edge projections on
        # circumferential direction.
        t_LE_rotor = turbine_meanline.p_rotor - rotor_ideal.inlet_passage_pitch
        t_TE_rotor = turbine_meanline.p_rotor - rotor_ideal.outlet_passage_pitch
        # Store the obtained projected LE thickness directly in the returned meanline object.
        turbine_meanline.t_LE_rotor = t_LE_rotor
        # Residual calculated the difference between obtained TE thickness and the TE thickness used for meanline
        # design.
        t_TE_rotor_residual = t_TE_rotor - turbine_meanline.t_TE_rotor
        # Calculate minimum thicknesses
        rotor_cosines = np.cos([results["beta_1"], rotor_outlet_angle])
        if np.min(rotor_cosines) <= 0:
            raise ValueError("Rotor inlet/outlet angles must have positive axial components.")
        rotor_metal_thicknesses = np.multiply([t_LE_rotor, t_TE_rotor], rotor_cosines)
        # Use the smaller one to constraint the design
        limiting_edge = np.argmin(rotor_metal_thicknesses)
        # Calculate the minimum normal-thickness margin across LE and TE.
        t_rotor_min_margin = rotor_metal_thicknesses[limiting_edge] - self.t_min_rotor
        # Calculate LE thickness residual: LE thickness is the shaping input for SupersonicRotorBlade, and changing it also
        # changes the obtained TE thickness through the passage scale and geometry. Thus, either edge's
        # minimum-thickness margin can drive LE thickness until the thinner edge reaches t_min_rotor. The cosine converts normal
        # to circumferential thickness. When TE is the thinner edge, this assumes that changing LE thickness changes
        # TE thickness by the same amount.
        rotor_LE_residual = -t_rotor_min_margin / rotor_cosines[limiting_edge]

        # Calculate total throat area needed to satisfy massflow requirements
        gamma = self.gas.gamma
        stator_throat_mass_flux = results["p_0"] / np.sqrt(results["T_0"]) * np.sqrt(gamma / self.gas.R) \
                           * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        A_throat_stator = results["mdot_total"] / stator_throat_mass_flux
        # Store the total stator throat area required to pass the admitted mass flow.
        turbine_meanline.A_throat_stator = A_throat_stator

        # Measure the narrowest gap between the uncorrected rotor passage walls at Euler radius.
        rotor_throat_width = self._calculate_rotor_throat_width(rotor_ideal)
        # Assume geometrically similar sections, with throat width proportional to radius: w(r)=w_Euler*r/R_Euler.
        # Integrate this width from hub to tip, then include one passage per blade and the admitted fraction.
        turbine_meanline.A_throat_rotor = (rotor_throat_width * turbine_meanline.no_blades_rotor
                                           * turbine_meanline.admission_fraction
                                           * (turbine_meanline.R_tip**2 - turbine_meanline.R_hub**2)
                                           / (2 * turbine_meanline.R_Euler))

        # Assume identical nozzles in one uniformly pitched admitted sector. Rectangles start at Euler radius
        # and shift radially when required to fit the nozzle in the turbine annulus.
        # Ellipses are centred at the annulus mean radius. Normal metal thickness uses the meanline flow outlet angle,
        # which can be assumed to be equal to the outlet blade angle.
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
        # Assemble dictionary with stator shape input arguments. Calculated quantities override any duplicate inputs
        stator_arguments = {**self.SupersonicStatorNozzle_arguments,
                            "requested_outlet_absolute_flow_mach": results["M_s1_ideal"],
                            "requested_outlet_absolute_flow_angle": results["alpha_1_deg"],
                            "fluid": self.SupersonicTurbineBlading_Fluid,
                            "upstream_total_temperature": results["T_0"],
                            "upstream_total_pressure": results["p_0"],
                            "trailing_edge_thickness": turbine_meanline.t_TE_stator if count > 1 else 0.}
        contour = stator_arguments["contour_method"]
        if contour not in ("moc", "conical"):
            raise ValueError("contour_method must be 'moc' or 'conical' (de Laval).")

        # Now separate procedures outlined at the top of this method can be followed to shape nozzles in the stator and
        # determine all remaining margins, residuals and optimality coefficients
        # If MOC stator nozzles are used
        if contour == "moc" and count > 1:
            # Set width so an inward extension to the hub tangent line has exactly the required minimum thickness.
            # Adjacent reference inner corners are separated by 2*R_hub*sin(delta/2) - w*cos(delta/2).
            width = (2 * R_hub * np.sin(delta / 2) - t_TE_stator_min) / np.cos(delta / 2)
            maximum_height = self._calculate_rectangle_maximum_height(width, R_hub, R_tip)
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**{**stator_arguments, "reference_length": width})

            # Get stator blade chord (from shape uncorrected for BL)
            stator_chord = self._calculate_nozzle_chord(stator_shape) * self.stator_nozzle_chord_scale
            # Calculate throat height from total throat area, reference width and the number of nozzles
            height = A_throat_stator / (count * stator_shape.dimensional_shapes.uncorrected.throat_width)
            # Calculate height optimality factor for MOC nozzle exit
            height_optimality = height - maximum_height
            # Place the nozzle at Euler radius first. If for calculated height it crosses the tip radius,
            # shift nozzles inward instead. When H is even greater and exceeds H_max, retain hub tangency and allow the
            # nozzle exit to cross the tip radius.
            inner_side_radius = max(R_hub, min(R_Euler - height / 2, R_hub + maximum_height - height))
            # Moving the inner side outward from the reference adds 2*(inner_side_radius-R_hub)*sin(delta/2) to its gap.
            # This equivalent gap formula avoids cancellation at the prescribed minimum thickness.
            t_TE_stator_min_obtained = t_TE_stator_min + 2 * (inner_side_radius - R_hub) * np.sin(delta / 2)
            t_TE_stator_min_margin = t_TE_stator_min_obtained - t_TE_stator_min
            # Width remains fixed by the hub-tangent reference construction.
            width_optimality = 0.
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the stator
            stator_ratio_residual = \
                stator_chord / turbine_meanline.p_stator - meanline_arguments["chord_over_pitch_stator"]
            # Calculate trailing edge thickness residuals between meanline design and blade shapes for the stator
            exit_arc = self._calculate_rectangle_exit_arc(width, height, inner_side_radius, R_Euler)
            signed_arc_gap = turbine_meanline.p_stator - exit_arc
            t_TE_stator_residual = signed_arc_gap - turbine_meanline.t_TE_stator

        # Follow the procedure if a single MOC stator nozzle is used
        elif contour == "moc":
            # Calculate width and height of the optimal nozzle exit shape to maximize area taken up by it
            # Choose width by maximizing rectangle area inside the tip circle with its centre fixed at Euler radius.
            width, _ = self._calculate_single_moc_exit_dimensions(R_tip, R_Euler)
            # At this width, the largest fitting height places the inner side at the hub and outer corners at the tip.
            maximum_height = self._calculate_rectangle_maximum_height(width, R_hub, R_tip)
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**{**stator_arguments, "reference_length": width})

            # Get stator blade chord (from shape uncorrected for BL)
            stator_chord = self._calculate_nozzle_chord(stator_shape) * self.stator_nozzle_chord_scale
            # From total throat area, calculate throat height
            height = A_throat_stator / stator_shape.dimensional_shapes.uncorrected.throat_width
            # Calculate height optimality factor for MOC nozzle exit
            height_optimality = height - maximum_height
            # Place the nozzle at Euler radius first. If for calculated height it crosses the tip radius,
            # shift nozzles inward instead. When H is even greater and exceeds H_max, retain hub tangency and allow the
            # nozzle exit to cross the tip radius.
            inner_side_radius = max(R_hub, min(R_Euler - height / 2, R_hub + maximum_height - height))
            # Also calculate admission fraction residual
            exit_arc = self._calculate_rectangle_exit_arc(width, height, inner_side_radius, R_Euler)
            admission_fraction_residual = exit_arc / (2 * np.pi * R_Euler) - turbine_meanline.admission_fraction
            # Width optimality factor and thickness margin are always zero
            width_optimality = t_TE_stator_min_margin = 0.
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the stator
            stator_ratio_residual = stator_chord / turbine_meanline.p_stator - meanline_arguments["chord_over_pitch_stator"]

        # Follow the procedure if multiple da Laval nozzles are used
        elif count > 1:
            # First calculate throat diameter based on total throat area and number of nozzles
            throat_diameter = np.sqrt(4 * A_throat_stator / (count * np.pi))
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**{**stator_arguments, "reference_length": throat_diameter})

            # Get stator blade chord (from shape uncorrected for BL)
            stator_chord = self._calculate_nozzle_chord(stator_shape) * self.stator_nozzle_chord_scale
            # Get nozzle exit width and height from stator_shape. Use stator shape uncorrected for the boundary layer
            uncorrected_stator_shape = stator_shape.dimensional_shapes.uncorrected
            width, height = uncorrected_stator_shape.nozzle_passage_pitch, uncorrected_stator_shape.nozzle_exit_width
            # Calculate height optimality from the outermost ellipse point,.
            R_max = self._calculate_ellipse_maximum_radius(width, height, R_mean)
            height_optimality = R_max - R_tip
            # Project two identical, radially oriented neighbouring elliptical exits onto the line joining their centres.
            # Their centre spacing is 2*R_mean*sin(delta/2); support is half of each ellipse's projected width on this line.
            # Subtract both projected half-widths from the spacing: positive means a gap, zero contact, negative overlap.
            # This straight projected gap is separate from the Euler-circle arc gap calculated below. This is metal
            # thickness, while the arc gap below is TE thickness that should be used for meanline calculations.
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
            exit_arc = self._calculate_ellipse_exit_arc(width, height, R_mean, R_Euler)
            signed_arc_gap = turbine_meanline.p_stator - exit_arc
            t_TE_stator_residual = max(0., signed_arc_gap) - turbine_meanline.t_TE_stator

        # Now follow the procedure if a single da Laval nozzle is used
        else:
            # First calculate throat diameter based on total throat area
            throat_diameter = np.sqrt(4 * A_throat_stator / np.pi)
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**{**stator_arguments, "reference_length": throat_diameter})

            # Get stator blade chord (from shape uncorrected for BL)
            stator_chord = self._calculate_nozzle_chord(stator_shape) * self.stator_nozzle_chord_scale
            # Calculate height optimality factor for nozzle exit
            uncorrected_stator_shape = stator_shape.dimensional_shapes.uncorrected
            width, height = uncorrected_stator_shape.nozzle_passage_pitch, uncorrected_stator_shape.nozzle_exit_width
            # Include the flanks when measuring how far the exit extends beyond the tip circle.
            R_max = self._calculate_ellipse_maximum_radius(width, height, R_mean)
            height_optimality = R_max - R_tip
            # Also calculate admission fraction residual
            exit_arc = self._calculate_ellipse_exit_arc(width, height, R_mean, R_Euler)
            admission_fraction_residual = exit_arc / (2 * np.pi * R_Euler) - turbine_meanline.admission_fraction
            # Width optimality factor and thickness margin are always zero
            width_optimality = t_TE_stator_min_margin = 0.
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the stator
            stator_ratio_residual = stator_chord / turbine_meanline.p_stator - meanline_arguments["chord_over_pitch_stator"]

        # Store the actual exit-centre radius for MOC or de Laval nozzle exits.
        turbine_meanline.R_nozzle = inner_side_radius + height / 2 if contour == "moc" else R_mean

        # Now pack all the results and return them
        # Use the stator TE thickness residual for multiple nozzles and the admission fraction residual for one nozzle.
        fourth_residual = t_TE_stator_residual if count > 1 else admission_fraction_residual
        # Components 1-4 are obtained minus assumed; component 5 corrects LE thickness to enforce the metal-thickness minimum.
        residuals = np.array([stator_ratio_residual, rotor_ratio_residual, t_TE_rotor_residual,
                              fourth_residual, rotor_LE_residual])
        coefficients = {"height_optimality": height_optimality, "width_optimality": width_optimality,
                        "t_TE_stator_min_margin": t_TE_stator_min_margin,
                        "t_rotor_min_margin": t_rotor_min_margin}
        if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(list(coefficients.values()))):
            raise ValueError("Supersonic turbine evaluation produced nonfinite residuals or geometry.")
        return residuals, turbine_meanline, stator_shape, rotor_shape, coefficients

    def _calculate_separation_mach_limits(self, M_in, M_out):
        """Return Goldman and Scullin's pressure/suction separation limits, NASA TN D-4421, equations (36)-(40).

        The pressure-recovery coefficient is 1/2: (p_down-p_up)/(gamma*p_up*M_up**2/2).
        The lower surface compresses inlet flow; the upper surface compresses to outlet flow. The empirical
        criterion comes from incompressible experiments and is an approximate supersonic design limit.

        :param float M_in: Actual relative inlet Mach, greater than one.
        :param float M_out: Isentropic relative outlet Mach, greater than one.
        :return: Minimum lower-surface Mach (floored at sonic) and maximum upper-surface Mach.
        :rtype: tuple[float, float]

        Reference: L. J. Goldman and V. J. Scullin, Analytical Investigation of Supersonic Turbomachinery
        Blading, I - Computer Program for Blading Design, NASA TN D-4421 (1968).
        https://ntrs.nasa.gov/api/citations/19680009151/downloads/19680009151.pdf
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

        # Start the upper-surface Mach search at twice the outlet Mach.
        upper_bound = 2 * M_out
        # Double the upper bound until the residual is nonnegative, bracketing the upper Mach limit.
        while upper_residual(upper_bound) < 0:
            upper_bound *= 2
        # Floor the lower Mach limit at one and solve for the upper limit with Brent's method.
        return np.sqrt(max(1., lower_squared)), cast(float, brentq(upper_residual, M_out, upper_bound,
                                                                  xtol=1e-12, full_output=False))

    @staticmethod
    def _calculate_rotor_throat_width(rotor_geometry):
        """Return the minimum gap between rotor passage walls, m.

        The stored pressure and suction surfaces bound one open passage, so no pitch translation is required.
        Represent each wall by straight line segments joining consecutive coordinate points.
        """
        # Read both passage walls in the axial/circumferential plane.
        pressure_surface, suction_surface = rotor_geometry.pressure_surface, rotor_geometry.suction_surface
        pressure = np.column_stack((pressure_surface.x, pressure_surface.y))
        suction = np.column_stack((suction_surface.x, suction_surface.y))
        # Require at least two points per wall, with x/y coordinates that contain neither NaN nor infinity.
        if any(len(wall) < 2 or not np.all(np.isfinite(wall)) for wall in (pressure, suction)):
            raise ValueError("Rotor throat measurement requires at least two points per wall, with finite x/y coordinates.")
        # Each wall is approximated by straight line segments connecting their nodes.
        # If two segments do not intersect, their closest points include an endpoint of at least one segment.
        # Check the points of each wall against the segments of the other wall, in both directions.
        # Process one segment at a time.
        minimum_squared = np.inf
        for points, wall in ((pressure, suction), (suction, pressure)):
            point_segments = np.diff(points, axis=0)
            for start, vector in zip(wall[:-1], np.diff(wall, axis=0)):
                length_squared = np.dot(vector, vector)
                # Skip repeated consecutive wall points: their identical coordinates give a segment of zero length.
                if length_squared == 0:
                    continue
                # Find the nearest position on this segment to each point on the other wall.
                # Limit the position to the segment's endpoints if its perpendicular projection falls beyond them.
                offsets = points - start
                fractions = np.clip(offsets @ vector / length_squared, 0., 1.)
                distances = offsets - fractions[:, None] * vector
                minimum_squared = min(minimum_squared, np.min(np.sum(distances**2, axis=1)))
                # Determine which side of this segment's infinite straight line each point of the other wall lies on.
                sides = offsets[:, 0] * vector[1] - offsets[:, 1] * vector[0]
                differences = sides[:-1] - sides[1:]
                # Interpolate between consecutive points of the other wall to find crossings of that straight line.
                crossings = np.divide(sides[:-1], differences, out=np.full_like(differences, -1.),
                                      where=differences != 0)
                along_segment = (offsets[:-1] + crossings[:, None] * point_segments) @ vector / length_squared
                # A crossing between the endpoints of both wall segments means the passage gap is zero.
                if np.any((crossings >= 0) & (crossings <= 1) & (along_segment >= 0) & (along_segment <= 1)):
                    return 0.
        # Reject an undefined distance, for example when all consecutive wall points have identical coordinates.
        if not np.isfinite(minimum_squared):
            raise ValueError("Rotor throat measurement requires a wall segment with distinct endpoints and a finite distance.")
        # Convert the smallest squared distance to the throat width in metres.
        return np.sqrt(minimum_squared)

    @staticmethod
    def _calculate_nozzle_chord(nozzle: blading.SupersonicStatorNozzle):
        """Return nozzle chord (throat-centre to exit-centre straight distance without BL correction), m.

        Stored nozzle coordinates follow the nozzle axis. The exit centre averages the two wall endpoints,
        including the suction-side straight extension needed to terminate the exit in the machine axial plane.
        """
        # Average the two endpoints of each opening before measuring the actual chord.
        nozzle_shape = nozzle.dimensional_shapes.uncorrected
        pressure_surface, suction_surface = nozzle_shape.pressure_surface, nozzle_shape.suction_surface
        return np.hypot(
            (pressure_surface.x[-1] + suction_surface.x[-1] - pressure_surface.x[0] - suction_surface.x[0]) / 2,
            (pressure_surface.y[-1] + suction_surface.y[-1] - pressure_surface.y[0] - suction_surface.y[0]) / 2)

    @staticmethod
    def _calculate_rectangle_maximum_height(width, R_hub, R_tip):
        """Return maximum height of a hub-tangent rectangle within the tip circle, m, or zero if its width cannot fit."""
        # Reject invalid dimensions, while retaining finite positive widths that overfill the annulus.
        if not np.isfinite(width) or width <= 0:
            raise ValueError("MOC width must be positive and finite.")
        # The inner side is hub-tangent, while the outer corners touch the tip circle at maximum height.
        # If width of the rectangle does not fit in the annulus, return zero height.
        return max(0., np.sqrt(max(0., R_tip**2 - width**2 / 4)) - R_hub)

    @staticmethod
    def _calculate_single_moc_exit_dimensions(R_tip, R_Euler):
        """Return width and height of the largest Euler-centred rectangle inside the tip circle, m."""
        # For half-height h, tip contact gives w=2*sqrt(R_tip**2-(R_Euler+h)**2).
        # The area derivative gives 2*h**2+3*R_Euler*h+R_Euler**2-R_tip**2=0.
        half_height = 2 * (R_tip**2 - R_Euler**2) / (np.sqrt(R_Euler**2 + 8 * R_tip**2) + 3 * R_Euler)
        return 2 * np.sqrt(R_tip**2 - (R_Euler + half_height)**2), 2 * half_height

    @staticmethod
    def _calculate_rectangle_exit_arc(width, height, R_hub, R_Euler):
        """Return total Euler-circle arc inside a rectangular exit, m, including tangential-side crossings."""
        # The inner-side coordinate R_hub locates the actual bottom of either a single or multiple MOC exit.
        # On the right semicircle, x=R_Euler*sin(theta) and y=R_Euler*cos(theta), with 0 <= theta <= pi.
        # The two tangential sides bound one angular interval; clipping also handles no intersection.
        first_angle = np.arccos(np.clip((R_hub + height) / R_Euler, -1., 1.))
        last_angle = np.arccos(np.clip(R_hub / R_Euler, -1., 1.))
        # The radial sides admit intervals near theta=0 and theta=pi, meeting when width >= 2*R_Euler.
        side_angle = np.arcsin(min(1., width / (2 * R_Euler)))
        upper_angle = max(0., min(last_angle, side_angle) - first_angle)
        lower_angle = max(0., last_angle - max(first_angle, np.pi - side_angle))
        # Sum both intervals and reflect across the rectangle centreline to include the left semicircle.
        return 2 * R_Euler * (upper_angle + lower_angle)

    @staticmethod
    def _calculate_ellipse_maximum_radius(width, height, R_mean):
        """Return the maximum distance from the turbine axis to the outermost edge of a mean-radius-centred ellipse, m."""
        a, b = width / 2, height / 2
        # The radial top is outermost unless the tangential semi-axis of the ellipse is sufficiently large.
        if a**2 - b**2 <= R_mean * b:
            return R_mean + b
        # Maximize r**2=a**2+R_mean**2+2*R_mean*b*c+(b**2-a**2)*c**2 over c=cos(phi).
        # The point farthest from the turbine axis lies on the ellipse's side at
        # c=cos(phi)=R_mean*b/(a**2-b**2), with 0<c<1 in this branch.
        return np.sqrt(a**2 + R_mean**2 + (R_mean * b)**2 / (a**2 - b**2))

    @staticmethod
    def _calculate_ellipse_exit_arc(width, height, R_mean, R_Euler):
        """Return Euler-circle arc inside a mean-radius-centred ellipse, m, for a local nozzle."""
        # Require the Euler circle to pass through the nozzle exit.
        a, b = width / 2, height / 2
        if a >= R_Euler or abs(R_Euler - R_mean) >= b:
            raise ValueError("Euler circle must cross a local de Laval exit centred at mean radius.")
        # Intersect x=R_Euler*sin(theta), y=R_Euler*cos(theta) with x**2/a**2+(y-R_mean)**2/b**2=1.
        def intersection(theta):
            """Return the ellipse equation evaluated on the Euler circle."""
            return (R_Euler * np.sin(theta) / a)**2 + ((R_Euler * np.cos(theta) - R_mean) / b)**2 - 1
        # Solve for the positive intersection angle, then use symmetry to obtain the full Euler-circle arc length.
        return 2 * R_Euler * cast(float, brentq(intersection, 0., np.arcsin(a / R_Euler),
                                               xtol=1e-14, full_output=False))
