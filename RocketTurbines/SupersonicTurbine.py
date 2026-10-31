from IdealGas import IdealGas
from Turbine1D import Turbine1D
from TraupelLossModel import TraupelLossModel
import SupersonicTurbineBlading as blading
from RelaxedAitken import solve_with_relaxed_aitken
import numpy as np
from scipy.optimize import brentq
import warnings
from typing import cast
import plotly.graph_objects as go
from mapbox_earcut import triangulate_float64

class SupersonicTurbine:
    def __init__(self, gas: IdealGas, loss_model: TraupelLossModel, Turbine1D_arguments,
                 SupersonicStatorNozzle_arguments, SupersonicRotorBlades_arguments, separation_margin,
                 t_min_stator, t_min_rotor, stator_nozzle_chord_scale=1.0, print_user_warnings=True,
                 symmetric_blades=False, min_MOC_nozzle_width=5e-3):
        """SupersonicTurbine is the class to represent a supersonic turbine, with consistent meanline design and blading
         shape. It ensures that chord/pitch ratios and other variables used in Turbine1D agrees with the values obtained from
         SupersonicTurbineBlading objects, which in turn also use Turbine1D meanline results. Thus, there is a coupling
          between meanline and blading design, which couple_meanline_and_blading solves numerically when called.
           Construction only prepares inputs and leaves the design results as None. This object also checks if the
           stator nozzles exit can fit optimally in the turbine annulus by producing continuous coefficients than can be
            either checked by the engineer or constrained by outer optimizer.

        To assist convergence, evaluate_design caps each TE thickness passed to Turbine1D.size_turbine at 0.24 of
        its row's pitch. Full assumed thicknesses is still used to decrease residuals to zero, but
        the returned Turbine1D object stores the capped thicknesses used for its calculation. A converged
        design with limited_TE_stator_losses or limited_TE_rotor_losses uses modified losses; its efficiency can be
        optimistic and must be reassessed after improving the blading. Boundary-layer calculations are disabled for
        both SupersonicTurbineBlading objects to improve speed and robustness. With symmetric_blades=False,
        rotor LE/TE circumferential thickness matching is ensured by varying ideal outlet Mach number used for rotor
        blade construction at fixed outlet angle. The ideal outlet Mach number from meanline analysis is an initial
        estimate for this rotor geometry solve. With symmetric_blades=True, rotor blades will be symmetric.

        :param IdealGas gas: Object representing calorically perfect working gas.
        :param TraupelLossModel loss_model: Object representing meanline loss model.
        :param dict Turbine1D_arguments: Arguments to Turbine1D.size_turbine, including explicit chord/pitch ratios,
         admission_fraction and initial estimates for design variables calculated in this class. In case a single
          stator nozzle is used, admission_fraction is its initial estimate; its final value is solved from the exit
           geometry like other design variables.
        :param dict SupersonicStatorNozzle_arguments: Arguments for SupersonicStatorNozzle, including contour_method.
         Exit Mach number and flow angle, reference_length and trailing_edge_thickness are calculated in this class.
        :param dict SupersonicRotorBlades_arguments: Arguments for SupersonicRotorBlades. Inlet & outlet Mach numbers
         and flow angles, surface Mach numbers, blade count, radius, speed and
          leading_edge_thickness_over_total_pitch are calculated in this class.
        :param float separation_margin: Flow separation margin, strictly between zero and one. A value of 0.05 reserves
         5% of the Mach interval bounded by Goldman and Scullin's separation limits (NASA TN D-4421, equations (36)-(40)).
        :param float t_min_stator: Minimum true stator metal thickness, m; nonnegative.
        :param float t_min_rotor: Minimum true rotor metal thickness, m; nonnegative.
        :param float stator_nozzle_chord_scale: Positive multiplier of throat-centre to exit-centre chord,
         to account for the upstream turning section, which SupersonicStatorNozzle does not calculate. Default 1.
        :param bool print_user_warnings: Whether to emit this object's user warnings. Default True.
         Traupel extrapolation warnings are controlled separately by loss_model.warn_on_extrapolation.
        :param bool symmetric_blades: Use the rotor inlet flow angle as the rotor outlet angle.
         Default False uses the blade-row outlet angle and enables LE/TE thickness matching by varying ideal outlet
         Mach number used for blade construction. The inlet and ideal outlet Mach numbers are supplied by the meanline
          design.
        :param float min_MOC_nozzle_width: Minimum reference width for multiple MOC stator nozzles, m; finite and
         positive. Default 0.005 m. Wider nozzles may violate the minimum metal thickness or overlap; their signed
         t_TE_stator_min_margin must be checked by the engineer or outer optimizer. Single nozzles are unaffected.
        :ivar numpy.ndarray x_solution: None until coupling converges; then [stator chord/pitch, rotor chord/pitch,
         rotor TE thickness, stator TE thickness (or single-nozzle admission)]; thicknesses are circumferential projections in m.
        :ivar dict coefficients: None until coupling converges; then optimality coefficients, thickness margins and
         boolean limited_TE_stator_losses / limited_TE_rotor_losses flags for the final evaluated design.
        :raises ValueError: If construction inputs are invalid.
        :raises KeyError: If a required input is missing from the argument dictionaries.
        """

        # Meanline - nozzle/blade shapes coupling logic diagram:
        #  SupersonicTurbine.__init__
        #    |
        #    v
        #  Copy input dictionaries; validate inputs and create the shared Fluid object.
        #  Disable mixed-flow Mach matching and BL calculations; match nonsymmetric rotor LE/TE thicknesses using outlet Mach.
        #  Initialize design results to None; return without solving.
        #  Loss-model assumption: evaluate_design caps each TE thickness/pitch input to Turbine1D.size_turbine at 0.24.
        #  Retain full assumed TE thicknesses for shaping and residual matching; flag any limited losses.
        #    |
        #    v  (when the user calls couple_meanline_and_blading())
        #  SupersonicTurbine.couple_meanline_and_blading
        #    |
        #    v
        #  Assemble four unknowns x and their initial estimates:
        #    [stator chord/pitch, rotor chord/pitch, rotor TE thickness, stator TE thickness or admission].
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
        #    |      v                                                                                         |
        #    |    SupersonicTurbine.evaluate_design                                                           |
        #    |      |                                                                                         |
        #    |      +--> Cap TE thickness inputs to Turbine1D.size_turbine at 0.24 of each row's pitch.       |
        #    |      |      Run nested loops; use uncapped TE thicknesses for shaping and residuals.           |
        #    |      |      Obtain flow states, blade-row angles, annulus radii and Euler-radius pitches.      |
        #    |      |                                                                                         |
        #    |      +--> Rotor shaping                                                                        |
        #    |      |      Apply Goldman--Scullin separation limits and separation_margin.                    |
        #    |      |      Select the blade-row outlet angle, or negative inlet angle for symmetric_blades.   |
        #    |      |      Calculate LE thickness projection = t_min_rotor/cos(inlet angle); construct rotor. |
        #    |      |      For nonsymmetric blades, match LE/TE thickness projections via outlet Mach.        |
        #    |      |      Pass rotor LE thickness/total pitch; pitch includes metal at Euler radius.         |
        #    |      |      Measure TE thickness projection and chord between blade edges.                     |
        #    |      |      Store the obtained LE thickness in the meanline object.                            |
        #    |      |      Report circumferential LE/TE thickness margins                                     |
        #    |      |                                                                                         |
        #    |      +--> Stator shaping and placement                                                         |
        #    |      |      Calculate the total choked throat area required by mass flow.                      |
        #    |      |      |                                                                                  |
        #    |      |      +--> Multiple MOC nozzles                                                          |
        #    |      |      |      Determine nozzle placement.                                                 |
        #    |      |      |      Calculate MOC nozzle exit width based on minimum TE thickness               |
        #    |      |      |      and allowable nozzle exit width.                                            |
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
        #    |             Store the actual exit-centre radius as turbine_meanline.R_nozzle.                  |
        #    |      |                                                                                         |
        #    |      v                                                                                         |
        #    |    Return four residuals r(x), meanline/blading objects and geometric coefficients:            |
        #    |      r1/r2: obtained minus assumed stator/rotor chord-to-pitch ratios.                         |
        #    |      r3:  obtained minus assumed rotor TE thickness.                                           |
        #    |      r4: stator TE thickness mismatch (multiple nozzles) or admission mismatch (one nozzle).   |
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
        #  Store x_solution, optimality coefficients, thickness margins and flags for capped TE thickness Turbine1D inputs.
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
        if not isinstance(symmetric_blades, (bool, np.bool_)):
            raise ValueError("symmetric_blades must be a boolean.")
        self.symmetric_blades = bool(symmetric_blades)

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
        if not np.isfinite(min_MOC_nozzle_width) or min_MOC_nozzle_width <= 0:
            raise ValueError("min_MOC_nozzle_width must be finite and positive.")
        self.t_min_stator, self.t_min_rotor = t_min_stator, t_min_rotor
        self.stator_nozzle_chord_scale = stator_nozzle_chord_scale
        self.min_MOC_nozzle_width = min_MOC_nozzle_width

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

        # The coupling uses uncorrected geometry and meanline losses. BL calculations on either row add cost
        # and failure modes without contributing to the coupled residuals, so always switch them off.
        for name, arguments in (("SupersonicStatorNozzle", self.SupersonicStatorNozzle_arguments),
                                ("SupersonicRotorBlade", self.SupersonicRotorBlades_arguments)):
            if arguments.get("calculate_boundary_layer", True) and self.print_user_warnings:
                warnings.warn(f"Boundary-layer calculations for {name} are disabled to speed up calculations"
                              " and improve robustness.", UserWarning, stacklevel=2)
            arguments["calculate_boundary_layer"] = False

        # Different inlet/outlet angles can produce excessive rotor TE thickness for unsymmetric blades. To ensure LE
        # and TE circumferential thicknesses mach, switch on the option that vary ideal outlet Mach number used for the
        # rotor blade construction, preserving the outlet angle.
        if not self.symmetric_blades:
            thickness_matching = {"match_LE_and_TE_thickness": True, "match_thickness_with_mach": True,
                                  "match_thickness_with_angle": False}
            if (any(self.SupersonicRotorBlades_arguments.get(name) != value
                    for name, value in thickness_matching.items()) and self.print_user_warnings):
                warnings.warn("Rotor LE/TE thickness matching through outlet Mach is enabled for"
                              " symmetric_blades=False to prevent excessive TE thickness when inlet and"
                              " outlet flow angles differ.",
                              UserWarning, stacklevel=2)
            self.SupersonicRotorBlades_arguments.update(thickness_matching)

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
            """Return the four obtained design variables and their associated design objects."""

            # If there are multiple nozzles, one of such variables is t_TE_stator
            if no_nozzles != 1:
                # Unpack input arguments
                chord_over_pitch_stator, chord_over_pitch_rotor, t_TE_rotor, t_TE_stator = x_0
                # Copy Turbine1D arguments dictionary so that only internal copy changes during iterations
                meanline_arguments = self.Turbine1D_arguments.copy()
                meanline_arguments.update({"chord_over_pitch_stator": chord_over_pitch_stator,
                                           "chord_over_pitch_rotor": chord_over_pitch_rotor,
                                           "t_TE_rotor": t_TE_rotor,
                                           "t_TE_stator": t_TE_stator})
                # Get all residuals, Turbine1D object, stator & rotor shapes and optimality coefficients & thickness
                # margins
                residuals, turbine_meanline, stator_shape, rotor_shape, coefficients =\
                    self.evaluate_design(meanline_arguments)
                # Obtain new estimate of design values from residuals
                x_1 = np.add(x_0, residuals)

            # If there is just one nozzle, t_TE_stator is always zero. However, admission fraction must be now
            # variable to adjust
            else:
                # Unpack input arguments
                chord_over_pitch_stator, chord_over_pitch_rotor, t_TE_rotor, admission_fraction = x_0
                # Copy Turbine1D arguments dictionary so that only internal copy changes during iterations
                meanline_arguments = self.Turbine1D_arguments.copy()
                meanline_arguments.update({"chord_over_pitch_stator": chord_over_pitch_stator,
                                           "chord_over_pitch_rotor": chord_over_pitch_rotor,
                                           "t_TE_rotor": t_TE_rotor,
                                           "admission_fraction": admission_fraction})
                # Get all residuals, Turbine1D object, stator & rotor shapes and optimality coefficients & thickness
                # margins
                residuals, turbine_meanline, stator_shape, rotor_shape, coefficients = \
                    self.evaluate_design(meanline_arguments)
                # Obtain new estimate of design values from residuals
                x_1 = np.add(x_0, residuals)

            # Return new solution estimate together with remaining results
            return x_1, turbine_meanline, stator_shape, rotor_shape, coefficients

        # Create initial solution array if there are multiple nozzles
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
        # Bound each row's TE thickness by its full pitch; one stator nozzle instead solves for admission fraction.
        lower = np.array([1e-8, 1e-8, 0., 0. if no_nozzles > 1 else 1e-6])
        upper = np.array([np.inf, np.inf, rotor_pitch, stator_pitch if no_nozzles > 1 else 1.])
        # Create solution scales for normalizing it. Solution is normalized because it mixes nondimensional &
        # dimensional variables. Use 12% of total pitch for each thickness scale.
        # Use finite substitutes when a dimensionless initial estimate is needed to construct solution scale.
        scale_estimates = np.nan_to_num(x_initial, nan=1., posinf=1., neginf=0.)
        # Use NumPy maxima to avoid PyCharm's incorrect array-type warning for the indexed scale values.
        solution_scales = \
            np.array([np.maximum(scale_estimates[0], 1.), np.maximum(scale_estimates[1], 1.), 0.12 * rotor_pitch,
                      0.12 * stator_pitch if no_nozzles > 1 else np.clip(scale_estimates[3], 0.01, 1.)])
        # Replace nonfinite or out-of-bound initial estimates with their corresponding solution scales and notify
        # the user. Use NumPy ufuncs to avoid PyCharm recursively expanding overloaded array comparison and bitwise
        # operators.
        invalid_initial = np.logical_or(np.logical_not(np.isfinite(x_initial)),
                                       np.logical_or(np.less(x_initial, lower), np.greater(x_initial, upper)))
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

    def print_design_coupling_results(self, coefficients=None):
        """Print packing coefficients, circumferential thickness margins and TE-loss limitation flags.

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
                      "t_TE_stator_min_margin": ">= 0 (nonnegative)",
                      "t_LE_rotor_min_margin": ">= 0 (nonnegative)",
                      "t_TE_rotor_min_margin": ">= 0 (nonnegative)"}
        # First print title
        print("\nOptimality coefficients and thickness margins:")
        # Print each supplied coefficient and the sign condition for a feasible design.
        for name, value in coefficients.items():
            if name in ("limited_TE_stator_losses", "limited_TE_rotor_losses"):
                print(f"{name}: {bool(value)}; True means cicrumferential TE thickness to pitch ratio was limited to 0.24 in"
                      f" meanline loss calculations.")
            else:
                print(f"{name}: {value * 1000:.4g} mm; feasible: {conditions[name]}")

    def evaluate_design(self, meanline_arguments):
        """This method allows to calculate the residuals between assumed values for 1D meanline calculations in
        Turbine 1D and obtained values from blade shaping with SupersonicTurbineBlading. These residuals can be driven
         to zero by couple_meanline_and_blading or the engineer to obtain consistent design. This method also returns
          some remaining residuals and optimality coefficients that can be constrained by outer optimizer or which can
           inform the engineer about manufacturability & optimality of the supersonic turbine.

        :param dict meanline_arguments: Dictionary with size_turbine inputs.
        :return: Four residuals, Turbine1D object, SupersonicStatorNozzle object, SupersonicRotorBlade object and
         optimality/margin coefficients.
         Residuals 1/2 are dimensionless chord/pitch differences. Residual 3 is geometric-minus-full-assumed rotor
         TE thickness (m). Residual 4 is stator arc thickness (m), or admission fraction for one nozzle.
         height_optimality and width_optimality are clearance values for nozzle edges (m); positive values indicate
         infeasible packing. t_TE_stator_min_margin is circumferential projection length of nozzle TE thickness minus the
         required value calculated from t_min_stator and stator outlet flow angle alpha_1. Each rotor margin
         (t_LE_rotor_min_margin and t_TE_rotor_min_margin) is obtained length of circumferential projection of edge
         thickness minus the required value, t_min_rotor/cos(rotor inlet flow angle; beta_1) for LE and
         t_min_rotor/cos(rotor outlet flow angle; beta_2) for TE. limited_TE_stator_losses and limited_TE_rotor_losses
         indicate which assumed TE circumferential thickness/pitch ratios exceed 0.24.
        :rtype: tuple[numpy.ndarray, Turbine1D, blading.SupersonicStatorNozzle, blading.SupersonicRotorBlade, dict]
        :raises ValueError: If the flow is not supersonic or the construction assumptions are violated.
        :raises KeyError: If a required input is missing from the argument dictionaries.
        """

        # The automatic coupling between Turbine1D (meanline code) and SupersonicTurbineBlading (blade shaping) is
        # available for matching chord/pitch ratios of the stator and the rotor by driving difference between assumed
        # and obtained values to zero. These are stator_chord_pitch_ratio_residual and rotor_chord_pitch_ratio_residual.
        # However, the code cannot automatically ensure stator nozzles match the annulus of the turbine, since it
        # requires varying nozzle number or flow coefficients, which are preferentially things adjusted by the user or
        # some outer optimizer. Fortunately, the code can determine if stator nozzles are placed optimally by using a
        # few simple & continuous coefficients, which can be constrained by outer optimizer or which can give user
        # indication of the feasibility/optimality of the stator. These calculations are required to calculate
        # stator_chord_pitch_ratio_residual, so they are performed already in the numerical solution loop. They also yield
        # arc length t_TE_stator between edges of adjacent nozzle exits, which must match the assumed value for
        # meanline calculations and blade shaping as well. Consequently, another design consistency
        # condition arise, t_TE_stator_residual, which the numerical solver will attempt to make zero.

        # Furthermore, a word about nomenclature and definition of trailing edge thickness. In Turbine1D, TraupelLossModel,
        # in this class or in SupersonicTurbineBlading library, t_TE or t_LE means the length of the projection of
        # unwrapped trailing/leading edge on the circumferential axis. This allows to easily perform loss or MOC
        # calculations. However, while often abbreviated as thickness/circumferential thickness/projection elsewhere,
        # this is not the same as true metal thickness - the blade/nozzles inlets and outlets are angled, so actual
        # metal thicknesses there are smaller. The minimal allowed values of this true metal thickness are given as
        # t_min_rotor and t_min_stator as inputs to this object. Stator blade outlet angle is then used to change
        # t_min_stator to t_TE_stator_min. Multiple MOC nozzles initially use this minimum to calculate their width,
        # but min_MOC_nozzle_width can widen them and reduce the obtained metal thickness, possibly causing overlap.
        # For da Laval nozzles, exit height and width are coupled and there is one less degree of freedom in shaping.
        # Both nozzle types therefore report the signed margin
        # t_TE_stator_min_margin = t_TE_stator_min_obtained - t_TE_stator_min, which the engineer or outer optimizer
        # must keep nonnegative by changing nozzle count, admission fraction, etc.
        # Rotor LE projection length, t_LE_rotor is calculated in this method as t_min_rotor/cos(beta_1),
        # and passed to SupersonicRotorBlade. This sets the rotor TE projection length, t_TE_rotor.
        # t_LE_rotor_min_margin and t_TE_rotor_min_margin compare the obtained circumferential edge
        # thicknesses with t_min_rotor/cos(flow angle) at the inlet and outlet angles. They are informational;
        # LE margin is zero to roundoff, while the TE margin can be positive or negative depending on the flow
        # outlet angle. An independent residual, t_TE_rotor_residual, matches the assumed rotor TE thickness for
        # meanline analysis to the obtained t_TE_rotor.

        # symmetric_blades uses -beta_1 as the shaping outlet angle; otherwise the blade-row outlet angle is used.
        # BL calculations are disabled for both rows, and only uncorrected geometry is used here.
        # Nonsymmetric rotors match LE/TE circumferential thicknesses through varying ideal outlet Mach number used for
        # the construction of the rotor blade; their minimum metal
        # thicknesses can still differ because the inlet and outlet angles differ. SupersonicRotorBlade takes
        # t_LE_rotor as input; t_TE_rotor is measured independently.

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
        # exit is tangent to the hub. If this width is below min_MOC_nozzle_width, increase it to that minimum and report
        # the resulting nozzle packing violation through t_TE_stator_min_margin instead of rejecting a nonpositive
        # reference width. Actual exit centres start at Euler radius. Then, maximum possible nozzle height,
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
        # Minimum TE spacing is measured at the nozzle corners and can be below t_TE_stator_min when the minimum nozzle width is applied.
        # Negative spacing denotes overlap and is retained in t_TE_stator_min_margin for the engineer/outer optimizer.
        # Finally, the arc length at Euler diameter between edges of adjacent nozzles, t_TE_stator_obtained,
        # can be calculated. Clip it at zero when exits overlap so negative thickness is not passed to Turbine1D.
        # The clipped value must equal the arc length t_TE_stator assumed during meanline calculations.
        # Their difference is t_TE_stator_residual and the numerical solver must hold it at zero - it means that assumed
        # and obtained trailing edge thicknesses are consistent, and that meanline code and nozzle shapes agree with
        # each other. This satisfies point 2. This clipping is not a problem if t_TE_stator_min_margin is nonnegative,
        # as then t_TE_stator_obtained will be always positive too.

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
        # ellipse minor diameter, which preferentially matches annulus height, H_annulus = R_tip - R_hub.
        # This is the first condition to
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

        # During numerical coupling, it may happen that TE thickness for either stator or rotor is above 0.2 of
        # respective pitch value. Above that value, TraupelLossModel can calculate very large loss coefficients and
        # entropy rises, which make Turbine1D fail. Consequently, each TE input to Turbine1D.size_turbine at is capped
        # 0.24 of its row's pitch. However, the full assumed TE thicknesses is kept for comparison with values obtained
        # from nozzle/rotor geometry. Thus, an assumed thickness above 24% of pitch can still match its obtained value.
        # Flags below identify designs whose efficiency uses capped TE thickness and may be optimistic; the engineer
        # must address their blading and reassess uncapped losses after convergence.

        # Work on a copy, so manual evaluations do not change the caller's setup.
        meanline_arguments = meanline_arguments.copy()

        # One nozzle has no adjacent trailing-edge wall, so blade thickness is set to zero
        if meanline_arguments["no_blades_stator"] == 1:
            meanline_arguments["t_TE_stator"] = 0.

        # Compute stator and rotor pitches for this trial.
        diameter = 2 * np.sqrt((meanline_arguments["shaft_power"] / meanline_arguments["mdot"])
                               / meanline_arguments["work_coefficient_tt"]) \
                   / (meanline_arguments["RPM"] * 2 * np.pi / 60)
        rotor_pitch = diameter * np.pi / meanline_arguments["no_blades_rotor"]
        stator_pitch = meanline_arguments["admission_fraction"] * diameter * np.pi \
                       / meanline_arguments["no_blades_stator"]

        # Create dictionary which will store arguments and capped TE thickness. Also create dictionary to store flags
        # whether TE thickness passed to Turbine1D was capped.
        capped_meanline_arguments = meanline_arguments.copy()
        limited_TE_losses = {}
        for row, pitch in (("stator", stator_pitch), ("rotor", rotor_pitch)):
            thickness = meanline_arguments[f"t_TE_{row}"]
            capped_meanline_arguments[f"t_TE_{row}"] = min(thickness, 0.24 * pitch)
            limited_TE_losses[row] = bool(thickness > 0.24 * pitch)

        # Obtain meanline analysis results from Turbine1D
        turbine_meanline = Turbine1D()
        turbine_meanline.size_turbine(**capped_meanline_arguments)
        limited_TE_stator_losses = limited_TE_losses["stator"]
        limited_TE_rotor_losses = limited_TE_losses["rotor"]

        # Read the actual inlet and isentropic rotor outlet reference states.
        results = turbine_meanline.analysis_results_at_design_point
        M_in, M_out = results["M_r1"], results["M_r2_ideal_r_real_s"]
        if min(M_in, M_out, results["M_s1_ideal"]) <= 1:
            raise ValueError("Supersonic blading requires M_r1, M_r2_ideal_r_real_s and M_s1_ideal > 1.")

        # Select the outlet angle for shaping; symmetric blade mirrors the inlet angle about the axial direction.
        rotor_inlet_angle = results["beta_1"]
        rotor_outlet_angle = (-rotor_inlet_angle if self.symmetric_blades else
                              turbine_meanline.blade_row_results_at_design_point["beta_2_blade"])
        rotor_cosines = np.cos([rotor_inlet_angle, rotor_outlet_angle])

        # Convert the normal minimum to circumferential projections at both edges, using their flow angles.
        # Prescribe the LE minimum for blade shaping; the contour determines the obtained TE thickness.
        t_LE_rotor_min, t_TE_rotor_min = np.divide(self.t_min_rotor, rotor_cosines)
        t_LE_rotor = t_LE_rotor_min

        # Now calculate rotor surface Mach numbers based on separation margins
        # First lower Mach number
        M_lower_min, M_upper_max = self._calculate_separation_mach_limits(M_in, M_out)
        M_lower = M_in - (1 - self.separation_margin) * (M_in - M_lower_min)
        # Now upper Mach number
        M_upper = M_out + (1 - self.separation_margin) * (M_upper_max - M_out)
        # An accelerating/decelerating rotor must also bracket both endpoint Mach numbers.
        if not 1 <= M_lower < min(M_in, M_out) or M_upper <= max(M_in, M_out):
            raise ValueError("Separation-margin surface Mach numbers do not bracket both rotor endpoint Mach numbers.")

        # Now calculate rotor shape and dimensions. Use the actual meanline inlet Mach number and isentropic outlet
        # Mach number. For nonsymmetric rotors, the ideal outlet Mach initializes the blading library's LE/TE thickness
        # matching scheme. Outlet Mach number used for the rotor blade construction can thus change, while keeping
        # fixed outlet angle fixed. Surface Mach numbers are obtained from the separation-based values above.
        # Create dictionary with arguments:
        rotor_arguments = {**self.SupersonicRotorBlades_arguments,
                           "ideal_inlet_relative_flow_mach": M_in,
                           "ideal_inlet_relative_flow_angle": np.degrees(rotor_inlet_angle),
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
        # Record the outlet Mach actually used for rotor blade construction. In case of nonsymmetric blades, it can be
        # different than the one obtained from Turbine1D
        results["M_r2_ideal_r_real_s_construction"] = rotor_shape.ideal_outlet_relative_flow_mach

        # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the rotor
        # The two stored surfaces bound a passage. Their endpoint averages give the vector from LE to TE
        # after translating the suction side by one total pitch; the constant translation cancels in the difference.
        # Length of that vactor is blade chord.
        rotor_ideal = rotor_shape.dimensionalize().uncorrected
        pressure_surface, suction_surface = rotor_ideal.pressure_surface, rotor_ideal.suction_surface
        rotor_chord = np.hypot(
            (pressure_surface.x[-1] + suction_surface.x[-1] - pressure_surface.x[0] - suction_surface.x[0]) / 2,
            (pressure_surface.y[-1] + suction_surface.y[-1] - pressure_surface.y[0] - suction_surface.y[0]) / 2)
        rotor_chord_pitch_ratio_residual = \
            rotor_chord / turbine_meanline.p_rotor - meanline_arguments["chord_over_pitch_rotor"]

        # Perform calculations related to rotor LE & TE thicknesses
        # First subtract open passages from total Euler-radius pitch to obtain blade TE projection on
        # circumferential direction.
        t_TE_rotor = turbine_meanline.p_rotor - rotor_ideal.outlet_passage_pitch
        # Store the calculated projected LE thickness directly in the returned meanline object.
        turbine_meanline.t_LE_rotor = t_LE_rotor
        # Match the assumed TE thickness with the obtained one, independently of the capped value used by Turbine1D.
        t_TE_rotor_residual = t_TE_rotor - meanline_arguments["t_TE_rotor"]
        # Report obtained-minus-required circumferential thickness projections at each rotor edge.
        # Use NumPy subtraction to avoid expensive overloaded-operator inference in PyCharm.
        t_LE_rotor_min_margin, t_TE_rotor_min_margin = np.subtract(
            [t_LE_rotor, t_TE_rotor], [t_LE_rotor_min, t_TE_rotor_min])

        # Calculate total throat area needed to satisfy massflow requirements
        gamma = self.gas.gamma
        stator_throat_mass_flux = results["p_0"] / np.sqrt(results["T_0"]) * np.sqrt(gamma / self.gas.R) \
                           * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        A_throat_stator = results["mdot_total"] / stator_throat_mass_flux
        # Store the total stator throat area required to pass the admitted mass flow.
        turbine_meanline.A_throat_stator = A_throat_stator

        # Measure the narrowest gap between the uncorrected rotor passage walls at Euler radius.
        w_Euler = self._calculate_rotor_throat_width(rotor_ideal)
        # Assume geometrically similar sections, with throat width proportional to radius: w(r)=w_Euler*r/R_Euler.
        # Integrate this width from hub to tip, then include one passage per blade and the admitted fraction.
        turbine_meanline.A_throat_rotor = (w_Euler * turbine_meanline.no_blades_rotor
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
        t_TE_stator_min = self.t_min_stator / np.cos(angle)
        delta = turbine_meanline.p_stator / R_Euler
        # Assemble dictionary with stator shape input arguments. Calculated quantities override any duplicate inputs
        stator_arguments = {**self.SupersonicStatorNozzle_arguments,
                            "requested_outlet_absolute_flow_mach": results["M_s1_ideal"],
                            "requested_outlet_absolute_flow_angle": results["alpha_1_deg"],
                            "fluid": self.SupersonicTurbineBlading_Fluid,
                            "upstream_total_temperature": results["T_0"],
                            "upstream_total_pressure": results["p_0"],
                            "trailing_edge_thickness": meanline_arguments["t_TE_stator"] if count > 1 else 0.}
        contour = stator_arguments["contour_method"]
        if contour not in ("moc", "conical"):
            raise ValueError("contour_method must be 'moc' or 'conical' (de Laval).")

        # Now separate procedures outlined at the top of this method can be followed to shape nozzles in the stator and
        # determine all remaining margins, residuals and optimality coefficients
        # If MOC stator nozzles are used
        if contour == "moc" and count > 1:
            # First calculate width for exactly the required minimum thickness at the hub tangent line.
            # Adjacent reference inner corners are separated by 2*R_hub*sin(delta/2) - w_nozzle*cos(delta/2).
            w_nozzle_reference = (2 * R_hub * np.sin(delta / 2) - t_TE_stator_min) / np.cos(delta / 2)
            # Keep a positive construction width even when this violates the minimum metal thickness.
            w_nozzle = max(self.min_MOC_nozzle_width, w_nozzle_reference)
            H_max = self._calculate_rectangle_maximum_height(w_nozzle, R_hub, R_tip)
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**{**stator_arguments, "reference_length": w_nozzle})

            # Get stator blade chord (from shape uncorrected for BL)
            stator_chord = self._calculate_nozzle_chord(stator_shape) * self.stator_nozzle_chord_scale
            # Calculate throat height from total throat area, reference width and the number of nozzles
            H_nozzle = A_throat_stator / (count * stator_shape.dimensional_shapes.uncorrected.throat_width)
            # Calculate height optimality factor for MOC nozzle exit
            height_optimality = H_nozzle - H_max
            # Place the nozzle at Euler radius first. If for calculated height it crosses the tip radius,
            # shift nozzles inward instead. When H_nozzle exceeds H_max, retain hub tangency and allow the
            # nozzle exit to cross the tip radius.
            inner_side_radius = max(R_hub, min(R_Euler - H_nozzle / 2, R_hub + H_max - H_nozzle))
            # Moving the inner side outward from the reference adds 2*(inner_side_radius-R_hub)*sin(delta/2) to its gap.
            # Widening the reference nozzle to minimum value reduces the gap by the extra width projected onto the
            # spacing line. Preserve a negative gap/margin for overlapping exits.
            t_TE_stator_min_obtained = (t_TE_stator_min
                                        - (w_nozzle - w_nozzle_reference) * np.cos(delta / 2)
                                        + 2 * (inner_side_radius - R_hub) * np.sin(delta / 2))
            t_TE_stator_min_margin = t_TE_stator_min_obtained - t_TE_stator_min
            # The thickness margin above reports any shortfall caused by widening the hub-tangent reference.
            width_optimality = 0.
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the stator
            stator_chord_pitch_ratio_residual = \
                stator_chord / turbine_meanline.p_stator - meanline_arguments["chord_over_pitch_stator"]
            # Calculate trailing edge thickness residuals between meanline design and blade shapes for the stator
            exit_arc = self._calculate_rectangle_exit_arc(w_nozzle, H_nozzle, inner_side_radius, R_Euler)
            # Clip the meanline TE thickness, so that Turbine1D does not fail. This clip will not occur if
            # t_TE_stator_min_margin is made to be nonnegative by the engineer or the outer optimizer.
            t_TE_stator_obtained = max(0., turbine_meanline.p_stator - exit_arc)
            t_TE_stator_residual = t_TE_stator_obtained - meanline_arguments["t_TE_stator"]

        # Follow the procedure if a single MOC stator nozzle is used
        elif contour == "moc":
            # Calculate width and height of the optimal nozzle exit shape to maximize area taken up by it
            # Choose width by maximizing rectangle area inside the tip circle with its centre fixed at Euler radius.
            w_nozzle, _ = self._calculate_single_moc_exit_dimensions(R_tip, R_Euler)
            # At this width, the largest fitting height places the inner side at the hub and outer corners at the tip.
            H_max = self._calculate_rectangle_maximum_height(w_nozzle, R_hub, R_tip)
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**{**stator_arguments, "reference_length": w_nozzle})

            # Get stator blade chord (from shape uncorrected for BL)
            stator_chord = self._calculate_nozzle_chord(stator_shape) * self.stator_nozzle_chord_scale
            # From total throat area, calculate throat height
            H_nozzle = A_throat_stator / stator_shape.dimensional_shapes.uncorrected.throat_width
            # Calculate height optimality factor for MOC nozzle exit
            height_optimality = H_nozzle - H_max
            # Place the nozzle at Euler radius first. If for calculated height it crosses the tip radius,
            # shift nozzles inward instead. When H_nozzle exceeds H_max, retain hub tangency and allow the
            # nozzle exit to cross the tip radius.
            inner_side_radius = max(R_hub, min(R_Euler - H_nozzle / 2, R_hub + H_max - H_nozzle))
            # Calculate the obtained admission fraction from the nozzle exit arc at Euler radius.
            exit_arc = self._calculate_rectangle_exit_arc(w_nozzle, H_nozzle, inner_side_radius, R_Euler)
            admission_fraction_obtained = exit_arc / (2 * np.pi * R_Euler)
            # Also calculate admission fraction residual
            admission_fraction_residual = admission_fraction_obtained - turbine_meanline.admission_fraction
            # Width optimality factor and thickness margin are always zero
            width_optimality = t_TE_stator_min_margin = 0.
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the stator
            stator_chord_pitch_ratio_residual = \
                stator_chord / turbine_meanline.p_stator - meanline_arguments["chord_over_pitch_stator"]

        # Follow the procedure if multiple da Laval nozzles are used
        elif count > 1:
            # First calculate throat diameter based on total throat area and number of nozzles
            D_throat = np.sqrt(4 * A_throat_stator / (count * np.pi))
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**{**stator_arguments, "reference_length": D_throat})

            # Get stator blade chord (from shape uncorrected for BL)
            stator_chord = self._calculate_nozzle_chord(stator_shape) * self.stator_nozzle_chord_scale
            # Get nozzle exit width and height from stator_shape. Use stator shape uncorrected for the boundary layer
            uncorrected_stator_shape = stator_shape.dimensional_shapes.uncorrected
            D_ellipse, D_nozzle = (uncorrected_stator_shape.nozzle_passage_pitch,
                                    uncorrected_stator_shape.nozzle_exit_width)
            # Calculate height optimality from the outermost ellipse point,.
            R_max = self._calculate_ellipse_maximum_radius(D_ellipse, D_nozzle, R_mean)
            height_optimality = R_max - R_tip
            # Project two identical, radially oriented neighbouring elliptical exits onto the line joining their centres.
            # Their centre spacing is 2*R_mean*sin(delta/2); support is half of each ellipse's projected width on this line.
            # Subtract both projected half-widths from the spacing: positive means a gap, zero contact, negative overlap.
            # This straight projected gap is separate from the Euler-circle arc gap calculated below. This is metal
            # thickness, while the arc gap below is TE thickness that should be used for meanline calculations.
            half_pitch = delta / 2
            support = np.hypot(D_ellipse / 2 * np.cos(half_pitch), D_nozzle / 2 * np.sin(half_pitch))
            t_TE_stator_min_obtained = 2 * (R_mean * np.sin(half_pitch) - support)
            available_half_width_squared = ((R_mean * np.sin(half_pitch) - t_TE_stator_min / 2)**2
                                            - (D_nozzle / 2 * np.sin(half_pitch))**2)
            available_width = (2 * np.sqrt(max(0., available_half_width_squared)) / np.cos(half_pitch)
                               if 2 * R_mean * np.sin(half_pitch) >= t_TE_stator_min else 0.)
            width_optimality = D_ellipse - available_width
            # Calculate trailing edge thickness margin
            # Do not clip negative clearance: it measures overlap and remains useful to an outer optimizer.
            t_TE_stator_min_margin = t_TE_stator_min_obtained - t_TE_stator_min
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the stator
            stator_chord_pitch_ratio_residual = \
                stator_chord / turbine_meanline.p_stator - meanline_arguments["chord_over_pitch_stator"]
            # Calculate trailing edge thickness residuals between meanline design and blade shapes for the stator
            exit_arc = self._calculate_ellipse_exit_arc(D_ellipse, D_nozzle, R_mean, R_Euler)
            t_TE_stator_obtained = turbine_meanline.p_stator - exit_arc
            t_TE_stator_residual = max(0., t_TE_stator_obtained) - meanline_arguments["t_TE_stator"]

        # Now follow the procedure if a single da Laval nozzle is used
        else:
            # First calculate throat diameter based on total throat area
            D_throat = np.sqrt(4 * A_throat_stator / np.pi)
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**{**stator_arguments, "reference_length": D_throat})

            # Get stator blade chord (from shape uncorrected for BL)
            stator_chord = self._calculate_nozzle_chord(stator_shape) * self.stator_nozzle_chord_scale
            # Calculate height optimality factor for nozzle exit
            uncorrected_stator_shape = stator_shape.dimensional_shapes.uncorrected
            D_ellipse, D_nozzle = (uncorrected_stator_shape.nozzle_passage_pitch,
                                    uncorrected_stator_shape.nozzle_exit_width)
            # Include the flanks when measuring how far the exit extends beyond the tip circle.
            R_max = self._calculate_ellipse_maximum_radius(D_ellipse, D_nozzle, R_mean)
            height_optimality = R_max - R_tip
            # Calculate the obtained admission fraction from the nozzle exit arc at Euler radius.
            exit_arc = self._calculate_ellipse_exit_arc(D_ellipse, D_nozzle, R_mean, R_Euler)
            admission_fraction_obtained = exit_arc / (2 * np.pi * R_Euler)
            # Also calculate admission fraction residual
            admission_fraction_residual = admission_fraction_obtained - turbine_meanline.admission_fraction
            # Width optimality factor and thickness margin are always zero
            width_optimality = t_TE_stator_min_margin = 0.
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes for the stator
            stator_chord_pitch_ratio_residual = \
                stator_chord / turbine_meanline.p_stator - meanline_arguments["chord_over_pitch_stator"]

        # Store the actual exit-centre radius for MOC or de Laval nozzle exits.
        turbine_meanline.R_nozzle = inner_side_radius + H_nozzle / 2 if contour == "moc" else R_mean

        # Now pack all the results and return them
        # Use the stator TE thickness residual for multiple nozzles and the admission fraction residual for one nozzle.
        fourth_residual = t_TE_stator_residual if count > 1 else admission_fraction_residual
        # All four components are obtained minus assumed; rotor manufacturing margins are informational only.
        residuals = np.array([stator_chord_pitch_ratio_residual, rotor_chord_pitch_ratio_residual, t_TE_rotor_residual,
                              fourth_residual])
        coefficients = {"height_optimality": height_optimality, "width_optimality": width_optimality,
                        "t_TE_stator_min_margin": t_TE_stator_min_margin,
                        "t_LE_rotor_min_margin": t_LE_rotor_min_margin,
                        "t_TE_rotor_min_margin": t_TE_rotor_min_margin,
                        "limited_TE_stator_losses": limited_TE_stator_losses,
                        "limited_TE_rotor_losses": limited_TE_rotor_losses}
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
                # Use NumPy functions for array arithmetic below to prevent PyCharm type-inference freezes.
                offsets = np.subtract(points, start)
                fractions = np.clip(np.divide(np.matmul(offsets, vector), length_squared), 0., 1.)
                distances = np.subtract(offsets, np.multiply(fractions[:, None], vector))
                minimum_squared = min(minimum_squared, np.min(np.sum(np.square(distances), axis=1)))
                # Determine which side of this segment's infinite straight line each point of the other wall lies on.
                sides = np.subtract(np.multiply(offsets[:, 0], vector[1]), np.multiply(offsets[:, 1], vector[0]))
                differences = np.subtract(sides[:-1], sides[1:])
                # Interpolate between consecutive points of the other wall to find crossings of that straight line.
                crossings = np.divide(sides[:-1], differences, out=np.full_like(differences, -1.),
                                      where=np.not_equal(differences, 0))
                along_segment = np.divide(
                    np.matmul(np.add(offsets[:-1], np.multiply(crossings[:, None], point_segments)), vector),
                    length_squared)
                # A crossing between the endpoints of both wall segments means the passage gap is zero.
                if np.any(np.all((np.greater_equal(crossings, 0), np.less_equal(crossings, 1),
                                  np.greater_equal(along_segment, 0), np.less_equal(along_segment, 1)), axis=0)):
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

    def plot_3D_geometry(self):
        """Display the assembled stator and rotor in an interactive Plotly browser view.

        :raises ValueError: If coupled geometry is unavailable or a contour cannot be triangulated.
        """

        # Require the stored coupled design before constructing any plotting geometry.
        if any(value is None for value in (self.turbine_meanline, self.stator_shape, self.rotor_shape)):
            raise ValueError("No coupled geometry available. Call couple_meanline_and_blading() first.")
        meanline = self.turbine_meanline
        required_geometry = ("R_hub", "R_tip", "R_Euler", "R_nozzle", "s_ax", "p_stator",
                             "A_throat_stator", "no_blades_stator", "no_blades_rotor")
        if any(getattr(meanline, geometry_field) is None for geometry_field in required_geometry):
            raise ValueError("Coupled turbine geometry is incomplete. Call couple_meanline_and_blading() first.")

        # Use NumPy functions for array arithmetic and masks to avoid PyCharm's expensive overloaded-operator inference.
        # Clean closed profiles before triangulation so that their end caps share the wall vertices.
        def triangulate_profile(profile_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            profile_points = np.asarray(profile_points, dtype=float)
            if profile_points.ndim != 2 or profile_points.shape[1] != 2 or not np.all(np.isfinite(profile_points)):
                raise ValueError("A plotted contour must contain finite axial and tangential coordinates.")
            tolerance = max(float(np.max(np.ptp(profile_points, axis=0))), float(np.finfo(float).tiny)) * 1e-10
            segment_lengths = np.linalg.norm(np.diff(profile_points, axis=0), axis=1)
            distinct_point_mask = np.concatenate(([True], np.greater(segment_lengths, tolerance)))
            profile_points = profile_points[distinct_point_mask]
            if (len(profile_points) > 1
                    and float(np.linalg.norm(np.subtract(profile_points[-1], profile_points[0]))) <= tolerance):
                profile_points = profile_points[:-1]

            # Remove redundant points on straight segments while retaining the actual curved contour.
            incoming_edges = np.subtract(profile_points, np.roll(profile_points, 1, axis=0))
            outgoing_edges = np.subtract(np.roll(profile_points, -1, axis=0), profile_points)
            edge_cross_products = np.subtract(np.multiply(incoming_edges[:, 0], outgoing_edges[:, 1]),
                                              np.multiply(incoming_edges[:, 1], outgoing_edges[:, 0]))
            adjacent_edge_lengths = np.add(np.linalg.norm(incoming_edges, axis=1),
                                           np.linalg.norm(outgoing_edges, axis=1))
            collinear_point_mask = np.logical_and(
                np.less_equal(np.abs(edge_cross_products), np.multiply(tolerance, adjacent_edge_lengths)),
                np.greater_equal(np.sum(np.multiply(incoming_edges, outgoing_edges), axis=1), 0))
            profile_points = profile_points[np.logical_not(collinear_point_mask)]
            if len(profile_points) < 3:
                raise ValueError("A plotted contour must enclose a nonzero area.")

            # Use consistent winding and constrained polygon triangulation for concave blade and nozzle sections.
            following_points = np.roll(profile_points, -1, axis=0)
            signed_area = float(np.sum(np.subtract(np.multiply(profile_points[:, 0], following_points[:, 1]),
                                                  np.multiply(profile_points[:, 1], following_points[:, 0])))) / 2
            if signed_area < 0:
                profile_points = profile_points[::-1]
            profile_points = np.ascontiguousarray(profile_points)
            end_cap_faces = triangulate_float64(profile_points, np.array([len(profile_points)], dtype=np.uint32)).reshape(-1, 3)
            triangle_vertices = profile_points[end_cap_faces]
            first_triangle_edges = np.subtract(triangle_vertices[:, 1], triangle_vertices[:, 0])
            second_triangle_edges = np.subtract(triangle_vertices[:, 2], triangle_vertices[:, 0])
            triangle_cross_products = np.subtract(np.multiply(first_triangle_edges[:, 0], second_triangle_edges[:, 1]),
                                                  np.multiply(first_triangle_edges[:, 1], second_triangle_edges[:, 0]))
            triangulated_area = float(np.sum(np.abs(triangle_cross_products))) / 2
            if (abs(signed_area) <= tolerance**2
                    or not np.isclose(triangulated_area, abs(signed_area), rtol=1e-7, atol=tolerance**2)):
                raise ValueError("A plotted contour cannot be triangulated into a valid closed section.")
            return profile_points, end_cap_faces

        # Sweep a profile between two radial levels; leave specified inlet and outlet edges open for nozzles.
        def extrude_profile(profile_points: np.ndarray, end_cap_faces: np.ndarray, lower_radius: float,
                            upper_radius: float, opening_edges: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
            point_count = len(profile_points)
            mesh_vertices = np.vstack((np.column_stack((profile_points, np.full(point_count, lower_radius))),
                                       np.column_stack((profile_points, np.full(point_count, upper_radius)))))
            mesh_faces = [end_cap_faces, np.add(end_cap_faces[:, ::-1], point_count)]
            tolerance = float(np.max(np.ptp(profile_points, axis=0))) * 1e-9
            for edge_start_index in range(point_count):
                edge_end_index = (edge_start_index + 1) % point_count
                profile_edge = profile_points[[edge_start_index, edge_end_index]]
                if any(np.allclose(profile_edge, opening_edge, rtol=0, atol=tolerance)
                       or np.allclose(profile_edge[::-1], opening_edge, rtol=0, atol=tolerance)
                       for opening_edge in opening_edges):
                    continue
                mesh_faces.append(np.array([[edge_start_index, edge_end_index + point_count, edge_end_index],
                                            [edge_start_index, edge_start_index + point_count, edge_end_index + point_count]]))
            return mesh_vertices, np.vstack(mesh_faces)

        # Triangulate a regular surface grid explicitly instead of filling its convex hull.
        def grid_mesh(surface_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            row_count, column_count = surface_grid.shape[:2]
            row_offsets = np.multiply(np.arange(row_count - 1)[:, None], column_count)
            cell_start_indices = np.add(row_offsets, np.arange(column_count - 1)).ravel()
            next_row_indices = np.add(cell_start_indices, column_count)
            next_column_indices = np.add(cell_start_indices, 1)
            mesh_faces = np.vstack((np.column_stack((cell_start_indices, next_row_indices, next_column_indices)),
                                    np.column_stack((next_column_indices, next_row_indices, np.add(next_row_indices, 1)))))
            return surface_grid.reshape(-1, 3), mesh_faces

        # Rotate local tangential/radial coordinates rigidly around the turbine axis for each stator nozzle.
        def place_nozzle(nozzle_vertices: np.ndarray, position_angle: float) -> np.ndarray:
            axial_coordinates, tangential_coordinates, radial_coordinates = nozzle_vertices.T
            cosine, sine = float(np.cos(position_angle)), float(np.sin(position_angle))
            positioned_y_coordinates = np.subtract(np.multiply(radial_coordinates, cosine),
                                                   np.multiply(tangential_coordinates, sine))
            positioned_z_coordinates = np.add(np.multiply(radial_coordinates, sine),
                                              np.multiply(tangential_coordinates, cosine))
            return np.column_stack((axial_coordinates, positioned_y_coordinates, positioned_z_coordinates))

        # Add mesh surfaces in millimetres, keeping each blade or nozzle separately identifiable on hover.
        figure = go.Figure()
        def add_mesh(mesh_vertices: np.ndarray, mesh_faces: np.ndarray, mesh_name: str, mesh_color: str, legend_group: str,
                     show_legend: bool, opacity: float = 1.0) -> None:
            triangle_vertices = mesh_vertices[mesh_faces]
            first_triangle_edges = np.subtract(triangle_vertices[:, 1], triangle_vertices[:, 0])
            second_triangle_edges = np.subtract(triangle_vertices[:, 2], triangle_vertices[:, 0])
            doubled_triangle_areas = np.linalg.norm(np.cross(first_triangle_edges, second_triangle_edges), axis=1)
            area_tolerance = float(np.finfo(float).eps) * float(np.max(np.ptp(mesh_vertices, axis=0)))**2
            mesh_faces = mesh_faces[np.greater(doubled_triangle_areas, area_tolerance)]
            coordinates_mm = np.multiply(mesh_vertices, 1e3)
            legend_name = {"stator": "Stator nozzles", "rotor": "Rotor blades"}.get(legend_group, mesh_name)
            figure.add_trace(go.Mesh3d(x=coordinates_mm[:, 0], y=coordinates_mm[:, 1], z=coordinates_mm[:, 2],
                                      i=mesh_faces[:, 0], j=mesh_faces[:, 1], k=mesh_faces[:, 2], color=mesh_color, opacity=opacity,
                                      name=legend_name if show_legend else mesh_name,
                                      legendgroup=legend_group, showlegend=show_legend,
                                      lighting={"ambient": 0.65, "diffuse": 0.75, "specular": 0.2},
                                      hovertemplate=(mesh_name + "<br>Axial: %{x:.2f} mm<br>Y: %{y:.2f} mm"
                                                     "<br>Z: %{z:.2f} mm<extra></extra>")))

        # Rotate the stored nozzle-axis coordinates into the turbine's axial/tangential plane.
        nozzle_geometry = self.stator_shape.dimensional_shapes.uncorrected
        pressure_surface = np.column_stack((nozzle_geometry.pressure_surface.x, nozzle_geometry.pressure_surface.y))
        suction_surface = np.column_stack((nozzle_geometry.suction_surface.x, nozzle_geometry.suction_surface.y))
        nozzle_outlet_angle = float(np.radians(self.stator_shape.outlet_metal_angle))
        outlet_cosine, outlet_sine = float(np.cos(nozzle_outlet_angle)), float(np.sin(nozzle_outlet_angle))
        nozzle_rotation = np.array([[outlet_cosine, -outlet_sine], [outlet_sine, outlet_cosine]])
        nozzle_exit_centre = np.matmul(nozzle_rotation, np.divide(np.add(pressure_surface[-1], suction_surface[-1]), 2))
        stator_nozzle_count = int(meanline.no_blades_stator)
        rotor_blade_count = int(meanline.no_blades_rotor)

        # Build both contoured side walls and the two radial walls of a rectangular MOC nozzle passage.
        if self.stator_shape.contour_method == "moc":
            pressure_surface = np.subtract(np.matmul(pressure_surface, nozzle_rotation.T), nozzle_exit_centre)
            suction_surface = np.subtract(np.matmul(suction_surface, nozzle_rotation.T), nozzle_exit_centre)
            nozzle_profile, nozzle_end_cap_faces = triangulate_profile(np.vstack((pressure_surface, suction_surface[::-1])))
            nozzle_height = meanline.A_throat_stator / (stator_nozzle_count * nozzle_geometry.throat_width)
            nozzle_opening_edges = (np.array([pressure_surface[-1], suction_surface[-1]]),
                                    np.array([suction_surface[0], pressure_surface[0]]))
            nozzle_vertices, nozzle_faces = extrude_profile(nozzle_profile, nozzle_end_cap_faces,
                                                            meanline.R_nozzle - nozzle_height / 2,
                                                            meanline.R_nozzle + nozzle_height / 2, nozzle_opening_edges)

        # Revolve the de Laval meridional contour and vary the straight extension around its oblique exit.
        elif self.stator_shape.contour_method == "conical":
            nozzle_azimuths = np.linspace(0, 2 * np.pi, 65)
            nozzle_exit_radius = float(np.abs(pressure_surface[-1, 1]))
            exit_axial_positions = np.divide(
                np.add(nozzle_exit_centre[0], np.multiply(np.multiply(nozzle_exit_radius, np.cos(nozzle_azimuths)), outlet_sine)),
                outlet_cosine)
            extension_fractions = np.linspace(0, 1, max(2, len(suction_surface) - len(pressure_surface) + 1))[1:]
            straight_extension = np.add(pressure_surface[-1, 0], np.multiply(
                extension_fractions[:, None], np.subtract(exit_axial_positions, pressure_surface[-1, 0])))
            meridian_grid_shape = (len(pressure_surface), len(nozzle_azimuths))
            nozzle_axial_coordinates = np.vstack((np.broadcast_to(pressure_surface[:, 0, None], meridian_grid_shape),
                                                  straight_extension))
            nozzle_radii = np.vstack((np.broadcast_to(np.abs(pressure_surface[:, 1, None]), meridian_grid_shape),
                                      np.full_like(straight_extension, nozzle_exit_radius)))
            nozzle_transverse_coordinates = np.multiply(nozzle_radii, np.cos(nozzle_azimuths))
            nozzle_radial_coordinates = np.add(meanline.R_nozzle, np.multiply(nozzle_radii, np.sin(nozzle_azimuths)))
            turbine_axial_coordinates = np.subtract(
                np.subtract(np.multiply(nozzle_axial_coordinates, outlet_cosine),
                            np.multiply(nozzle_transverse_coordinates, outlet_sine)), nozzle_exit_centre[0])
            turbine_tangential_coordinates = np.subtract(
                np.add(np.multiply(nozzle_axial_coordinates, outlet_sine),
                       np.multiply(nozzle_transverse_coordinates, outlet_cosine)), nozzle_exit_centre[1])
            nozzle_surface_grid = np.stack((turbine_axial_coordinates, turbine_tangential_coordinates,
                                            nozzle_radial_coordinates), axis=-1)
            nozzle_vertices, nozzle_faces = grid_mesh(nozzle_surface_grid)
        else:
            raise ValueError("The plotted stator contour must be 'moc' or 'conical'.")

        # Place the exact nozzle count at its Euler-radius angular pitch in one centred admission sector.
        colors = ("blue", "red")
        nozzle_position_angles = np.divide(np.multiply(np.subtract(np.arange(stator_nozzle_count),
                                                                  (stator_nozzle_count - 1) / 2), meanline.p_stator),
                                          meanline.R_Euler)
        for nozzle_index, nozzle_position_angle in enumerate(nozzle_position_angles):
            add_mesh(place_nozzle(nozzle_vertices, nozzle_position_angle), nozzle_faces, f"Stator nozzle {nozzle_index + 1}",
                     colors[nozzle_index % 2], "stator", nozzle_index == 0)

        # The CAD profile joins the pressure wall to the periodically translated suction wall of one blade.
        # Convert its millimetres to metres, centre the leading edge, and put its upstream edge after the axial gap.
        blade_profile = np.multiply(np.column_stack((self.rotor_shape.uncorrected_blade_profile_x_CAD,
                                                    self.rotor_shape.uncorrected_blade_profile_y_CAD)), 1e-3)
        blade_profile[:, 0] = np.subtract(blade_profile[:, 0], np.min(blade_profile[:, 0]))
        blade_profile[:, 1] = np.subtract(blade_profile[:, 1], self.rotor_shape.dimensional_leading_edge_thickness / 2)
        blade_profile, blade_end_cap_faces = triangulate_profile(blade_profile)
        blade_vertices, blade_faces = extrude_profile(blade_profile, blade_end_cap_faces, meanline.R_hub, meanline.R_tip)

        # Wrap the Euler-radius section around the annulus and sweep it radially to preserve the blade count and span.
        for blade_index in range(rotor_blade_count):
            blade_axial_coordinates, blade_tangential_coordinates, blade_radial_coordinates = blade_vertices.T
            blade_position_angles = np.add(np.divide(blade_tangential_coordinates, meanline.R_Euler),
                                          2 * np.pi * blade_index / rotor_blade_count)
            positioned_blade_vertices = np.column_stack((np.add(blade_axial_coordinates, meanline.s_ax),
                                                        np.multiply(blade_radial_coordinates, np.cos(blade_position_angles)),
                                                        np.multiply(blade_radial_coordinates, np.sin(blade_position_angles))))
            add_mesh(positioned_blade_vertices, blade_faces, f"Rotor blade {blade_index + 1}",
                     colors[blade_index % 2], "rotor", blade_index == 0)

        # Draw only the transparent rotor hub boundary without inventing a hub or rotor-shroud thickness.
        hub_ring_angles = np.linspace(0, 2 * np.pi, max(129, 4 * rotor_blade_count + 1))
        hub_axial_coordinates, hub_position_angles = np.meshgrid(
            [meanline.s_ax, meanline.s_ax + float(np.max(blade_profile[:, 0]))], hub_ring_angles, indexing="ij")
        hub_surface_grid = np.stack((hub_axial_coordinates, np.multiply(meanline.R_hub, np.cos(hub_position_angles)),
                                    np.multiply(meanline.R_hub, np.sin(hub_position_angles))), axis=-1)
        hub_vertices, hub_faces = grid_mesh(hub_surface_grid)
        add_mesh(hub_vertices, hub_faces[:, ::-1], "Rotor hub", "grey", "Rotor hub",
                 True, opacity=0.2)

        # Keep true geometric proportions and enable orbit, pan and wheel zoom in a local browser window.
        figure.update_layout(title="Turbine 3D geometry", template="plotly_white",
                             scene={"aspectmode": "data", "dragmode": "orbit",
                                    "xaxis_title": "Axial x (mm)", "yaxis_title": "Y (mm)", "zaxis_title": "Z (mm)",
                                    "camera": {"eye": {"x": -1.8, "y": 2.2, "z": 1.2}}},
                             showlegend=False, margin={"l": 15, "r": 15, "b": 30, "t": 45})
        figure.show(renderer="browser", config={"scrollZoom": True, "displaylogo": False, "responsive": True})
