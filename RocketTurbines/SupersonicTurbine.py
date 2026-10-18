from IdealGas import IdealGas
from Turbine1D import Turbine1D
from TraupelLossModel import TraupelLossModel
import SupersonicTurbineBlading as blading
from RelaxedAitken import solve_with_relaxed_aitken

class SupersonicTurbine:
    def __init__(self, gas: IdealGas, loss_model: TraupelLossModel, Turbine1D_arguments,
                 SupersonicStatorNozzle_arguments, SupersonicRotorBlades_arguments, separation_margin):
        """SupersonicTurbine is the class to represent a supersonic turbine, with consistent meanline design and blading
         shape. It ensures that chord/pitch ratios and other variables used in Turbine1D agrees with the values obtained from
         SupersonicTurbineBlading objects, which in turn also use Turbine1D meanline results. Thus, there is a coupling
          between meanline and blading design, which this class solves numerically. This object also checks if the
           stator nozzles exit can fit optimally in the turbine annulus by producing continuous coefficients than can be
            either checked bt the engineer or constrained by outer optimizer."""

        # Store objects
        self.gas = gas
        self.loss_model = loss_model

        # Store dictionaries with Turbine1D, SupersonicRotorBlade and SupersonicStatorNozzle
        # (both from SupersonicTurbineBlading library) input arguments as dictionaries. These can be used to call
        # any method or function by passing **dictionary to them
        self.Turbine1D_arguments = Turbine1D_arguments
        self.SupersonicStatorNozzle_arguments = SupersonicStatorNozzle_arguments
        self.SupersonicRotorBlades_arguments = SupersonicRotorBlades_arguments

        # Store separation margin which will be used to calculate surface Mach numbers for upper and lower rotor surface
        # in relative reference frame
        self.separation_margin = separation_margin

        # Create and store Fluid object for SupersonicTurbineBlading objects.
        self.SupersonicTurbineBlading_Fluid = blading.Fluid(ideal_gas=gas)

        # Determine number of stator nozzles, since this will determine what is the solution vector to achieve
        # consistent design. For more details, see comments in evaluate_design.
        no_nozzles = self.Turbine1D_arguments["no_blades_stator"]
        # If there is just one nozzle, admission fraction is not a free design variable anymore. In such case,
        # remove it from Turbine1D_arguments
        if no_nozzles == 1:
            del self.Turbine1D_arguments["admission_fraction"]
        # Rotor and stator objects from SupersonicTurbineBlading can adjust the blading to achieve flow angles from
        # meanline design. However, in this case, they cannot adjust the blading to achieve some real Mach number after
        # mixing. This is because isentropic values from meanline design used to construct the blading,
        # already correspond to real values to be achieved when mixing and losses occur. Consequently, if the blading
        # was to be adjusted by SupersonicTurbineBlading based on boundary layer and
        # mixing calculations to achieve real Mach number from the meanline code, this would double count the losses.
        # Consequently, if the user specifies this option in the arguments to SupersonicTurbineBlading objects, it
        # should be removed here. Raise a user warning about it too.
        if ...:
            del self.SupersonicStatorNozzle_arguments[...]
            del self.SupersonicRotorBlades_arguments[...]

        # Define a function that calculates new values of design variables from evaluate_design based on assumed ones
        def calculate_new_solution(x_0):
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
                x_1 = ...
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
                x_1 = ...
            # Return new solution estimate together with remaining results
            return x_1, turbine_meanline, stator_shape, rotor_shape, coefficients

        # Create initial solution array if there are multiple nozzles
        if no_nozzles != 1:
            x_initial = ...
        # If there is just one nozzle
        else:
            x_initial = ...

        # Use numerical solver to that will create a consistent design, in which values assumed for sizing
        # in Turbine1D are equivalent to those obtained from SupersonicTurbineBlading. Use Aitken's scheme.
        x_solution = solve_with_relaxed_aitken(...)

        # Obtain consistent design objects after solution was found
        dummy, self.turbine_meanline, self.rotor_shape, self.stator_shape, coefficients = \
            calculate_new_solution(x_solution)
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

    def evaluate_design(self, meanline_arguments):
        """This method allows to calculate the residuals between assumed values for 1D meanline calculations in
        Turbine 1D and obtained values from blade shaping with SupersonicTurbineBlading. These residuals can be driven
         to zero by a numerical solver in the constructor of this class or the engineer to obtain consistent design.
          This method also returns some remaining residuals and optimality coefficients that can be constrained by outer
           optimizer or which can inform the engineer about manufacturability & optimality of the supersonic turbine."""

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

        # FUrthermore, a word about nomenclature and definition of trailing edge thickness. In Turbine1D, TraupelLossModel,
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
        # fraction, etc. Rotor blade inlet and outlet angles are also used to change t_TE_rotor to
        # t_TE_rotor_min_obtained. These are then used to calculate
        # t_TE_rotor_optimality = t_TE_rotor_min_obtained - t_min_rotor. To ensure fair comparison,
        # the engineer or outer optimizer should always compare rotors with the same metal thickness. Consequently,
        # the numerical solver in this class will also ensure that t_TE_rotor_optimality is zero by varying t_TE_rotor.
        # To change t_TE_stator and t_TE_rotor to t_TE_stator_min_obtained and t_TE_rotor_min_obtained, flow angles
        # from meanline design are always used. These are approximately the same as nozzle/rotor blade metal angles.
        # Blade metal angles can be a few degrees different if the user specifies in input arguments to
        # SupersonicTurbineBlading objects that they should be adjusted to obtain flow angles equal to those from
        # meanline design, however this has a negligible influence. Finally, boundary layer correction for stator and
        # rotor shapes can lead to blade thickness for BL corrected shapes to be different from t_TE_stator_min_obtained
        # or t_TE_rotor_min_obtained. SupersonicRotorBlading library also uses the same blade thicknesses for leading
        # and trailing edge of the rotor.

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
        # greatest area for given annulus dimensions. Its width and height are w_nozzle and H_max.
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
        turbine_meanline = Turbine1D()
        turbine_meanline.size_turbine(**meanline_arguments)

        # Now calculate rotor surface Mach numbers based on separation margins
        # First lower Mach number
        ...
        # Now upper Mach number
        ...
        # Now calculate rotor shape and dimensions. Use real Mach number at station 1 from Turbine 1D and Mach number
        # at station 2 that assumes isentropic expansion through the rotor
        rotor_shape = ...

        # Calculate t_TE_rotor_optimality. t_TE_rotor_optimality is part of residuals, since it will be always
        # decreased to zero by the optimizer.
        ...

        # Calculate total throat area needed to satisfy massflow requirements
        ...

        # Now separate procedures outlined at the top of this method can be followed to shape nozzles in the stator and
        # determines all remaining margins, residuals and optimality coefficients
        # If MOC stator nozzles are used
        if ...:
            # First calculate reference width for the MOC nozzle such that minimum thickness is always satisfied.
            ...
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**self.SupersonicStatorNozzle_arguments, ...)

            # Calculate throat height from total throat area, reference width and the number of nozzles
            ...
            # Calculate height optimality factor for MOC nozzle exit
            ...
            # Width optimality factor and thickness margin are always zero
            ...
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes
            ...
            # Calculate trailing edge thickness residuals between meanline design and blade shapes
            ...
            # Now pack all the results and return them
            residuals = ...
            coefficients = ...
            return residuals, turbine_meanline, stator_shape, rotor_shape, coefficients

        # Follow the procedure if a single MOC stator nozzle is used
        elif ...:
            # Calculate width and height of the optimal nozzle exit shape to maximize area taken up by it
            ...
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**self.SupersonicStatorNozzle_arguments, ...)

            # From total throat area, calculate throat height
            ...
            # Calculate height optimality factor for MOC nozzle exit
            ...
            # Also calculate admission fraction residual
            ...
            # Width optimality factor and thickness margin are always zero
            ...
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes
            ...
            # Calculate trailing edge thickness residuals between meanline design and blade shapes
            ...
            # Now pack all the results and return them
            residuals = ...
            coefficients = ...
            return residuals, turbine_meanline, stator_shape, rotor_shape, coefficients
        # Follow the procedure if multiple da Laval nozzles are used
        elif ...:
            # First calculate throat diameter based on total throat area and number of nozzles
            ...
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**self.SupersonicStatorNozzle_arguments, ...)

            # Get nozzle exit width and height from stator_shape. Use stator shape uncorrected for the boundary later
            ...
            # Calculate height and width optimality factors
            ...
            # Calculate trailing edge thickness margin
            ...
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes
            ...
            # Calculate trailing edge thickness residuals between meanline design and blade shapes
            ...
            # Now pack all the results and return them
            residuals = ...
            coefficients = ...
            return residuals, turbine_meanline, stator_shape, rotor_shape, coefficients
        # Now follow the procedure if a single da Laval nozzle is used
        elif ...:
            # First calculate throat diameter based on total throat area
            ...
            # Now calculate stator shape and dimensions. Use isentropic outlet Mach number at station 1 from Turbine1D,
            # as SupersonicStatorNozzle requires ideal value.
            stator_shape = blading.SupersonicStatorNozzle(**self.SupersonicStatorNozzle_arguments, ...)

            # Calculate height optimality factor for nozzle exit
            ...
            # Also calculate admission fraction residual
            ...
            # Width optimality factor and thickness margin are always zero
            ...
            # Calculate chord - pitch ratio residuals between meanline design and blade shapes
            ...
            # Calculate trailing edge thickness residuals between meanline design and blade shapes
            ...
            # Now pack all the results and return them
            residuals = ...
            coefficients = ...
            return residuals, turbine_meanline, stator_shape, rotor_shape, coefficients