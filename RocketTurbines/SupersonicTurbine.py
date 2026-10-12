from IdealGas import IdealGas
from Turbine1D import Turbine1D
from TraupelLossModel import TraupelLossModel
import SupersonicTurbineBlading as blading

class SupersonicTurbine:
    def __init__(self):
        """SupersonicTurbine is the class to represent a supersonic turbine, with consistent meanline design and blading
         shape. It ensures that chord/pitch ratios used in Turbine1D agrees with the values obtained from
         SupersonicTurbineBlading objects, which in turn also use Turbine1D meanline results. Thus, there is a coupling
          between meanline and blading design, which this class solves numerically. It also checks if the stator nozzle
           exit can fit optimally in the turbine annulus."""

        # Store objects

        # Store dictionaries with Turbine1D, SupersonicRotorBlade and SupersonicStatorNozzle
        # (both from SupersonicTurbineBlading library) input arguments as dictionaries. These can be used to call
        # any method ot function by passing **dictionary to them

        # Use numerical solver to that will create a consistent design, in which chord/pitch ratios assumed for sizing
        # in Turbine1D are equivalent to those obtained from SupersonicTurbineBlading

        # Obtain consistent design after solution was found

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
        # Consequently, thickness_TE_stator_optimality = t_TE_stator_min_obtained - t_TE_stator_min is calculated for
        # it, which the outer optimizer or designed must always keeep above 0 by changing number of nozzles, admission
        # fraction, etc. Rotor blade inlet and outlet angles are also used to change t_LE_rotor and t_TE_rotor to
        # t_LE_rotor_min_obtained and t_TE_rotor_min_obtained. These are then used to calculate
        # thickness_LE_rotor_optimality = t_LE_rotor_min_obtained - t_min_rotor and
        # thickness_TE_rotor_optimality = t_TE_rotor_min_obtained - t_min_rotor. To ensure fair comparison,
        # the designer or outer optimizer should always compare rotors with the same metal thickness. Consequently,
        # the numerical solver in this class will also ensure that thickness_LE_rotor_optimality and
        # thickness_TE_rotor_optimality are zero by varying t_LE_rotor and t_TE_rotor.

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
        # nozzle for the designer to worry about metal thickness. Point 1 sets the dimensions of the rectangle with
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
        # designer or outer optimizer. To do so, the minimum distance between edges of adjacent nozzles,
        # t_TE_stator_min_obtained, can be calculated. It is very close to t_TE_stator_obtained, but still
        # different. Consequently, thickness_TE_stator_margin can be calculated, which is t_TE_stator_min_obtained
        # - t_TE_stator_min. This factor must be zero or positive for geometry to be manufacturable and for point 3 to
        # be satisfied. If the designer or outer optimizer hold thickness_TE_stator_margin at least at zero, it means
        # that nozzles are not overlapping and that t_TE_stator_obtained is positive, since it is always greater than
        # thickness_TE_stator_margin. Consequently, the design is then consistent and the nozzles are feasible.

        # This was a procedure when multiple da Laval nozzles are used. If there is a single nozzle, analogous
        # procedure to singular MOC nozzle is followed. Points 2 and 3 go away. height_optimality is again calculated
        # the same way. D_ellipse is known, so admission_fraction_obtained can be calculated.
        # admission_fraction_residual can be defined, which the numerical solver must hold at zero.


        ...