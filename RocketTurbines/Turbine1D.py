from itertools import zip_longest
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar, least_squares
from tabulate import tabulate
from TraupelLossModel import TraupelLossModel
from IdealGas import IdealGas
from RelaxedAitken import solve_with_relaxed_aitken

class Turbine1D:
    def __init__(self):
        """A class representing 1D, nonisentropic turbine. This class was created with supersonic turbines in """ \
        """mind, but can be also used for subsonic ones. It takes into account nonisentropic processes when """ \
        """calculating fluid states and flow velocities and angles. The object can automatically size the turbine """ \
        """based on given stage coefficients, massflow and shaft power through Turbine1D.size_turbine method.

        This method uses numerical solver, which calls Turbine1D.calculate_entropy_rise and finds consistent """ \
        """entropy increase through each blade row, such that assumed and calculated entropies are with agreement """ \
        """with each other. This is the outer numerical loop. During each evaluation of """ \
        """Turbine1D.calculate_entropy_rise there is one internal numerical loop.

        The internal loop calculates consistent thermodynamic state at each station, by varying h_2/h_t0 """ \
        """such that energy balance is achieved. Once h_2/h_t0 is found, """ \
        """__calculate_blade_rows_outlet_velocities updates the analysis results with the remaining outlet """ \
        """velocities, isentropic row reference states, sound speeds, Mach numbers and Reynolds numbers. """ \
        """These calculations include all stage losses. After this, the flow angles are calculated. """ \
        """__calculate_blade_row_flow_state then calculates the flow state for the blade row alone at the same outlet """ \
        """static pressure, without partial admission, clearance and disk friction losses and only taking into account aerodynamic """ \
        """losses in blade row passages. Continuity and rothalpy determine its outlet velocities and flow angles. """ \
        """Flow state for blade row alone is required by """ \
        """Traupel loss model.

        Afterwards, the rest of the geometry is calculated and the loss model is called to calculate entropy """ \
        """increase, which feeds back to outer numerical loop.

        For more documentation, sizing logic diagram, assumptions and nomenclature, see comments in """ \
        """Turbine1D.__init__.
        """

        # Sizing logic diagram:
        #  Turbine1D.size_turbine
        #    |
        #    +--> OUTER LOOP ---------------------------------------------------------------+
        #    |      Vary: stator, rotor aerodynamic and additional rotor entropy rises      |
        #    |      Residuals: calculated minus assumed entropy rise for each loss group   |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    Turbine1D.calculate_entropy_rise                                          |
        #    |      |                                                                       |
        #    |      +--> INNER LOOP -------------------------------------------+            |
        #    |      |      Vary: h_2/h_t0                                      |            |
        #    |      |      Residual: calculated minus prescribed outlet        |            |
        #    |      |                total enthalpy, divided by h_t0           |            |
        #    |      |      |                                                   |            |
        #    |      |      v                                                   |            |
        #    |      |    Turbine1D.__calculate_thermodynamic_properties --------+            |
        #    |      |                                                                       |
        #    |      v  (after inner loop converges)                                         |
        #    |    Analysis results: station states and absolute rotor outlet velocity       |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    Turbine1D.__calculate_blade_rows_outlet_velocities                         |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    Updated analysis results: outlet velocities, Mach and Reynolds numbers    |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    Turbine1D.__calculate_flow_angles                                          |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    Analysis results with flow angles                                         |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    Turbine1D.__calculate_blade_row_flow_state                                 |
        #    |      Prescribed: stage outlet static pressure and aerodynamic entropy rise   |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    Blade-row results: states, velocities and flow angles                       |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    Turbine1D.__calculate_geometry                                             |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    Turbine geometry, together with analysis and blade-row results             |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    TraupelLossModel.calculate_entropy_increase                                 |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    Calculated entropy rises and loss results                                  |
        #    |      |                                                                       |
        #    |      +--> Calculate entropy-rise residuals ----------------------------------+
        #    |
        #    v  (after outer loop converges)
        #  Consistent entropy rises, analysis results, blade-row results and loss results
        #    |
        #    v
        #  Turbine1D.__calculate_geometry
        #    |
        #    v
        #  Sized turbine geometry and stored design-point results

        # Nomenclature:
        # Total properties have _t prefix. Static properties are assumed by default, they do not have any prefix.
        # Properties in stationary and rotary reference frame have _s and _r prefixes respectively.
        # Real properties are assumed by default, while ideal/isentropic ones have _ideal prefix.
        # _ideal_r_real_s means ideal expansion at the rotor, but non-ideal expansion at the stator.
        # Station 0 is station before stator, station 1 is after stator and station 2 is after rotor.
        # _blade specifies that the results were calculated only for the blade row and its passage aerodynamic losses,
        # without an inclusion of additional losses like clearances, disk friction or partial admission. In other words,
        # _blade represent velocities or thermodynamic properties representative of flow in stationary test cascade,
        # at the same outlet static pressure as the whole stage.
        # Angles refer to flow angles and are stored in analysis_results or blade_row_results.
        # _ts means total-to-static quantity, _ss means static-to-static quantity, _tt means total-to-total quantity.

        # Assumptions:
        # 1. This class utilizes ideal gas model for calculations.
        # 2. Velocity at station 0 is assumed to be negligible.
        # 3. Flow areas at station 1 and 2 are equal.
        # 4. Constant Euler diameter.
        # 5. Axial width of the rotor blade row assumed to be equal to chord of the blade.

        # Geometry stored in the object properties. Metal angles are not calculated by the code since they may depend
        # on specific blade profiles - only flow angles are calculated. These are stored in dictionaries below.
        self.D_hub = None   # Hub diameter, m
        self.D_tip = None   # Tip diameter, m
        self.D_mean = None  # Mean diameter, m
        self.D_Euler = None # Euler diameter, m
        self.R_hub = None   # Hub radius, m
        self.R_tip = None   # Tip radius, m
        self.R_mean = None  # Mean radius, m
        self.R_Euler = None # Euler radius, m
        self.R_nozzle = None # Stator nozzle exit-centre radius, m
        self.l_stator = None  # Stator blade/nozzle radial length, m
        self.l_rotor = None # Rotor blade radial length, m
        self.A_throat_stator = None # Total stator throat area, m^2
        self.A_throat_rotor = None # Total admitted rotor throat area, m^2
        self.s_ax = None    # Axial distance between stator blade/nozzles and rotor blades, m
        self.s_r = None     # Radial distance between rotor blades / shroud and the casing, m
        self.shrouded_rotor = None # Boolean whether rotor is shrouded
        self.s_ax_shroud = None # Axial distance between rotor shroud and the casing, m
        self.t_shroud = None # Rotor shroud thickness, m
        self.h_shroud = None # Rotor blade overlap with the casing (how much above outer stator diameter rotor blades
        # end and shroud begins when it is used), m
        self.h_shroud_over_blade_length = None # Overlap of the blade with the casing over rotor blade length, -
        self.s_ax_shroud_over_h_shroud = None # Axial distance between shroud and the casing over blade overlap with
        # it, -
        self.seal_teeth_number = None # Number of teeth in shroud's half-labirynth seal, -.
        self.seal_teeth_spacing = None # Spacing between teeth in shroud's half-labirynth seal, m.
        self.c_stator = None # Stator chord, m. Assumed to be equal to stator blade width / length of projection over
        # meridional axis.
        self.p_stator = None # Stator pitch at Euler radius, m
        self.c_rotor = None # Rotor chord, m. Assumed to be equal to rotor blade width / length of projection over
        # meridional axis
        self.p_rotor = None # Rotor pitch at Euler radius, m
        self.t_TE_stator = None # Stator trailing edge thickness at Euler radius, m.
        self.t_LE_rotor = None # Circumferential rotor leading edge thickness, m.
        self.t_TE_rotor = None # Rotor trailing edge thickness, m.
        self.no_blades_rotor = None # Number of blades in the rotor, -
        self.no_blades_stator = None # Number of nozzles/blades in the stator, -
        self.admission_fraction = None # Admission fraction of the turbine stage, -
        self.partial_admission_rotor = None # Type of partial admission rotor, either "free" or "enclosed"
        self.sand_grain_roughness = None # Sound grain equivalent roughness, m

        # Analysis results at the design point
        self.analysis_results_at_design_point = {"gas": None, # IdealGas object
                                                 "mdot_total": None, # Total massflow through the turbine, kg/s
                                                 "RPM": None, # Rotations per minute, rpm
                                                 "omega": None, # Angular velocity, rad/s
                                                 "T_0": None,   # Total/static temperature at station 0, K
                                                 "p_0": None,   # Total/static pressure at station 0, Pa
                                                 "loss_model": None,  # TraupelLossModel object used for calculations, -
                                                 "h_0": None,   # Total/static enthalpy at station 0, J/kg
                                                 "eta_tt": None, # Total-to-total efficiency, -
                                                 "eta_ts": None, # Total-to-static efficiency, -
                                                 "eta_poly_tt": None, # Total-to-total polytropic efficiency, -
                                                 "eta_poly_ts": None, # Total-to-static polytropic efficiency, -
                                                 "zeta_stator_Denton": None, # Stator entropy loss coefficient based
                                                 # on T_2 and absolute stator outlet kinetic energy, according to Denton
                                                 # definition, -
                                                 "zeta_rotor_Denton": None, # Rotor entropy loss coefficient including
                                                 # additional losses, based on T_2 and relative outlet kinetic energy,
                                                 # according to Denton definition -
                                                 "psi_tt": None, # Total-to-total real work coefficient, -
                                                 "psi_tt_ideal": None, # Total-to-total ideal work coefficient, -
                                                 "psi_ss_ideal": None,  # Static-to-static ideal work coefficient, -
                                                 "psi_ss_ideal_design": None,  # Static-to-static ideal work coefficient
                                                 # at the design point, -
                                                 "theta_1": None, # Flow coefficient at station 1, -
                                                 "theta_2": None, # Flow coefficient at station 2, -
                                                 "R_h_tt_ideal": None, # Ideal enthalpic reaction of the stage based on
                                                 # total-to-total enthalpy drop of the stage, -
                                                 "R_h_tt": None,   # Real enthalpic reaction of the stage based on
                                                 # total-to-total enthalpy drop of the stage, -
                                                 "R_p": None,   # Real pressure reaction of the stage, -
                                                 "p_0_over_p_2": None,  # Pressure ratio of the stage, -
                                                 "u": None, # Blade velocity at Euler radius, m/s
                                                 "P_shaft": None,   # Shaft power of the turbine, W
                                                 "p_1": None,   # Static pressure at station 1, Pa
                                                 "p_t1": None,  # Total pressure at station 1, Pa
                                                 "T_1": None,   # Static temperature at station 1, K
                                                 "rho_1": None, # Static density at station 1, kg/m3
                                                 "T_1_ideal": None, # Static temperature assuming ideal expansion at
                                                 # station 1, K
                                                 "h_1": None, # Static enthalpy at station 1, J/kg
                                                 "h_t1": None,  # Total enthalpy at station 1, J/kg
                                                 "h_1_ideal": None, # Static enthalpy assuming ideal expansion at
                                                 # station 1, J/kg
                                                 "delta_s_stator": None, # Total entropy increase at stator, J/kg/K
                                                 "a_1": None,   # Sound velocity at station 1, m/s
                                                 "a_1_ideal": None, # Sound velocity assuming ideal expansion at
                                                 # station 1, m/s
                                                 "v_1": None, # Absolute velocity at station 1, m/s
                                                 "v_1_ideal": None, # Absolute velocity assuming ideal expansion at
                                                 # station 1, m/s
                                                 "w_1": None, # Relative velocity at station 1, m/s
                                                 "M_s1": None, # Station 1 Mach number in stationary frame, -
                                                 "M_s1_ideal": None, # Mach number in stationary reference frame
                                                 # assuming ideal expansion at station 1, -
                                                 "M_r1": None, # Mach number in rotary reference frame at station 1, -
                                                 "M_ax_1": None, # Axial Mach number at station 1, -
                                                 "p_2": None, # Static pressure at station 2, Pa
                                                 "p_t2": None, # Total pressure at station 2, Pa
                                                 "T_2": None, # Static temperature at station 2, K
                                                 "rho_2": None, # Static density at station 2, kg/m3
                                                 "T_2_ideal_r_real_s": None, # Static temperature assuming
                                                 # ideal expansion at station 2, K
                                                 "h_2": None, # Static enthalpy at station 2, J/kg
                                                 "h_t2": None, # Total enthalpy at station 2, J/kg
                                                 "h_2_ideal_r_real_s": None, # Static enthalpy assuming ideal
                                                 # expansion at station 2, J/kg
                                                 "delta_h_t": None, # Total enthalpy drop through the stage, J/kg
                                                 "delta_h_ideal": None,  # Isentropic enthalpy drop through
                                                 # the stage, J/kg
                                                 "delta_s_rotor": None, # Entropy increase at rotor due to aerodynamic
                                                 # losses in the blade row, J/kg/K
                                                 "delta_s_rotor_additional": None,  # Entropy increase at rotor
                                                 # due to additional losses like clearances, disk friction or partial
                                                 # admission, J/kg/K
                                                 "delta_h_stator_ideal": None,  # Isentropic enthalpy drop through the
                                                 # stator, J/kg
                                                 "delta_h_rotor_ideal": None,  # Isentropic enthalpy drop through the
                                                 # rotor assuming real expansion through the stator, J/kg
                                                 "a_2": None, # Sound velocity at station 2, m/s
                                                 "a_2_ideal_r_real_s": None, # Sound velocity assuming ideal
                                                 # expansion at station 2, m/s
                                                 "v_2": None, # Absolute velocity at station 2, m/s
                                                 "w_2": None, # Relative velocity at station 2, m/s
                                                 "w_2_ideal_r_real_s": None, # Relative velocity assuming ideal
                                                 # expansion at station 2, m/s
                                                 "M_s2": None, # Station 2 Mach number in stationary frame, -
                                                 "M_r2": None, # Mach number in rotary reference frame at station 2, -
                                                 "M_ax_2": None, # Axial Mach number at station 2, -
                                                 "M_r2_ideal_r_real_s": None, # Mach number in rotary reference
                                                 # frame assuming ideal expansion at station 2, -
                                                 "M_r2_ideal_r_real_s_construction": None, # Mach number used for rotor
                                                 # blade construction. It is in rotary reference frame and assumes ideal
                                                 # expansion at station 2, -; set by SupersonicTurbine, -
                                                 "Re_s1": None, # Reynolds number at station 1 in stationary reference
                                                 # frame, -
                                                 "Re_r2": None, # Reynolds number based on relative velocity at
                                                 # station 2 and rotor chord, -
                                                 "alpha_1": None, # Absolute flow angle at station 1, rad
                                                 "alpha_2": None, # Absolute flow angle at station 2, rad
                                                 "beta_1": None, # Relative flow angle at station 1, rad
                                                 "beta_2": None, # Relative flow angle at station 2, rad
                                                 "alpha_1_deg": None,  # Absolute flow angle at station 1, degree
                                                 "alpha_2_deg": None,  # Absolute flow angle at station 2, degree
                                                 "beta_1_deg": None,  # Relative flow angle at station 1, degree
                                                 "beta_2_deg": None,  # Relative flow angle at station 2, degree
                                                 "incidence": None, # Incidence angle between flow and rotor blade, rad
                                                 "incidence_deg": None, # Incidence angle between flow and rotor blade,
                                                 # deg
                                                 }
        # Analysis results at the design point for the blade row alone at the stage outlet static pressure, without
        # clearance, disk friction or partial admission losses.
        self.blade_row_results_at_design_point = {
            "v_1_blade": None,  # Absolute velocity at station 1 for the blade row alone, m/s
            "w_1_blade": None,  # Relative velocity at station 1 for the blade row alone, m/s
            "p_1_blade": None,  # Static pressure at station 1 for the blade row alone, Pa
            "T_1_blade": None,  # Static temperature at station 1 for the blade row alone, K
            "h_1_blade": None,  # Static enthalpy at station 1 for the blade row alone, J/kg
            "rho_1_blade": None,  # Static density at station 1 for the blade row alone, kg/m3
            "M_s1_blade": None,  # Absolute Mach number at station 1 for the blade row alone, -
            "v_2_blade": None,  # Absolute velocity at station 2 for the blade row alone, m/s
            "w_2_blade": None,  # Relative velocity at station 2 for the blade row alone, m/s
            "p_2_blade": None,  # Static pressure at station 2 for the blade row alone, Pa
            "p_t2_blade": None,  # Total pressure at station 2 for the blade row alone, Pa
            "T_2_blade": None,  # Static temperature at station 2 for the blade row alone, K
            "h_2_blade": None,  # Static enthalpy at station 2 for the blade row alone, J/kg
            "h_t2_blade": None,  # Total specific enthalpy at station 2 for the blade row alone, J/kg
            "rho_2_blade": None,  # Static density at station 2 for the blade row alone, kg/m3
            "M_r1_blade": None,  # Relative Mach number at station 1 for the blade row alone, -
            "M_r2_blade": None,  # Relative Mach number at station 2 for the blade row alone, -
            "theta_2_blade": None,  # Flow coefficient at station 2 for the blade row alone, -
            "alpha_2_blade": None,  # Absolute flow angle at station 2 for the blade row alone, rad
            "beta_2_blade": None,  # Relative flow angle at station 2 for the blade row alone, rad
            "alpha_2_blade_deg": None,  # Absolute flow angle at station 2 for the blade row alone, deg
            "beta_2_blade_deg": None,  # Relative flow angle at station 2 for the blade row alone, deg
            "psi_tt_blade": None,  # Total-to-total work coefficient for the blade row alone, -
            "R_h_tt_blade": None,  # Enthalpic reaction for the blade row alone based on total-to-total enthalpy drop of
            # the stage, -
            "p_0_over_p_2_blade": None,  # Pressure ratio for the blade row alone, -
            "Re_s1_blade": None,  # Reynolds number based on v_1_blade and stator chord, -
            "Re_r2_blade": None,  # Reynolds number based on w_2_blade and rotor chord, -
        }

        # Traupel loss analysis results at the design point.
        self.loss_model_results_at_design_point = {
            "zeta_aerodynamic_stator": None,  # Stator blade-row aerodynamic loss coefficient, -
            "zeta_incidence_stator": None,  # Stator incidence loss coefficient, -
            "zeta_aerodynamic_rotor": None,  # Rotor blade-row aerodynamic loss coefficient, -
            "zeta_incidence_rotor": None,  # Rotor incidence loss coefficient, -
            "zeta_disk_friction": None,  # Rotor disk-friction loss coefficient, -
            "zeta_clearance_rotor": None,  # Rotor tip or shroud-clearance loss coefficient, -
            "zeta_admission": None,  # Rotor partial-admission loss coefficient, -
            "zeta_rotor_additional": None,  # Sum of rotor disk-friction, clearance and admission coefficients, -
            "delta_h_loss_stator": None,  # Specific enthalpy dissipated in the stator, J/kg
            "delta_h_loss_rotor": None,  # Specific enthalpy dissipated in the rotor blade row, J/kg
            "delta_h_loss_rotor_additional": None,  # Specific work lost due to additional rotor losses, J/kg
            "delta_h_loss_rotor_sum": None,  # Sum of enthalpy dissipated in the rotor blade-row and total
            # work lost due to additional losses, J/kg
            "delta_h_loss_rotor_sum_static": None, # Sum of dissipated enthalpy in the rotor blade row and static
            # enthalpy increment through the rotor due to additional losses, J/kg
            "delta_s_stator": None,  # Specific entropy rise across the stator, J/(kg K)
            "delta_s_rotor": None,  # Specific entropy rise across the rotor blade row, J/(kg K)
            "delta_s_rotor_additional": None,  # Specific entropy rise from additional rotor losses, J/(kg K)
            "mdot_leak": None,  # Mass flow leaking through the rotor shroud seal (zero if unshrouded), kg/s
            # Breakdown of the stator aerodynamic loss coefficient:
            "zeta_p0_stator": None,  # Uncorrected stator profile loss coefficient, -
            "corrected_zeta_p0_stator": None,  # Stator profile loss after Mach and Reynolds corrections, -
            "zeta_h_stator": None,  # Stator trailing-edge underpressure loss coefficient, -
            "zeta_C_stator": None,  # Stator trailing-edge mixing/blockage loss coefficient, -
            "zeta_p_stator": None,  # Total stator profile loss coefficient, -
            "zeta_f_stator": None,  # Stator fanning loss coefficient, -
            "integrated_zeta_p_stator": None,  # Sum of stator profile and fanning loss coefficients, -
            "zeta_a_stator": None,  # Stator endwall loss coefficient in the axial gap, -
            "zeta_short_blade_stator": None,  # Blade-length contribution to stator residual loss, -
            "zeta_rest_stator": None,  # Total stator residual endwall loss coefficient, -
            # Breakdown of the rotor aerodynamic loss coefficient:
            "zeta_p0_rotor": None,  # Uncorrected rotor profile loss coefficient, -
            "corrected_zeta_p0_rotor": None,  # Rotor profile loss after Mach and Reynolds corrections, -
            "zeta_h_rotor": None,  # Rotor trailing-edge underpressure loss coefficient, -
            "zeta_C_rotor": None,  # Rotor trailing-edge mixing/blockage loss coefficient, -
            "zeta_p_rotor": None,  # Total rotor profile loss coefficient, -
            "zeta_f_rotor": None,  # Rotor fanning loss coefficient, -
            "integrated_zeta_p_rotor": None,  # Sum of rotor profile and fanning loss coefficients, -
            "zeta_a_rotor": None,  # Rotor endwall loss coefficient in the axial shroud gap, -
            "zeta_short_blade_rotor": None,  # Blade-length contribution to rotor residual loss, -
            "zeta_rest_rotor": None,  # Total rotor residual endwall loss coefficient, -
        }

    def size_turbine(self, gas: IdealGas, work_coefficient_tt, flow_coefficient, reaction_isentropic_tt, RPM,
                     shaft_power, mdot, T_0, p_0, radial_clearance, no_blades_stator, no_blades_rotor,
                     chord_over_pitch_stator, t_TE_stator, t_TE_rotor, Ra_roughness, loss_model: TraupelLossModel,
                     admission_fraction=1, partial_admission_rotor="free", chord_over_pitch_rotor=2.5,
                     s_ax_over_pitch_rotor=0.35, delta_s_estimate=[0, 0, 0], shrouded_rotor=False,
                     h_shroud_over_blade_length=0.008, s_ax_shroud_over_h_shroud=2, t_shroud=None,
                     seal_teeth_number=None):
        """A method to size the turbine based on given requirements. It changes properties of the object.
        This method calls calculate_entropy_rise, which is an outer numerical solution loop, to calculate consistent
        fluid and flow state at each station. This allows to calculate the geometry of the turbine.

        :param IdealGas gas: Working gas of the turbine. Must be an instance of the IdealGas class.
        :param float or integer work_coefficient_tt: Real, nonisentropic work coefficient of the stage based on
         total-to-total enthalpy drop.
        :param float or integer flow_coefficient: Flow coefficient of the turbine at station 2.
        :param float reaction_isentropic_tt: Enthalpic, isentropic reaction of the turbine. This is static enthalpy drop
         of the rotor over total-to-total enthalpy drop of the stage for isentropic turbine. It should be noted
          enthalpic reaction is not the same as pressure reaction, although two quantities are similar.
        :param float or integer RPM: Rotations per minute of the turbine.
        :param float or integer shaft_power: Shaft power (W) that the turbine must deliver.
        :param float or integer mdot: Massflow (kg/s) through the turbine.
        :param float or integer T_0: Total temperature (K) at station 0. It is assumed it is approximately equal to
            static temperature there due to negligible velocity.
        :param float or integer p_0: Total pressure (Pa) at station 0. It is assumed it is approximately equal to
            static pressure there due to negligible velocity.
        :param float or integer radial_clearance: Radial clearance (m) of the rotor blade.
        :param integer no_blades_stator: Number of stator blades/nozzles within the admitted circumference (-).
        :param integer no_blades_rotor: Number of rotor (-) blades.
        :param float or integer chord_over_pitch_stator: Ratio of stator chord to its blade/nozzle pitch (-).
        :param float or integer chord_over_pitch_rotor: Ratio of rotor chord to its blade pitch (-). By default,
         equal to 2.5, which is in the range of 2.5-3.3 recommended for supersonic impulse rotor cascades by Traupel
         in "Thermal Turbomachines".
        :param float or integer admission_fraction: Admission fraction (-) of the turbine stage. By default, 1.
        :param string partial_admission_rotor: Whether partial admission rotor (if used) is "enclosed" or "free".
         By default, "free".
        :param float or integer t_TE_stator: Circumferential thickness (m) of the trailing edge for the stator.
         In other words, length of the projection of the thickness on the circumferential axis.
        :param float or integer t_TE_rotor: Circumferential thickness (m) of the trailing edge for the rotor.
         In other words, length of the projection of the thickness on the circumferential axis.
        :param float or integer Ra_roughness: Roughness of the turbine flow surfaces in Ra (m).
        :param float or integer s_ax_over_pitch_rotor: Axial distance between stator and rotor over rotor pitch (-).
         By default, 0.35, which is in the range of 0.3-0.35 given by Traupel.
        :param boolean shrouded_rotor: Boolean whether the rotor is shrouded. By default, False.
        :param float h_shroud_over_blade_length: Overlap of the blade with the casing over rotor blade length.
         By default, 0.008, which is below the value of 0.01 recommended by Traupel.
        :param float or integer s_ax_shroud_over_h_shroud: Axial distance between shroud and the casing over blade
         overlap with it. By default, 2, which is above the value of 1.5 recommended by Traupel.
        :param float or integer t_shroud: Shroud thickness, m. By default, None.
        :param integer seal_teeth_number: Number of half-labirynth seal teeth. Must be greater than 2.
         By default, None.
        :param TraupelLossModel loss_model: Traupel loss model to be used in the analysis.
        :param list delta_s_estimate: List of initial estimates of the entropy rises (J/kg/K) due to stator, rotor and
         additional rotor losses. Additional rotor losses represent clearance, partial admission or disk friction
          losses. The default [0, 0, 0] uses the loss-model result evaluated at zero entropy rises instead.
          Each nonfinite or negative loss-model estimate is replaced by half its corresponding input estimate.
        :raises TypeError: If gas is not an IdealGas instance or loss_model is not a TraupelLossModel instance.
        :raises ValueError: If the specified shaft work and outlet axial kinetic energy leave no positive outlet
            static enthalpy, or the prescribed blade-row state is infeasible.
        :raises RuntimeError: If the entropy-rise solve does not converge, returns nonfinite or negative entropy
            rises, or has a nonfinite or excessive entropy residual (absolute value must be below 1e-3 J/(kg K)),
             or the h_2/h_t0 solve fails its convergence, solution value or residual checks.
        """

        # Verify that user inputs are correct.
        if not isinstance(loss_model, TraupelLossModel):
            raise TypeError("loss_model must be a TraupelLossModel instance.")
        # Require the IdealGas class used by the turbine's thermodynamic calculations.
        if not isinstance(gas, IdealGas):
            raise TypeError("gas must be an IdealGas instance.")

        # Now verify all the scalars.
        positive_inputs = {
            "work_coefficient_tt": work_coefficient_tt, "flow_coefficient": flow_coefficient,
            "RPM": RPM, "shaft_power": shaft_power, "mdot": mdot, "T_0": T_0, "p_0": p_0,
            "no_blades_stator": no_blades_stator, "no_blades_rotor": no_blades_rotor,
            "chord_over_pitch_stator": chord_over_pitch_stator,
            "chord_over_pitch_rotor": chord_over_pitch_rotor, "admission_fraction": admission_fraction,
            "radial_clearance": radial_clearance, "s_ax_over_pitch_rotor": s_ax_over_pitch_rotor
        }
        nonnegative_inputs = {
            "t_TE_stator": t_TE_stator, "t_TE_rotor": t_TE_rotor, "Ra_roughness": Ra_roughness,
            "reaction_isentropic_tt": reaction_isentropic_tt
        }
        shroud_inputs = {"h_shroud_over_blade_length": h_shroud_over_blade_length,
                         "s_ax_shroud_over_h_shroud": s_ax_shroud_over_h_shroud}
        if shrouded_rotor:
            positive_inputs.update(shroud_inputs)
        else:
            nonnegative_inputs.update({name: value for name, value in shroud_inputs.items() if value is not None})
        # Optional shroud dimensions and tooth count must be also valid whenever they are supplied.
        if t_shroud is not None:
            positive_inputs["t_shroud"] = t_shroud
        if seal_teeth_number is not None:
            positive_inputs["seal_teeth_number"] = seal_teeth_number
        # Check variables in dictionaries above.
        for name, value in {**positive_inputs, **nonnegative_inputs}.items():
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)):
                raise ValueError(f"{name} must be a finite real scalar.")
        for name, value in positive_inputs.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        # Zero clearances, trailing-edge thickness and roughness remain valid limiting inputs.
        for name, value in nonnegative_inputs.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")

        # Restrict conventional turbine designs to isentropic reaction at or below one.
        if reaction_isentropic_tt > 1:
            raise ValueError(f"reaction_isentropic_tt must be smaller than or equal to one.")

        # Admission must represent a nonzero fraction of the circumference, with a supported rotor type.
        if admission_fraction > 1:
            raise ValueError("admission_fraction must satisfy 0 < admission_fraction <= 1.")
        if not isinstance(partial_admission_rotor, str) or partial_admission_rotor not in ("free", "enclosed"):
            raise ValueError('partial_admission_rotor must be "free" or "enclosed".')

        # Blade and seal-tooth counts must be whole numbers; the seal requires more than two teeth.
        for name, value in {"no_blades_stator": no_blades_stator,
                            "no_blades_rotor": no_blades_rotor}.items():
            if value != int(value):
                raise ValueError(f"{name} must be a positive integer.")
        if seal_teeth_number is not None and (seal_teeth_number != int(seal_teeth_number)
                                              or seal_teeth_number <= 2):
            raise ValueError("seal_teeth_number must be an integer greater than 2.")
        # A shrouded rotor needs a defined seal and positive overlap; optional thickness may remain None.
        if not isinstance(shrouded_rotor, (bool, np.bool_)):
            raise TypeError("shrouded_rotor must be a boolean.")
        if shrouded_rotor and seal_teeth_number is None:
            raise ValueError("seal_teeth_number is required for a shrouded rotor.")
        if shrouded_rotor and h_shroud_over_blade_length <= 0:
            raise ValueError("h_shroud_over_blade_length must be positive for a shrouded rotor.")

        # The entropy solver needs exactly three finite, nonnegative initial entropy estimates.
        try:
            valid_estimate_length = len(delta_s_estimate) == 3
        except TypeError:
            valid_estimate_length = False
        if not valid_estimate_length:
            raise ValueError("delta_s_estimate must contain exactly three entropy estimates.")
        for value in delta_s_estimate:
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or value < 0):
                raise ValueError("delta_s_estimate must contain finite, nonnegative real scalars.")

        # Shaft work and the prescribed exit axial velocity must leave positive exit static enthalpy.
        available_exit_static_enthalpy = gas.Cp * T_0 - (shaft_power / mdot) * \
            (1 + flow_coefficient**2 / (2 * work_coefficient_tt))
        if not np.isfinite(available_exit_static_enthalpy) or available_exit_static_enthalpy <= 0:
            raise ValueError("Shaft work and exit axial kinetic energy leave no positive exit static enthalpy.")
        # Tangential trailing-edge thickness must be smaller than its corresponding blade pitch.
        validation_D_Euler = 2 * np.sqrt((shaft_power / mdot) / work_coefficient_tt) / (RPM * 2 * np.pi / 60)
        if (t_TE_stator >= admission_fraction * validation_D_Euler * np.pi / no_blades_stator
                or t_TE_rotor >= validation_D_Euler * np.pi / no_blades_rotor):
            raise ValueError("Each trailing-edge thickness must be smaller than its blade pitch.")

        # Set object properties to None's in case the turbine is being resized.
        for name, value in vars(self).items():
            setattr(self, name, dict.fromkeys(value) if isinstance(value, dict) else None)

        # Calculate the blade speed from given requirements
        delta_h_t = shaft_power / mdot  # J/kg
        u = np.sqrt(delta_h_t / work_coefficient_tt)  # m/s

        # Calculate angular speed and Euler diameter of the turbine
        omega = RPM * 2 * np.pi / 60  # rad/s
        self.D_Euler = 2 * u / omega  # m
        self.R_Euler = self.D_Euler / 2  # m

        # Calculate thermodynamic properties which are already known
        h_t0 = gas.Cp * T_0 # J/kg
        h_t2 = h_t0 - delta_h_t # J/kg
        T_t2 = h_t2 / gas.Cp # K

        # Calculate available geometry already
        self.p_stator = admission_fraction * self.D_Euler * np.pi / no_blades_stator  # m
        self.p_rotor = self.D_Euler * np.pi / no_blades_rotor  # m
        self.c_stator = self.p_stator * chord_over_pitch_stator  # m
        self.c_rotor = self.p_rotor * chord_over_pitch_rotor  # m
        self.s_ax = self.p_rotor * s_ax_over_pitch_rotor  # m

        # Assign remaining geometry that is already known
        self.t_TE_rotor = t_TE_rotor  # m
        # Assume equal rotor LE/TE thicknesses. This can be overwritten by SupersonicTurbine class.
        self.t_LE_rotor = t_TE_rotor  # m
        self.t_TE_stator = t_TE_stator  # m
        self.admission_fraction = admission_fraction  # -
        self.s_r = radial_clearance  # m
        self.no_blades_rotor = no_blades_rotor  # -
        self.no_blades_stator = no_blades_stator  # -
        self.shrouded_rotor = shrouded_rotor
        self.t_shroud = t_shroud  # m
        self.h_shroud_over_blade_length = h_shroud_over_blade_length  # -
        self.s_ax_shroud_over_h_shroud = s_ax_shroud_over_h_shroud  # -
        self.partial_admission_rotor = partial_admission_rotor
        self.seal_teeth_number = seal_teeth_number  # -
        if self.shrouded_rotor: self.seal_teeth_spacing = self.c_rotor / (seal_teeth_number - 1)  # m
        # If not shrouded, set all shroud related geometry to None in case turbine is resized.
        if not self.shrouded_rotor:
            self.seal_teeth_spacing = None
            self.seal_teeth_number = None
            self.t_shroud = None
            self.h_shroud = None
            self.s_ax_shroud = None
            self.h_shroud_over_blade_length = None
            self.s_ax_shroud_over_h_shroud = None
        # Change Ra roughness to sand grain equivalent roughness
        self.sand_grain_roughness = 5.863 * Ra_roughness # m

        # Put known variables in analysis_results_at_design_point already
        self.analysis_results_at_design_point.update({"gas": gas,
                                                      "psi_tt": work_coefficient_tt,  # -
                                                      "theta_2": flow_coefficient,  # -
                                                      "R_h_tt_ideal": reaction_isentropic_tt,  # -
                                                      "RPM": RPM,  # rpm
                                                      "omega": omega,  # rad/s
                                                      "u": u,  # m/s
                                                      "P_shaft": shaft_power,  # W
                                                      "mdot_total": mdot,  # kg/s
                                                      "delta_h_t": delta_h_t, # J/kg
                                                      "T_0": T_0,  # K
                                                      "p_0": p_0,  # Pa
                                                      "h_t0": h_t0, # J/kg
                                                      "h_t2": h_t2, # J/kg
                                                      "T_t2": T_t2, # K
                                                      "loss_model": loss_model})


        # Calculate entropy rise across each blade row. Entropy rise depends on the loss model, which depends on the
        # velocity, and thus entropy as well. The model is thus implicit and requires a numerical solve.
        # First define the function to get entropy residual
        def get_entropy_residual(delta_s):
            return self.calculate_entropy_rise(delta_s[0], delta_s[1], delta_s[2], loss_model)[0]
        # The fixed-point map returns calculated entropy rises rather than their residuals.
        def get_new_entropy(delta_s):
            return np.asarray(get_entropy_residual(delta_s)) + delta_s
        # Estimate losses with input entropy rise (delta_s_estimate), then use calculated entropy rise as the default
        # initial solution guess and variable scales for numerical solver. The reason entropy change is estimated this
        # way, is because delta_s_estimate can be a list of zeros, which is a poor initial solution for a real turbine
        # flow, and it is also non-feasible scales for normalizing the variables.
        calculated_delta_s_estimate = np.array(get_entropy_residual(delta_s_estimate)) + np.array(delta_s_estimate)
        # Replace each negative or nonfinite estimate with half its supplied initial estimate. Half of supplied initial
        # estimate is used, so that negative / nonfinite entropy change is not obtained again. In a way, this works as
        # underrelaxed fixed point iteration.
        invalid_estimate = ~np.isfinite(calculated_delta_s_estimate) | (calculated_delta_s_estimate < 0)
        calculated_delta_s_estimate = np.where(invalid_estimate, 0.5 * np.asarray(delta_s_estimate),
                                               calculated_delta_s_estimate)
        # First, use calculated entropy estimates as scales to normalise variables and speed up convergence.
        # To avoid division by zero, ensure minimum scale is 1. That floor value is rarely crossed for designs with
        # losses, so it is deemed a good choice.
        entropy_scales = np.maximum(calculated_delta_s_estimate, 1.)  # J/(kg K)
        # Set initial solution to calculated estimates
        initial_entropy = calculated_delta_s_estimate
        # Try vector Aitken first, using the same initial entropy, scales and bounds as the least_squares fallback.
        try:
            entropy = solve_with_relaxed_aitken(
                get_new_entropy, initial_entropy, initial_relaxation=0.3, min_relaxation=0.05, max_relaxation=3.,
                bounds=(np.zeros(3), 5 * np.maximum(entropy_scales, initial_entropy)), x_scale=entropy_scales,
                xtol=1e-3, maxiter=300)
        # If this does not work, use least_squares fallback
        except (ValueError, RuntimeError, ArithmeticError):
            # Call the numerical solver. The upper bound is five times multiple of the larger of entropy_scales and
            # initial_entropy.
            entropy_solution = least_squares(get_entropy_residual, initial_entropy,
                                            bounds=(np.zeros(3), 5 * np.maximum(entropy_scales, initial_entropy)),
                                            method="dogbox", jac="2-point", x_scale=entropy_scales,
                                            loss="linear", tr_solver="exact", diff_step=1e-4,
                                            xtol=1e-5, ftol=None, gtol=1e-5, max_nfev=300, verbose=2)
            # Raise an error if not converged
            if not entropy_solution.success:
                raise RuntimeError("Numerical solve for the stator and rotor entropy rises did not converge.")
            entropy = entropy_solution.x
        # Entropy rises must be finite and nonnegative; zero is an admissible lossless limit.
        if not np.all(np.isfinite(entropy)) or np.any(entropy < 0):
            raise RuntimeError("Numerical solve for the stator and rotor entropy rises returned nonfinite or negative"
                               " values.")
        # Get entropy increases and the rest of the results
        delta_s_stator, delta_s_rotor, delta_s_rotor_additional = entropy  # J/(kg K)
        entropy_residual, analysis_results, blade_row_results, loss_model_results = \
            self.calculate_entropy_rise(delta_s_stator, delta_s_rotor, delta_s_rotor_additional, loss_model)
        # Require each unscaled final entropy residual to be finite and below 1e-3 J/(kg K) in absolute value.
        entropy_residual = np.array(entropy_residual)  # J/(kg K)
        if (not np.all(np.isfinite(entropy_residual))
                or np.any(np.abs(entropy_residual) >= 1e-3)):
            raise RuntimeError("Numerical solve for the stator and rotor entropy rises has an invalid or excessive "
                               f"entropy residual: {entropy_residual} J/(kg K); each absolute value must be below"
                               f" 1e-3 J/(kg K).")


        # Update analysis_results_at_design_point and blade row results
        self.analysis_results_at_design_point = analysis_results
        self.blade_row_results_at_design_point = blade_row_results
        self.loss_model_results_at_design_point = loss_model_results

        # Calculate and assign remaining properties related to geometry
        self.D_hub, self.D_tip, self.D_mean, self.l_rotor, self.h_shroud, self.s_ax_shroud =\
            self.__calculate_geometry(self.analysis_results_at_design_point)  # m
        self.l_stator = self.l_rotor  # m
        self.R_mean = self.D_mean / 2  # m
        self.R_hub = self.D_hub / 2  # m
        self.R_tip = self.D_tip / 2  # m

    def calculate_entropy_rise(self, delta_s_stator, delta_s_rotor, delta_s_rotor_additional,
                               loss_model: TraupelLossModel):
        """A method to calculate residuals between the assumed entropy rises and the entropy rises returned by the loss
        model, together with the turbine analysis results. This method has one internal numerical loop to calculate
         consistent thermodynamic state at each station for assumed entropy increases through blade rows with
         __calculate_thermodynamic_properties. __calculate_blade_row_flow_state then calculates the blade row flow
         state at the stage outlet static pressure, without inclusion of additional losses like clearance, partial
         admission or disk friction.

        Definitions of the values read from analysis_results_at_design_point are in Turbine1D.__init__.

        :param float delta_s_stator: Specific entropy rise due to aerodynamic losses in the stator (J/kg/K).
        :param float delta_s_rotor: Specific entropy rise due to aerodynamic losses in the rotor (J/kg/K).
        :param float delta_s_rotor_additional: Specific entropy rise due to additional rotor losses such as clearance,
            partial admission or disk friction losses (J/(kg K)).
        :param TraupelLossModel loss_model: Traupel loss model used to calculate the entropy rises.
        :return: Entropy-rise residuals, turbine analysis results, blade-row results and Traupel loss results,
            respectively. Dictionary key definitions are in Turbine1D.__init__.
        :rtype: tuple[list, dict, dict, dict]
        :raises TypeError: If loss_model is not a TraupelLossModel instance.
        :raises ValueError: If shaft work and outlet axial kinetic energy leave no positive outlet static enthalpy,
            an h_2/h_t0 trial is not finite and positive, or the prescribed blade-row state is infeasible.
        :raises RuntimeError: If the internal solve does not converge, returns a nonpositive or nonfinite solution,
            or its absolute dimensional total-enthalpy residual is not below 1e-3 J/kg.
        """

        if not isinstance(loss_model, TraupelLossModel):
            raise TypeError("loss_model must be a TraupelLossModel instance.")

        # In rocket turbines, there are no clearance losses in stators, while partial admission losses belong to the
        # rotor. Therefore, entropy generation in the stator due to aerodynamic losses constitute total entropy
        # generation through the stator.
        delta_s_stator_total = delta_s_stator  # J/(kg K)
        # In the rotor, aside from aerodynamic losses, there are additional losses like clearance, partial admission
        # or disk friction losses. Total entropy generation is thus the sum of all of these.
        delta_s_rotor_total = delta_s_rotor + delta_s_rotor_additional  # J/(kg K)

        # Thermodynamic properties at each station must be calculated. Thermodynamic properties depend on h_2/h_t0.
        # It needs to be numerically found, such that energy balance is satisfied and total enthalpy at station 2
        # agrees with the already calculated one. The inner solution loop first attempts to use relaxed Aitken
        # iterations. If this does not work, it attempts to use TOMS748 bracketing scheme. As the last fallback,
        # Newton scheme is attempted.

        # First, retrieve some variables from analysis_results_at_design_point
        # that will be used to obtain solution estimate, as well as feasible bracket for numerical scheme. Bracket
        # allows to use reliable toms748 scheme.
        gas = self.analysis_results_at_design_point["gas"]
        h_t0 = self.analysis_results_at_design_point["h_t0"]  # J/kg
        h_t2 = self.analysis_results_at_design_point["h_t2"]  # J/kg
        u = self.analysis_results_at_design_point["u"]  # m/s
        theta_2 = self.analysis_results_at_design_point["theta_2"]  # -
        psi_tt = self.analysis_results_at_design_point["psi_tt"]
        R_h_tt_ideal = self.analysis_results_at_design_point["R_h_tt_ideal"]
        delta_h_t = self.analysis_results_at_design_point["delta_h_t"]
        # The known axial velocity sets the largest possible exit static enthalpy (zero exit swirl).
        h_2_max = h_t2 - (theta_2 * u)**2 / 2  # J/kg
        if h_2_max <= 0:
            raise ValueError("Shaft work and exit axial kinetic energy leave no positive exit static enthalpy.")
        # Divide by the inlet total enthalpy to obtain the upper bound on h_2/h_t0.
        h_2_max_over_h_t0 = h_2_max / h_t0  # -

        # Calculate the lower bound on h_2/h_t0 for the bracket as well. These calculations are more complex.
        # They assume nonnegative row entropy rises and ideal reaction, and positive shaft work/loading.
        # Newton scheme is first retained as the default fallback.
        numerical_scheme = "newton"
        # If assumptions are satisfied, calculate upper bracket and set bracketing scheme as fallback
        if (delta_s_stator_total >= 0 and delta_s_rotor_total >= 0 and R_h_tt_ideal >= 0
                and psi_tt > 0 and delta_h_t > 0):
            # Rotor entropy changes the isentropic enthalpy ratio by factor below. Rearranging the reaction relation
            # gives h_1 = rotor_entropy_factor * h_2 + h_1_at_h_2_zero, so h_1 is linear in exit static enthalpy h_2.
            rotor_entropy_factor = np.exp(-delta_s_rotor_total / gas.Cp)
            # Calculate the limiting rotor-inlet enthalpy as h_2 tends to zero, taking into account both rows' entropy
            # rises. This is the intercept of the linear relation h_1 = rotor_entropy_factor * h_2 + h_1_at_h_2_zero.
            h_1_at_h_2_zero = R_h_tt_ideal * (h_t0 * np.exp(delta_s_stator_total / gas.Cp) - rotor_entropy_factor * h_t2)
            # From R_h_tt = (h_1 - h_2) / delta_h_t, obtain the limiting reaction at the low-enthalpy end
            # (h_1_at_h_2_zero) of the possible solution interval.
            R_h_tt_at_h_2_zero = h_1_at_h_2_zero / delta_h_t
            # Obtain reaction at the other end with the largest possible exit enthalpy, set by zero exit swirl.
            R_h_tt_at_h_2_max = ((rotor_entropy_factor - 1) * h_2_max + h_1_at_h_2_zero) / delta_h_t
            # Since reaction is linear in h_2, its largest absolute value on [0, h_2_max] occurs at an endpoint.
            R_h_tt_abs_max = max(abs(R_h_tt_at_h_2_zero), abs(R_h_tt_at_h_2_max))
            # Continuity gives theta_1 = theta_2 * rotor_entropy_factor * (rotor_entropy_factor * h_2 / h_1)**(Cp/R - 1)
            # The entropy factor is <= 1 and h_1 >= rotor_entropy_factor * h_2, so theta_1**2 <= theta_2**2.
            # Applying the absolute operator to each right term of v_2_tangential/u = 1 - R_h_tt - psi_tt/2
            # + (theta_2**2 - theta_1**2)/(2*psi_tt) and using maximum possible values for these terms therefore bounds
            # the magnitude of exit swirl v_2_tangential from above
            v_2_tangential_over_u_bound = abs(1 - psi_tt / 2) + R_h_tt_abs_max + theta_2**2 / (2 * psi_tt)
            # Subtracting largest possible value of the swirl from estimate of h_2 without any swirl,
            # gives a conservative lower bound on h_2 at any energy-balanced solution.
            h_2_min = h_2_max - (u * v_2_tangential_over_u_bound)**2 / 2
            # Use a positive lower endpoint for the h_2/h_t0 bracket.
            h_2_min_over_h_t0 = max(h_2_min / h_t0, np.finfo(float).eps * h_2_max_over_h_t0)  # -
            bracket = [h_2_min_over_h_t0, h_2_max_over_h_t0]
            numerical_scheme = "bracketing"

        # Define the total-enthalpy residual. It is already normalized by h_t0.
        def get_enthalpy_residual(h_2_over_h_t0_initial):
            return self.__calculate_thermodynamic_properties(h_2_over_h_t0_initial, delta_s_stator_total,
                                                            delta_s_rotor_total, delta_s_rotor_additional)[1]

        # Subtract the normalized energy residual to obtain the next exit static-enthalpy ratio.
        def get_new_enthalpy_ratio(h_2_over_h_t0_initial):
            return h_2_over_h_t0_initial - get_enthalpy_residual(h_2_over_h_t0_initial)

        # Attempt to solve for enthalpy ratio now, first with Aitken scheme.
        # Start at the same upper estimate as Newton and use the solution bracket already estimated for toms748 scheme.
        # Initial relaxation 1 and cap 1.2 gave the lowest evaluation count across multiple designs.
        try:
            h_2_over_h_t0 = solve_with_relaxed_aitken(
                get_new_enthalpy_ratio, h_2_max_over_h_t0,
                initial_relaxation=1., min_relaxation=0.05, max_relaxation=1.2,
                bounds=bracket if numerical_scheme == "bracketing" else
                (np.finfo(float).eps * h_2_max_over_h_t0, h_2_max_over_h_t0),
                xtol=1e-11, rtol=1e-11, maxiter=1000)
        # If Aitken does not work, try fallback schemes
        except (ValueError, RuntimeError, ArithmeticError):
            # Verify the candidate bracket. Finite, ordered endpoints and opposite residual
            # signs are required; a zero endpoint residual is already a root and is accepted by toms748.
            if numerical_scheme == "bracketing":
                # Check if values of the bracket are ordered and finite
                if not np.all(np.isfinite(bracket)) or bracket[0] >= bracket[1]:
                    numerical_scheme = "newton"
                # If so, evaluate residual at endpoints
                else:
                    try:
                        residual_at_bracket = [get_enthalpy_residual(h_2_over_h_t0) for h_2_over_h_t0 in bracket]
                    # If endpoints fail, choose Newton scheme as fallback instead
                    except ValueError:
                        numerical_scheme = "newton"
                    # Also choose Newton scheme if residuals are not zero and don't have opposite signs
                    else:
                        if not np.all(np.isfinite(residual_at_bracket)) or (
                                residual_at_bracket[0] != 0 and residual_at_bracket[1] != 0
                                and np.signbit(residual_at_bracket[0]) == np.signbit(residual_at_bracket[1])):
                            numerical_scheme = "newton"

            # Fallback to toms748 scheme if it all checks for it passed.
            if numerical_scheme == "bracketing":
                try:
                    h_2_over_h_t0_solution = root_scalar(get_enthalpy_residual, bracket=bracket, method="toms748",
                                                         options={"k": 2}, maxiter=1000, xtol=1e-11, rtol=1e-11)
                # An inadmissible state encountered inside the bracket requires the Newton fallback.
                except ValueError:
                    numerical_scheme = "newton"
                # If the solution did not converge, use Newton scheme fallback as well.
                else:
                    if not h_2_over_h_t0_solution.converged:
                        numerical_scheme = "newton"
            # Use Newton when no valid bracket is available or when the bracketing solve failed to converge.
            if numerical_scheme == "newton":
                h_2_over_h_t0_solution = root_scalar(get_enthalpy_residual, x0=h_2_max_over_h_t0, method="newton",
                                                     maxiter=1000, xtol=1e-11, rtol=1e-11)

            # Raise error if the solution obtained from toms748 or Newton is not converged
            if not h_2_over_h_t0_solution.converged:
                raise RuntimeError("Numerical solve for h_2/h_t0 did not converge.")
            # Finally obtain the root
            h_2_over_h_t0 = h_2_over_h_t0_solution.root  # -

        # The solved enthalpy ratio must be finite and positive before calculating the final thermodynamic state.
        if not np.isfinite(h_2_over_h_t0) or h_2_over_h_t0 <= 0:
            raise RuntimeError("Numerical solve for h_2/h_t0 returned a nonfinite or nonpositive solution.")
        analysis_results, residual = self.__calculate_thermodynamic_properties(
            h_2_over_h_t0, delta_s_stator_total, delta_s_rotor_total, delta_s_rotor_additional)  # residual: -
        # Restore the dimensional energy residual and require an absolute error below 1e-3 J/kg.
        enthalpy_residual = residual * h_t0  # J/kg
        if not np.isfinite(enthalpy_residual) or abs(enthalpy_residual) >= 1e-3:
            raise RuntimeError("Numerical solve for h_2/h_t0 has an invalid or excessive total-enthalpy residual: "
                               f"{enthalpy_residual} J/kg; its absolute value must be below 1e-3 J/kg.")

        # Complete the outlet flow properties only after h_2/h_t0 satisfies the energy balance.
        analysis_results = self.__calculate_blade_rows_outlet_velocities(analysis_results)

        # Flow angles can be now calculated. These are affected by the total entropy
        # generation in the rotor, as it affects theta_1 through rho_2.
        alpha_1, beta_1, alpha_2, beta_2 = self.__calculate_flow_angles(analysis_results)  # rad
        # Append analysis results with these flow angles
        analysis_results.update({"alpha_1": alpha_1,  # rad
                                 "alpha_2": alpha_2,  # rad
                                 "beta_1": beta_1,  # rad
                                 "beta_2": beta_2,  # rad
                                 "alpha_1_deg": alpha_1 * 180 / np.pi,  # deg
                                 "alpha_2_deg": alpha_2 * 180 / np.pi,  # deg
                                 "beta_1_deg": beta_1 * 180 / np.pi,  # deg
                                 "beta_2_deg": beta_2 * 180 / np.pi,  # deg
                                 "incidence": 0,  # rad
                                 "incidence_deg": 0,  # deg
                                 })


        # Recalculate rotor flow states at the same outlet static pressure, but without additional losses like disk
        # friction, clearance or partial admission, as the loss model may require a separate blade row flow state.
        # These results only take into account aerodynamic losses in the blade passage itself. The outlet flow
        # coefficient and flow angles follow directly from entropy, rothalpy and continuity.
        blade_row_results = self.__calculate_blade_row_flow_state(analysis_results)

        # Some additional geometry must be calculated for the loss model. Pack to dictionary to pass to the loss model.
        D_hub, D_tip, D_mean, l_rotor, h_shroud, s_ax_shroud = self.__calculate_geometry(analysis_results)  # m
        l_stator = l_rotor  # m
        turbine_geometry = {"D_hub": D_hub,  # m
                            "D_tip": D_tip,  # m
                            "D_mean": D_mean,  # m
                            "D_Euler": self.D_Euler,  # m
                            "R_hub": D_hub / 2,  # m
                            "R_tip": D_tip / 2,  # m
                            "R_mean": D_mean / 2,  # m
                            "R_Euler": self.R_Euler,  # m
                            "l_stator": l_stator,  # m
                            "l_rotor": l_rotor,  # m
                            "s_ax": self.s_ax,  # m
                            "s_r": self.s_r,  # m
                            "shrouded_rotor": self.shrouded_rotor,  # -
                            "s_ax_shroud": s_ax_shroud,  # m
                            "seal_teeth_number": self.seal_teeth_number, # -
                            "seal_teeth_spacing": self.seal_teeth_spacing, # m
                            "t_shroud": self.t_shroud,  # m
                            "h_shroud": h_shroud,  # m
                            "h_shroud_over_blade_length": self.h_shroud_over_blade_length,  # -
                            "s_ax_shroud_over_h_shroud": self.s_ax_shroud_over_h_shroud,  # -
                            "c_stator": self.c_stator,  # m
                            "p_stator": self.p_stator,  # m
                            "c_rotor": self.c_rotor,  # m
                            "p_rotor": self.p_rotor,  # m
                            "t_TE_stator": self.t_TE_stator,  # m
                            "t_LE_rotor": self.t_LE_rotor,  # m
                            "t_TE_rotor": self.t_TE_rotor,  # m
                            "no_blades_rotor": self.no_blades_rotor,  # -
                            "no_blades_stator": self.no_blades_stator,  # -
                            "admission_fraction": self.admission_fraction, # -
                            "partial_admission_rotor": self.partial_admission_rotor,  # string
                            "sand_grain_roughness": self.sand_grain_roughness,  # m
                            }

        # Call loss model, calculate entropy rise and loss coefficients.
        delta_s_stator_output, delta_s_rotor_output, delta_s_rotor_additional_output, loss_model_results =\
            loss_model.calculate_entropy_increase(
                analysis_results, turbine_geometry, blade_row_results)  # entropy rises: J/(kg K)


        # Calculate residual
        residual = [delta_s_stator_output - delta_s_stator, delta_s_rotor_output - delta_s_rotor,
                    delta_s_rotor_additional_output - delta_s_rotor_additional]


        # Return all calculated results
        return residual, analysis_results, blade_row_results, loss_model_results

    def __calculate_geometry(self, analysis_results):
        """A method to calculate some of the turbine geometry.

        :param dict analysis_results: Turbine analysis results. Variable definitions are in Turbine1D.__init__.
        """

        # First obtain design blade velocity and theta_2.
        u = analysis_results["u"]  # m/s
        theta_2 = analysis_results["theta_2"]  # -

        # Calculate axial velocity.
        v_ax = theta_2 * u  # m/s

        # From massflow, density and axial velocity, calculate annulus area.
        mdot = analysis_results["mdot_total"]  # kg/s
        rho_2 = analysis_results["rho_2"]  # kg/m^3
        area = mdot / (v_ax * rho_2 * self.admission_fraction)  # m^2

        # Reject invalid flow areas or diameters before evaluating the annulus square roots.
        if not np.isfinite(area) or area <= 0 or not np.isfinite(self.D_Euler) or self.D_Euler <= 0:
            raise ValueError("Annulus area and Euler diameter must be finite and positive.")
        # A larger annulus would require a negative hub-diameter square at the specified Euler diameter.
        if self.D_Euler**2 - (2 * area / np.pi) < 0:
            raise ValueError("Required annulus area exceeds the maximum allowed by the Euler diameter.")

        # From annulus area, calculate blade length.
        D_hub = np.sqrt(self.D_Euler**2 - (2 * area / np.pi))  # m
        D_tip = np.sqrt(self.D_Euler**2 + (2 * area / np.pi))  # m
        D_mean = (D_hub + D_tip) / 2  # m
        l_blade = (D_tip - D_hub) / 2  # m

        # If shroud is used, calculate remaining variables. Otherwise, these are None.
        if self.shrouded_rotor:
            h_shroud = self.h_shroud_over_blade_length * l_blade  # m
            s_ax_shroud = self.s_ax_shroud_over_h_shroud * h_shroud  # m
        else:
            h_shroud = None
            s_ax_shroud = None

        # Return the results
        return D_hub, D_tip, D_mean, l_blade, h_shroud, s_ax_shroud

    def __calculate_thermodynamic_properties(self, h_2_over_h_t0_initial, total_delta_s_stator, total_delta_s_rotor,
                                             delta_s_rotor_additional):
        """Calculate thermodynamic states at turbine stations 1 and 2 and the outlet total-enthalpy residual for
        the assumed h_2/h_t0.

        Definitions of the values read from analysis_results_at_design_point are in Turbine1D.__init__.

        :param float h_2_over_h_t0_initial: Assumed static enthalpy at station 2 divided by inlet total enthalpy,
         h_2/h_t0 (-). Must be positive.
        :param float total_delta_s_stator: Total specific entropy rise across the stator (J/kg/K).
        :param float total_delta_s_rotor: Total specific entropy rise across the rotor (J/kg/K).
        :param float delta_s_rotor_additional: Specific entropy rise due to clearance, partial admission and disk
            friction losses in the rotor (J/kg/K).
        :return: Dictionary with turbine analysis results and residual of the total enthalpy at station 2
         normalized by h_t0 (-), respectively.
        :rtype: tuple[dict, float]
        :raises ValueError: If h_2/h_t0 is not finite and positive.
        """

        if not np.isfinite(h_2_over_h_t0_initial) or h_2_over_h_t0_initial <= 0:
            raise ValueError("h_2/h_t0 must be finite and positive.")

        # Retrieve already known variables from analysis_results_at_design_point
        psi_tt = self.analysis_results_at_design_point["psi_tt"]  # -
        R_h_tt_ideal = self.analysis_results_at_design_point["R_h_tt_ideal"]  # -
        theta_2 = self.analysis_results_at_design_point["theta_2"]  # -
        h_t2 = self.analysis_results_at_design_point["h_t2"]  # J/kg
        h_t0 = self.analysis_results_at_design_point["h_t0"]  # J/kg
        T_0 = self.analysis_results_at_design_point["T_0"]  # K
        p_0 = self.analysis_results_at_design_point["p_0"]  # Pa
        gas = self.analysis_results_at_design_point["gas"] # IdealGas object
        u = self.analysis_results_at_design_point["u"]  # m/s
        delta_h_t = self.analysis_results_at_design_point["delta_h_t"]  # J/kg

        # Calculate total entropy increase
        delta_s = total_delta_s_rotor + total_delta_s_stator  # J/(kg K)
        # Calculate h_0
        h_0 = gas.Cp * T_0  # J/kg
        # Calculate static enthalpy, temperature, pressure and density at station 2 from h_2/h_t0.
        h_2 = h_2_over_h_t0_initial * h_t0  # J/kg
        T_2 = h_2 / gas.Cp  # K
        p_2 = p_0 * np.exp(-delta_s / gas.R) * h_2_over_h_t0_initial**(gas.Cp / gas.R)  # Pa
        p_0_over_p_2 = p_0 / p_2  # -
        rho_2 = gas.calculate_density(p_2, T_2)  # kg/m^3

        # Calculate isentropic reference state at station 2 assuming that h_0 = h_t0
        h_2_ideal = h_2 * np.exp(-delta_s / gas.Cp)
        h_t2_ideal = h_t2 * np.exp(-delta_s / gas.Cp)
        # Calculate difference between total ideal enthalpies
        delta_h_t_ideal = h_0 - h_t2_ideal
        # Now using ideal reaction definition, find ratio of isentropic enthalpies at station 1 and 2
        h_1_over_h_2_isentropic = 1 + R_h_tt_ideal * delta_h_t_ideal / h_2_ideal
        # Now the ratio of real enthalpies can be found
        h_1_over_h_2 = h_1_over_h_2_isentropic * np.exp(-total_delta_s_rotor / gas.Cp) # -

        # Calculate p_1, T_1, rho_1, h_1, h_t1
        h_1 = h_2 * h_1_over_h_2  # J/kg
        T_1 = h_1 / gas.Cp  # K
        p_1 = p_2 * np.exp(total_delta_s_rotor / gas.R) * h_1_over_h_2**(gas.Cp / gas.R)  # Pa
        rho_1 = gas.calculate_density(p_1, T_1)  # kg/m^3
        # Energy is conserved between station 0 and station 1, so:
        h_t1 = h_0  # J/kg

        # Calculate total pressures from isentropic deceleration at each station.
        p_t1 = p_1 * (h_t1 / h_1)**(gas.Cp / gas.R)  # Pa
        p_t2 = p_2 * (h_t2 / h_2)**(gas.Cp / gas.R)  # Pa

        # Calculate theta_1
        theta_1 = theta_2 * rho_2 / rho_1  # -
        # Calculate real reaction from its definition
        R_h_tt = (h_1 - h_2) / delta_h_t # -
        # Calculate absolute velocity at the outlet
        v_2_over_u = np.sqrt(theta_2**2 + (1 - R_h_tt - psi_tt / 2 + (theta_2**2 - theta_1**2) / (2 * psi_tt))**2)  # -
        v_2 = v_2_over_u * u  # m/s
        # Calculate total enthalpy at the outlet
        h_t2_calculated = h_2 + v_2**2 / 2 # J/kg
        # Finally, the total enthalpy residual can be calculated. It is normalized by h_t0.
        residual = (h_t2_calculated - h_t2) / h_t0 # -

        # Calculate remaining values of interest.
        # First calculate ideal psi_tt
        h_t2_over_h_t0 = h_t2_calculated / h_0  # -
        psi_tt_ideal = psi_tt * (1 - h_t2_over_h_t0 * np.exp(-delta_s / gas.Cp)) / (1 - h_t2_over_h_t0)  # -
        # Calculate real pressure reaction, R_p
        R_p = (p_1 - p_2) / (p_0 - p_2)  # -
        # Calculate static-to-static work coefficient
        delta_h_ideal = h_0 - h_2_ideal # J/kg
        psi_ss_ideal = delta_h_ideal / u ** 2 # -

        # Calculate Denton loss coefficients
        w_2_squared = u**2 * (theta_2**2 + (-R_h_tt - psi_tt / 2
                                          + (theta_2**2 - theta_1**2) / (2 * psi_tt))**2)
        zeta_stator_Denton = T_2 * total_delta_s_stator / (h_0 - h_1)  # -
        zeta_rotor_Denton = 2 * T_2 * total_delta_s_rotor / w_2_squared  # -
        # Calculate total-to-total and total-to-static efficiencies
        eta_tt = delta_h_t / delta_h_t_ideal  # -
        eta_ts = delta_h_t / delta_h_ideal  # -
        # Calculate total-to-total polytropic efficiency for a calorically perfect gas.
        log_h_t2_over_h_t0 = np.log(h_t2_over_h_t0)  # -
        eta_poly_tt = log_h_t2_over_h_t0 / (log_h_t2_over_h_t0 - delta_s / gas.Cp)  # -
        # Calculate total-to-static polytropic efficiency.
        log_h_2_over_h_t0 = np.log(h_2 / h_t0)  # -
        delta_h_poly_ts = (h_0 - h_2) * (1 - delta_s / (gas.Cp * log_h_2_over_h_t0))  # J/kg
        eta_poly_ts = delta_h_t / delta_h_poly_ts  # -

        # Package thermodynamic properties, v_2 and loading coefficients into analysis_results dictionary.
        # Make it a copy of analysis_results_at_design_point, such that the values there stay constant during
        # iterations.
        analysis_results = self.analysis_results_at_design_point.copy()
        analysis_results.update({"p_0_over_p_2": p_0_over_p_2, # -
                                 "h_0": h_0,  # J/kg
                                 "p_1": p_1,  # Pa
                                 "p_t1": p_t1,  # Pa
                                 "T_1": T_1,  # K
                                 "rho_1": rho_1,  # kg/m^3
                                 "h_1": h_1,  # J/kg
                                 "h_t1": h_t1,  # J/kg
                                 "delta_s_stator": total_delta_s_stator,  # J/(kg K)
                                 "p_2": p_2,  # Pa
                                 "p_t2": p_t2,  # Pa
                                 "T_2": T_2,  # K
                                 "rho_2": rho_2,  # kg/m^3
                                 "h_2": h_2,  # J/kg
                                 "h_t2": h_t2,  # J/kg
                                 "delta_s_rotor": total_delta_s_rotor - delta_s_rotor_additional,  # J/(kg K)
                                 "delta_s_rotor_additional": delta_s_rotor_additional,  # J/(kg K)
                                 "v_2": v_2,  # m/s
                                 "R_p": R_p,  # -
                                 "psi_tt_ideal": psi_tt_ideal,  # -
                                 "psi_ss_ideal": psi_ss_ideal,  # -
                                 "psi_ss_ideal_design": psi_ss_ideal,  # -
                                 "R_h_tt": R_h_tt,  # -
                                 "theta_1": theta_1,  # -
                                 "delta_h_ideal": delta_h_ideal, # J/kg
                                 "zeta_stator_Denton": zeta_stator_Denton,  # -
                                 "zeta_rotor_Denton": zeta_rotor_Denton,  # -
                                 "eta_tt": eta_tt,  # -
                                 "eta_ts": eta_ts,  # -
                                 "eta_poly_tt": eta_poly_tt,  # -
                                 "eta_poly_ts": eta_poly_ts,  # -
                                 })

        # Return residual and analysis results
        return analysis_results, residual

    def __calculate_blade_rows_outlet_velocities(self, analysis_results):
        """Calculate outlet velocities and related flow properties for the rotor and stator blade rows and update the
         supplied analysis_results dictionary.

        This method adds real and isentropic reference velocities, sound speeds, Mach numbers and Reynolds numbers at
         stations 1 and 2 to that dictionary. The stator isentropic reference starts at station 0; the rotor isentropic
          reference starts at the real station 1 state. The corresponding reference enthalpies and temperatures are also
           added to the dictionary. Reconstruction of the flow state of the blade row alone without additional rotor
            losses is performed separately by __calculate_blade_row_flow_state.

        :param dict analysis_results: Dictionary returned by __calculate_thermodynamic_properties for the solved
            h_2/h_t0. Required values include gas, thermodynamic states, v_2, u, psi_tt, R_h_tt and flow coefficients.
            Variable definitions are in Turbine1D.__init__.
        :return: The supplied analysis_results dictionary with added outlet flow properties and isentropic reference
         states.
        :rtype: dict
        """

        # Read the flow states and stage coefficients.
        gas = analysis_results["gas"]  # IdealGas object
        h_0 = analysis_results["h_0"]  # J/kg
        p_0 = analysis_results["p_0"]  # Pa
        p_1 = analysis_results["p_1"]  # Pa
        T_1 = analysis_results["T_1"]  # K
        rho_1 = analysis_results["rho_1"]  # kg/m^3
        h_1 = analysis_results["h_1"]  # J/kg
        p_2 = analysis_results["p_2"]  # Pa
        T_2 = analysis_results["T_2"]  # K
        rho_2 = analysis_results["rho_2"]  # kg/m^3
        h_2 = analysis_results["h_2"]  # J/kg
        v_2 = analysis_results["v_2"]  # m/s
        u = analysis_results["u"]  # m/s
        psi_tt = analysis_results["psi_tt"]  # -
        R_h_tt = analysis_results["R_h_tt"]  # -
        theta_1 = analysis_results["theta_1"]  # -
        theta_2 = analysis_results["theta_2"]  # -

        # Now calculate velocity and Mach number at station 1. First calculate velocity of sound there:
        a_1 = gas.calculate_sound_velocity(T_1)  # m/s
        h_1_ideal = h_0 * (p_1 / p_0) ** (gas.R / gas.Cp)  # J/kg
        T_1_ideal = h_1_ideal / gas.Cp  # K
        a_1_ideal = gas.calculate_sound_velocity(T_1_ideal)  # m/s
        # Real and ideal velocity and Mach number in stationary reference frame:
        v_1 = np.sqrt(2 * (h_0 - h_1))  # m/s
        delta_h_loss_stator = h_1 - h_1_ideal  # J/kg
        v_1_ideal = np.sqrt(v_1**2 + 2 * delta_h_loss_stator)  # m/s
        M_s1 = v_1 / a_1  # -
        M_s1_ideal = v_1_ideal / a_1_ideal  # -
        # Now real values in rotary reference frame:
        dummy_a1 = 1 - R_h_tt + psi_tt / 2 + (theta_2**2 - theta_1**2) / (2 * psi_tt)  # -
        dummy_b1 = dummy_a1 - 1  # -
        w_1 = u * np.sqrt(theta_1**2 + dummy_b1**2)  # m/s
        M_r1 = w_1 / a_1  # -
        # Calculate isentropic enthalpy drop through the stator
        delta_h_stator_ideal = h_0 - h_1_ideal

        # Now velocity and Mach number at station 2. Ideal values at station 2
        # still assume real values at station 1. First calculate velocity of sound at station 2.
        a_2 = gas.calculate_sound_velocity(T_2)  # m/s
        h_2_ideal_r_real_s = h_1 * (p_2 / p_1) ** (gas.R / gas.Cp)  # J/kg
        T_2_ideal_r_real_s = h_2_ideal_r_real_s / gas.Cp  # K
        a_2_ideal_r_real_s = gas.calculate_sound_velocity(T_2_ideal_r_real_s)  # m/s
        # Real values in stationary reference frame:
        M_s2 = v_2 / a_2  # -
        # Now real and ideal values in rotary reference frame:
        dummy_a2 = dummy_a1 - psi_tt  # -
        dummy_b2 = dummy_a2 - 1  # -
        w_2 = u * np.sqrt(theta_2**2 + dummy_b2**2)  # m/s
        delta_h_loss_rotor = h_2 - h_2_ideal_r_real_s  # J/kg
        w_2_ideal_r_real_s = np.sqrt(w_2**2 + 2 * delta_h_loss_rotor)  # m/s
        M_r2 = w_2 / a_2  # -
        M_r2_ideal_r_real_s = w_2_ideal_r_real_s / a_2_ideal_r_real_s  # -
        # Calculate isentropic enthalpy drop through the rotor, assuming real expansion at the stator
        delta_h_rotor_ideal = h_1 - h_2_ideal_r_real_s

        # Calculate axial Mach numbers from axial velocity, theta*u, and sound velocity at each station.
        M_ax_1 = theta_1 * u / a_1  # -
        M_ax_2 = theta_2 * u / a_2  # -

        # Calculate Reynolds numbers only for the solved outlet states.
        Re_s1 = rho_1 * v_1 * self.c_stator / gas.calculate_dynamic_viscosity(T_1)  # -
        Re_r2 = rho_2 * w_2 * self.c_rotor / gas.calculate_dynamic_viscosity(T_2)  # -

        # Extend the dictionary with results at the design point.
        analysis_results.update({"T_1_ideal": T_1_ideal,  # K
                                 "h_1_ideal": h_1_ideal,  # J/kg
                                 "a_1": a_1,  # m/s
                                 "a_1_ideal": a_1_ideal,  # m/s
                                 "v_1": v_1,  # m/s
                                 "v_1_ideal": v_1_ideal,  # m/s
                                 "w_1": w_1,  # m/s
                                 "M_s1": M_s1,  # -
                                 "M_s1_ideal": M_s1_ideal,  # -
                                 "M_r1": M_r1,  # -
                                 "M_ax_1": M_ax_1,  # -
                                 "T_2_ideal_r_real_s": T_2_ideal_r_real_s,  # K
                                 "h_2_ideal_r_real_s": h_2_ideal_r_real_s,  # J/kg
                                 "a_2": a_2,  # m/s
                                 "a_2_ideal_r_real_s": a_2_ideal_r_real_s,  # m/s
                                 "w_2": w_2,  # m/s
                                 "w_2_ideal_r_real_s": w_2_ideal_r_real_s,  # m/s
                                 "M_s2": M_s2,  # -
                                 "M_r2": M_r2,  # -
                                 "M_ax_2": M_ax_2,  # -
                                 "M_r2_ideal_r_real_s": M_r2_ideal_r_real_s,  # -
                                 "Re_s1": Re_s1,  # -
                                 "Re_r2": Re_r2,  # -
                                 "delta_h_stator_ideal": delta_h_stator_ideal, # J/kg
                                 "delta_h_rotor_ideal": delta_h_rotor_ideal, # J/kg
                                 })
        return analysis_results

    @staticmethod
    def __calculate_flow_angles(analysis_results):
        """A method to calculate absolute and relative flow angles at stations 1 and 2.

        :param dict analysis_results: Dictionary with turbine analysis results required to calculate the flow angles.
            Variable definitions are in Turbine1D.__init__.
        :return: Absolute flow angle at station 1 (rad), relative flow angle at station 1 (rad),
         absolute flow angle at station 2 (rad) and relative flow angle at station 2 (rad), respectively.
        :rtype: tuple[float, float, float, float]
        """

        # Retrieve variables from analysis_results
        theta_1 = analysis_results["theta_1"]  # -
        theta_2 = analysis_results["theta_2"]  # -
        R_h_tt = analysis_results["R_h_tt"]  # -
        psi_tt = analysis_results["psi_tt"]  # -
        # Calculate support variable
        dummy_theta = (theta_2**2 - theta_1**2) / (2 * psi_tt)  # -
        # Calculate flow angles
        alpha_1 = np.atan2(1 - R_h_tt + psi_tt / 2 + dummy_theta, theta_1)  # rad
        beta_1 = np.atan2(-R_h_tt + psi_tt / 2 + dummy_theta, theta_1)  # rad
        beta_2 = np.atan2(-R_h_tt - psi_tt / 2 + dummy_theta, theta_2)  # rad
        alpha_2 = np.atan2(1 - R_h_tt - psi_tt / 2 + dummy_theta, theta_2)  # rad
        # Return all flow angles
        return alpha_1, beta_1, alpha_2, beta_2

    def __calculate_blade_row_flow_state(self, analysis_results):
        """Calculate the flow state for the blade row alone without additional rotor losses such as clearance,
         partial admission or disk friction losses, at the same outlet static pressure as the whole stage.
          Aerodynamic entropy rise determines outlet temperature; constant-radius rothalpy and equal-area continuity
           determine outlet velocities, flow coefficient and flow angles.

        :param dict analysis_results: Dictionary with turbine analysis results required to calculate the blade row
            flow state. Variable definitions are in Turbine1D.__init__.
        :return: Dictionary with blade row flow analysis results.
        :rtype: dict
        :raises ValueError: If outlet temperature, relative speed or flow coefficient is not finite and positive,
            or continuity requires an axial velocity greater than the available relative velocity.
        """

        # The rotor inlet flow state at station 1 is taken from the stage calculation. Only the outlet state is
        # reconstructed without additional rotor losses.
        gas = analysis_results["gas"]
        p_1_blade = analysis_results["p_1"]  # Pa
        T_1_blade = analysis_results["T_1"]  # K
        rho_1_blade = analysis_results["rho_1"]  # kg/m^3
        h_1_blade = analysis_results["h_1"]  # J/kg
        w_1_blade = analysis_results["w_1"]  # m/s
        v_1_blade = analysis_results["v_1"]  # m/s
        M_s1_blade = analysis_results["M_s1"]  # -
        theta_1_blade = analysis_results["theta_1"]  # -
        alpha_1 = analysis_results["alpha_1"]  # rad
        h_0 = analysis_results["h_0"]  # J/kg
        p_0 = analysis_results["p_0"]  # Pa
        # Unpack entropy increase across the rotor due to aerodynamic losses
        delta_s_rotor = analysis_results["delta_s_rotor"]  # J/(kg K)
        # Blade velocity is also the same
        u = analysis_results["u"]  # m/s

        # Prescribe the stage outlet static pressure and use only aerodynamic entropy rise to calculate temperature.
        p_2_blade = analysis_results["p_2"]  # Pa
        T_2_blade = T_1_blade * np.exp((gas.R * np.log(p_2_blade / p_1_blade) + delta_s_rotor) / gas.Cp)  # K
        if not np.isfinite(T_2_blade) or T_2_blade <= 0:
            raise ValueError("Blade-row outlet temperature must be finite and positive.")
        h_2_blade = gas.Cp * T_2_blade  # J/kg
        rho_2_blade = gas.calculate_density(p_2_blade, T_2_blade)  # kg/m^3

        # From conservation of rothalpy at constant Euler radius, calculate the relative outlet velocity.
        w_2_blade_squared = w_1_blade**2 + 2 * (h_1_blade - h_2_blade)  # m^2/s^2
        if not np.isfinite(w_2_blade_squared) or w_2_blade_squared <= 0:
            raise ValueError("Blade-row pressure and entropy rise leave no finite, positive relative outlet speed.")
        w_2_blade = np.sqrt(w_2_blade_squared)  # m/s

        # Equal-area continuity determines the outlet flow coefficient and axial velocity.
        theta_2_blade = rho_1_blade * theta_1_blade / rho_2_blade  # -
        if not np.isfinite(theta_2_blade) or theta_2_blade <= 0:
            raise ValueError("Blade-row outlet flow coefficient must be finite and positive.")
        w_2_blade_axial = u * theta_2_blade  # m/s
        w_2_tangential_blade_squared = w_2_blade_squared - w_2_blade_axial**2  # m^2/s^2
        # Check if tangential velocity is non-negative, but take floating point round-off into account in case the flow
        # is purely axial. A factor of 10 is used here for robustness.
        roundoff_tolerance = 10 * np.finfo(float).eps * max(w_2_blade_squared, w_2_blade_axial**2)
        if (not np.isfinite(w_2_tangential_blade_squared)
                or w_2_tangential_blade_squared < -roundoff_tolerance):
            raise ValueError("Blade-row continuity requires an axial velocity greater than the relative outlet speed.")
        w_2_tangential_blade = -np.sqrt(max(0, w_2_tangential_blade_squared))  # m/s
        beta_2_blade = np.atan2(w_2_tangential_blade, w_2_blade_axial)  # rad
        v_2_blade = np.hypot(w_2_blade_axial, u + w_2_tangential_blade)  # m/s

        # Some other quantities can be also calculated for the blade row alone.
        h_t2_blade = h_2_blade + v_2_blade**2 / 2
        p_t2_blade = p_2_blade * (h_t2_blade / h_2_blade)**(gas.Cp / gas.R)  # Pa
        alpha_2_blade = np.atan2(u + w_2_tangential_blade, w_2_blade_axial)  # rad
        psi_tt_blade = theta_1_blade * np.tan(alpha_1) - theta_2_blade * np.tan(beta_2_blade) - 1  # -
        R_h_tt_blade = (h_1_blade - h_2_blade) / (h_0 - h_t2_blade)  # -
        p_0_over_p_2_blade = p_0 / p_2_blade  # -
        M_r2_blade = w_2_blade / gas.calculate_sound_velocity(T_2_blade)  # -
        M_r1_blade = w_1_blade / gas.calculate_sound_velocity(T_1_blade)  # -

        # Calculate Reynolds numbers
        Re_s1_blade = rho_1_blade * v_1_blade * self.c_stator / gas.calculate_dynamic_viscosity(T_1_blade)  # -
        Re_r2_blade = rho_2_blade * w_2_blade * self.c_rotor / gas.calculate_dynamic_viscosity(T_2_blade)  # -

        # Now pack the results and return them
        blade_row_results = {"v_1_blade": v_1_blade,  # m/s
                             "w_1_blade": w_1_blade,  # m/s
                             "p_1_blade": p_1_blade,  # Pa
                             "T_1_blade": T_1_blade,  # K
                             "h_1_blade": h_1_blade,  # J/kg
                             "rho_1_blade": rho_1_blade,  # kg/m^3
                             "M_s1_blade": M_s1_blade,  # -
                             "v_2_blade": v_2_blade,  # m/s
                             "w_2_blade": w_2_blade,  # m/s
                             "p_2_blade": p_2_blade,  # Pa
                             "p_t2_blade": p_t2_blade,  # Pa
                             "T_2_blade": T_2_blade,  # K
                             "h_2_blade": h_2_blade, # J/kg
                             "h_t2_blade": h_t2_blade, # J/kg
                             "rho_2_blade": rho_2_blade,  # kg/m^3
                             "M_r1_blade": M_r1_blade,  # -
                             "M_r2_blade": M_r2_blade,  # -
                             "theta_2_blade": theta_2_blade,  # -
                             "alpha_2_blade": alpha_2_blade,  # rad
                             "beta_2_blade": beta_2_blade,  # rad
                             "alpha_2_blade_deg": alpha_2_blade * 180 / np.pi,  # deg
                             "beta_2_blade_deg": beta_2_blade * 180 / np.pi,  # deg
                             "psi_tt_blade": psi_tt_blade,  # -
                             "R_h_tt_blade": R_h_tt_blade,  # -
                             "p_0_over_p_2_blade": p_0_over_p_2_blade,  # -
                             "Re_s1_blade": Re_s1_blade,  # -
                             "Re_r2_blade": Re_r2_blade,  # -
                             }
        return blade_row_results

    def print_analysis_results(self):
        """Print stored stage results, rotor blade-row outlet results and loss breakdowns as GitHub tables.

        :raises ValueError: If no design-point analysis is available. Call size_turbine first.
        """

        # First obtain the stored design-point results and check that analysis has been performed.
        results = self.analysis_results_at_design_point
        if results.get("eta_tt") is None:
            raise ValueError("No design-point analysis results available. Call size_turbine() first.")

        # Define a helper to convert units and format each entry with its symbol, value and unit.
        def format_entry(symbol, unit, factor, source_results):
            # Leave missing quantities blank.
            value = source_results.get(symbol)
            if value is None:
                return ""
            # Use three significant figures for all zeta coefficients, retaining trailing zeros.
            # Other quantities keep two decimal places; dimensionless entries have no unit suffix.
            number_format = "#.3g" if "zeta" in symbol else ".2f"
            return f"{symbol}, {value * factor:{number_format}} {unit}".rstrip()

        # Define whole-turbine quantities in display order, together with units and conversion factors.
        turbine_fields = [
            ("psi_tt", "", 1),
            ("theta_2", "", 1),
            ("R_h_tt_ideal", "", 1),
            ("p_0_over_p_2", "", 1),
            ("P_shaft", "kW", 1e-3),
            ("RPM", "rpm", 1),
            ("u", "m/s", 1),
            ("eta_tt", "%", 100),
            ("eta_ts", "%", 100),
            ("eta_poly_tt", "%", 100),
            ("eta_poly_ts", "%", 100),
        ]
        # Map each station quantity to its stored symbols, display unit and conversion factor.
        # None marks an unavailable quantity. Entropy rises are shown separately for each loss group.
        station_fields = [
            (("T_0", "T_1", "T_2"), "K", 1),
            (("p_0", "p_1", "p_2"), "bar", 1e-5),
            ((None, "p_t1", "p_t2"), "bar", 1e-5),
            ((None, "h_1", "h_2"), "kJ/kg", 1e-3),
            (("h_t0", "h_t1", "h_t2"), "kJ/kg", 1e-3),
            ((None, "rho_1", "rho_2"), "kg/m^3", 1),
            ((None, "delta_s_stator", "delta_s_rotor"), "J/kg", 1),
            ((None, None, "delta_s_rotor_additional"), "J/kg", 1),
            ((None, "v_1", "v_2"), "m/s", 1),
            ((None, "w_1", "w_2"), "m/s", 1),
            ((None, "alpha_1", "alpha_2"), "deg", 180 / np.pi),
            ((None, "beta_1", "beta_2"), "deg", 180 / np.pi),
            ((None, "M_s1", "M_s2"), "", 1),
            ((None, "M_r1", "M_r2"), "", 1),
            ((None, "M_ax_1", "M_ax_2"), "", 1),
            ((None, "M_s1_ideal", "M_r2_ideal_r_real_s"), "", 1),
            ((None, "theta_1", "theta_2"), "", 1),
            ((None, "Re_s1", "Re_r2"), "", 1),
        ]
        # Format the first column with the whole turbine quantities.
        columns = [[format_entry(*field, source_results=results) for field in turbine_fields]]
        # Add a column for each station, keeping corresponding station quantities on the same row.
        for station in range(3):
            columns.append([format_entry(symbols[station], unit, factor, source_results=results)
                            for symbols, unit, factor in station_fields])

        # Transpose the columns into rows, pad shorter columns with blanks and print a GitHub-style table.
        headers = ["Turbine", "Station 0", "Station 1", "Station 2"]
        print(tabulate(zip_longest(*columns, fillvalue=""), headers=headers, tablefmt="github"))

        # Define rotor blade-row outlet quantities, using the same units as the stage table.
        blade_row_fields = [
            ("T_2_blade", "K", 1),
            ("p_2_blade", "bar", 1e-5),
            ("p_t2_blade", "bar", 1e-5),
            ("h_2_blade", "kJ/kg", 1e-3),
            ("h_t2_blade", "kJ/kg", 1e-3),
            ("rho_2_blade", "kg/m^3", 1),
            ("v_2_blade", "m/s", 1),
            ("w_2_blade", "m/s", 1),
            ("alpha_2_blade", "deg", 180 / np.pi),
            ("beta_2_blade", "deg", 180 / np.pi),
            ("M_r2_blade", "", 1),
            ("psi_tt_blade", "", 1),
            ("theta_2_blade", "", 1),
            ("Re_r2_blade", "", 1),
        ]
        # Format and print the station 2 results for the rotor blade row without additional losses.
        blade_row_column = [format_entry(*field, source_results=self.blade_row_results_at_design_point)
                            for field in blade_row_fields]
        print()
        print(tabulate([[entry] for entry in blade_row_column if entry],
                       headers=["Station 2, rotor blade row alone\nwithout additional losses"], tablefmt="github"))

        # Combine the stored loss results with the stage's Denton coefficients for the loss table.
        loss_results = self.loss_model_results_at_design_point.copy()
        loss_results.update({"zeta_stator_Denton": results.get("zeta_stator_Denton"),
                             "zeta_rotor_Denton": results.get("zeta_rotor_Denton")})

        # Define the existing loss-summary quantities, omitting delta_h_loss_rotor_sum_static.
        loss_fields = [
            ("zeta_aerodynamic_stator", "", 1),
            ("zeta_incidence_stator", "", 1),
            ("zeta_aerodynamic_rotor", "", 1),
            ("zeta_incidence_rotor", "", 1),
            ("zeta_disk_friction", "", 1),
            ("zeta_clearance_rotor", "", 1),
            ("zeta_admission", "", 1),
            ("zeta_rotor_additional", "", 1),
            ("zeta_stator_Denton", "", 1),
            ("zeta_rotor_Denton", "", 1),
            ("delta_h_loss_stator", "kJ/kg", 1e-3),
            ("delta_h_loss_rotor", "kJ/kg", 1e-3),
            ("delta_h_loss_rotor_additional", "kJ/kg", 1e-3),
            ("delta_h_loss_rotor_sum", "kJ/kg", 1e-3),
            ("delta_s_stator", "J/kg", 1),
            ("delta_s_rotor", "J/kg", 1),
            ("delta_s_rotor_additional", "J/kg", 1),
            ("mdot_leak", "kg/s", 1),
        ]
        # Define the aerodynamic components in the same order for the stator and rotor breakdowns.
        aerodynamic_loss_fields = [
            "zeta_p0", "corrected_zeta_p0", "zeta_h", "zeta_C", "zeta_p", "zeta_f",
            "integrated_zeta_p", "zeta_a", "zeta_short_blade", "zeta_rest",
        ]

        # Format the loss summary followed by the stator and rotor aerodynamic breakdown columns.
        loss_columns = [[format_entry(*field, source_results=loss_results) for field in loss_fields]]
        for blade_row in ("stator", "rotor"):
            loss_columns.append([format_entry(f"{symbol}_{blade_row}", "", 1, source_results=loss_results)
                                 for symbol in aerodynamic_loss_fields])

        # Print the loss table, padding the shorter aerodynamic breakdowns with blank entries.
        loss_headers = ["Turbine loss coefficients and\ndissipated enthalpies",
                        "Stator aerodynamic loss\ncoefficient - breakdown",
                        "Rotor aerodynamic loss\ncoefficient - breakdown"]
        print()
        print(tabulate(zip_longest(*loss_columns, fillvalue=""), headers=loss_headers, tablefmt="github"))

    def plot_velocity_triangles(self):
        """Plot the design-point velocity triangles at stations 1 and 2.

        :raises ValueError: If design-point velocities are unavailable. Call size_turbine first.
        """

        # Obtain the stored velocity magnitudes and flow angles required to draw both triangles.
        results = self.analysis_results_at_design_point
        required_fields = ("u", "v_1", "v_2", "w_1", "w_2", "alpha_1", "alpha_2", "beta_1", "beta_2")
        if any(results.get(field) is None for field in required_fields):
            raise ValueError("No design-point velocity results available. Call size_turbine() first.")

        # Create two subplots with common velocity scales and a shared tangential-velocity axis.
        figure, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 5), layout="compressed")
        u = results["u"]  # m/s

        # Reconstruct the absolute and relative velocity components from angles measured from the axial direction.
        for station, axis in zip((1, 2), axes):
            v = results[f"v_{station}"]  # m/s
            w = results[f"w_{station}"]  # m/s
            alpha = results[f"alpha_{station}"]  # rad
            beta = results[f"beta_{station}"]  # rad
            absolute_tip = (v * np.cos(alpha), v * np.sin(alpha))
            relative_tip = (w * np.cos(beta), u + w * np.sin(beta))

            # Position the vectors head to tail so that the blade-speed and relative vectors form the absolute one.
            vectors = [
                ((0, 0), absolute_tip, "red", f"Absolute velocity, v_{station}"),
                ((0, u), relative_tip, "blue", f"Relative velocity, w_{station}"),
                ((0, 0), (0, u), "cyan", "Blade speed, u"),
            ]

            # Draw each velocity line with a legend label and an arrowhead at its endpoint.
            for start, end, color, label in vectors:
                axis.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=2, label=label)
                axis.annotate("", xy=end, xytext=start,
                              arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 2,
                                          "mutation_scale": 15, "shrinkA": 0, "shrinkB": 0})

            # Label each station and use equal coordinate scales to preserve the actual triangle angles.
            axis.set_title(f"Station {station}")
            axis.set_xlabel("Axial velocity (m/s)")
            axis.set_aspect("equal", adjustable="box")

            # Add major and minor grids behind the vectors and a legend in each subplot's upper-right corner.
            axis.set_axisbelow(True)
            axis.minorticks_on()
            axis.grid(which="major", alpha=0.4)
            axis.grid(which="minor", linestyle=":", alpha=0.25)
            axis.legend(loc="upper right")

        # Leave space around the arrowheads and label the shared tangential-velocity axis.
        axes[0].margins(x=0.08, y=0.05)
        axes[0].set_ylabel("Tangential velocity (m/s)")

        # Measure the rendered legends and arrows so that the layout adapts to the actual velocity triangles.
        for _ in range(20):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            padding = renderer.points_to_pixels(8)
            legend_bounds = [axis.get_legend().get_window_extent(renderer) for axis in axes]

            # Enlarge the figure if necessary to fit the legends without widening the axial-velocity bounds.
            width_scale = max((legend.width + 2 * padding) / axis.get_window_extent(renderer).width
                              for axis, legend in zip(axes, legend_bounds))
            if width_scale > 1.01:
                width, height = figure.get_size_inches()
                required_width = sum(legend.width + 2 * padding for legend in legend_bounds) / figure.dpi
                figure.set_size_inches(max(width, required_width + 1.5), height * width_scale, forward=True)
                continue

            # Reserve enough space above every arrow for the legend and a small gap, using display coordinates.
            y_min, y_max = axes[0].get_ylim()
            upper_limit = y_max
            for axis, legend in zip(axes, legend_bounds):
                arrow_top = max(annotation.arrow_patch.get_window_extent(renderer).y1
                                for annotation in axis.texts if annotation.arrow_patch is not None)
                overlap = arrow_top + padding - legend.y0
                if overlap > 0.5:
                    data_coordinates = axis.transData.inverted()
                    extra_margin = (data_coordinates.transform((0, overlap))[1]
                                    - data_coordinates.transform((0, 0))[1])
                    upper_limit = max(upper_limit, y_max + extra_margin)

            # Repeat only when the shared upper bound changed, since equal aspect can change the rendered layout.
            if upper_limit == y_max:
                break
            axes[0].set_ylim(y_min, upper_limit)

        # Display the velocity triangles.
        plt.show()
