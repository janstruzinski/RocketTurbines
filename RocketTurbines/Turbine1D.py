import numpy as np
from scipy.optimize import root_scalar, root
from .TraupelLossModel import TraupelLossModel

class Turbine1D:
    def __init__(self):
        """A class representing 1D, nonisentropic turbine. This class was created with supersonic turbines in """ \
        """mind, but can be also used for subsonic ones. It takes into account nonisentropic processes when """ \
        """calculating fluid states and flow velocities and angles. The object can automatically size the turbine """ \
        """based on given stage coefficients, massflow and shaft power through Turbine1D.size_turbine method.

        This method uses numerical solver, which calls Turbine1D.calculate_entropy_rise and finds consistent """ \
        """entropy increase through each blade row, such that assumed and calculated entropies are with agreement """ \
        """with each other. This is the outer numerical loop. During each evaluation of """ \
        """Turbine1D.calculate_entropy_rise there are two internal numerical loops.

        The first one calculates consistent thermodynamic state at each station, by varying pressure ratio """ \
        """through the turbine such that energy balance is achieved. After this, the blade angles are calculated. """ \
        """Once these are known, second internal numerical loop finds flow states for blade rows alone, without """ \
        """any consideration for partial admission, clearance and disk friction losses and only taking into """ \
        """account aerodynamic losses in blade row passages. This numerical loop varies flow coefficient for the """ \
        """blade row alone, until it agrees with the calculated one. Flow state for blade row alone is required by """ \
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
        #    |      +--> INNER LOOP 1 -----------------------------------------+            |
        #    |      |      Vary: stage pressure ratio                          |            |
        #    |      |      Residual: calculated minus prescribed outlet        |            |
        #    |      |                total enthalpy                            |            |
        #    |      |      |                                                   |            |
        #    |      |      v                                                   |            |
        #    |      |    Turbine1D.__calculate_thermodynamic_properties --------+            |
        #    |      |                                                                       |
        #    |      v  (after inner loop 1 converges)                                       |
        #    |    Analysis results: station states and velocities                           |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    Turbine1D.__calculate_flow_angles                                          |
        #    |      |                                                                       |
        #    |      v                                                                       |
        #    |    Analysis results with flow angles, used as metal angles                    |
        #    |      |                                                                       |
        #    |      +--> INNER LOOP 2 -----------------------------------------+            |
        #    |      |      Vary: blade-row outlet flow coefficient             |            |
        #    |      |      Residual: calculated minus assumed blade-row        |            |
        #    |      |                outlet flow coefficient                   |            |
        #    |      |      |                                                   |            |
        #    |      |      v                                                   |            |
        #    |      |    Turbine1D.__calculate_blade_row_velocities ------------+            |
        #    |      |                                                                       |
        #    |      v  (after inner loop 2 converges)                                       |
        #    |    Blade-row results: states and velocities                                   |
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
        # _blade represent velocities or thermodynamic properties representative of flow in stationary test cascade.
        # By default, angles refer to flow angles. If they refer to geometric metal blade angle, they have a suffix
        # _metal.

        # Assumptions:
        # 1. This class utilizes ideal gas model for calculations.
        # 2. Velocity at station 0 is assumed to be negligible.
        # 3. Flow areas at station 1 and 2 are equal.
        # 4. Constant Euler diameter.
        # 5. Axial width of the rotor blade row assumed to be equal to chord of the blade.

        # Geometry stored in the object properties
        self.D_hub = None   # Hub diameter, m
        self.D_tip = None   # Tip diameter, m
        self.D_mean = None  # Mean diameter, m
        self.D_Euler = None # Euler diameter, m
        self.R_hub = None   # Hub radius, m
        self.R_tip = None   # Tip radius, m
        self.R_mean = None  # Mean radius, m
        self.R_Euler = None # Euler radius, m
        self.l_stator = None  # Stator blade/nozzle radial length, m
        self.l_rotor = None # Rotor blade radial length, m
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
        self.t_TE_rotor = None # Rotor trailing edge thickness, m.
        self.alpha_1_metal_deg = None  # Metal outlet angle of the stator blades/nozzle, deg
        self.beta_1_metal_deg = None  # Metal inlet angle of the rotor blades, deg
        self.beta_2_metal_deg = None  # Metal outlet angle of the rotor blades, deg
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
                                                 "psi": None, # Real loading coefficient, -
                                                 "psi_ideal": None, # Total-to-total ideal work coefficient, -
                                                 "psi_ideal_design": None,  # Total-to-total ideal work coefficient
                                                 # at the design point, -
                                                 "theta_1": None, # Flow coefficient at station 1, -
                                                 "theta_2": None, # Flow coefficient at station 2, -
                                                 "R_h_ideal": None, # Ideal enthalpic reaction of the stage, -
                                                 "R_h": None,   # Real enthalpic reaction of the stage, -
                                                 "R_p": None,   # Real pressure reaction of the stage, -
                                                 "p_0_over_p_2": None,  # Pressure ratio of the stage, -
                                                 "u": None, # Blade velocity at Euler radius, m/s
                                                 "P_shaft": None,   # Shaft power of the turbine, W
                                                 "p_1": None,   # Static pressure at station 1, Pa
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
                                                 "p_2": None, # Static pressure at station 2, Pa
                                                 "T_2": None, # Static temperature at station 2, K
                                                 "rho_2": None, # Static density at station 2, kg/m3
                                                 "T_2_ideal_r_real_s": None, # Static temperature assuming
                                                 # ideal expansion at station 2, K
                                                 "h_2": None, # Static enthalpy at station 2, J/kg
                                                 "h_t2": None, # Total enthalpy at station 2, J/kg
                                                 "h_2_ideal_r_real_s": None, # Static enthalpy assuming ideal
                                                 # expansion at station 2, J/kg
                                                 "delta_s_rotor": None, # Total entropy increase at rotor, J/kg/K
                                                 "delta_s_rotor_additional": None,  # Total entropy increase at rotor
                                                 # due to additional losses like clearances, disk friction or partial
                                                 # admission, J/kg/K
                                                 "a_2": None, # Sound velocity at station 2, m/s
                                                 "a_2_ideal_r_real_s": None, # Sound velocity assuming ideal
                                                 # expansion at station 2, m/s
                                                 "v_2": None, # Absolute velocity at station 2, m/s
                                                 "w_2": None, # Relative velocity at station 2, m/s
                                                 "w_2_ideal_r_real_s": None, # Relative velocity assuming ideal
                                                 # expansion at station 2, m/s
                                                 "M_s2": None, # Station 2 Mach number in stationary frame, -
                                                 "M_r2": None, # Mach number in rotary reference frame at station 2, -
                                                 "M_r2_ideal_r_real_s": None, # Mach number in rotary reference
                                                 # frame assuming ideal expansion at station 2, -
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
        # Analysis results at the design point for the blade row alone if there were no clearance, disk friction or
        # partial admission losses.
        self.blade_row_results_at_design_point = {
            "v_1_blade": None,  # Absolute velocity at station 1 for the blade row alone, m/s
            "w_1_blade": None,  # Relative velocity at station 1 for the blade row alone, m/s
            "p_1_blade": None,  # Static pressure at station 1 for the blade row alone, Pa
            "T_1_blade": None,  # Static temperature at station 1 for the blade row alone, K
            "rho_1_blade": None,  # Static density at station 1 for the blade row alone, kg/m3
            "M_s1_blade": None,  # Absolute Mach number at station 1 for the blade row alone, -
            "v_2_blade": None,  # Absolute velocity at station 2 for the blade row alone, m/s
            "w_2_blade": None,  # Relative velocity at station 2 for the blade row alone, m/s
            "p_2_blade": None,  # Static pressure at station 2 for the blade row alone, Pa
            "T_2_blade": None,  # Static temperature at station 2 for the blade row alone, K
            "rho_2_blade": None,  # Static density at station 2 for the blade row alone, kg/m3
            "M_r1_blade": None,  # Relative Mach number at station 1 for the blade row alone, -
            "M_r2_blade": None,  # Relative Mach number at station 2 for the blade row alone, -
            "alpha_2_blade": None,  # Absolute flow angle at station 2 for the blade row alone, rad
            "psi_blade": None,  # Loading coefficient for the blade row alone, -
            "R_h_blade": None,  # Enthalpic reaction for the blade row alone, -
            "p_0_over_p_2_blade": None,  # Pressure ratio for the blade row alone, -
            "Re_s1_blade": None,  # Reynolds number based on v_1_blade and stator chord, -
            "Re_r2_blade": None,  # Reynolds number based on v_2_blade and rotor chord, -
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
            "delta_h_loss_rotor_additional": None,  # Specific enthalpy dissipated by additional rotor losses, J/kg
            "delta_h_loss_rotor_total": None,  # Total specific enthalpy dissipated in the rotor, J/kg
            "delta_s_stator": None,  # Specific entropy rise across the stator, J/(kg K)
            "delta_s_rotor": None,  # Specific entropy rise across the rotor blade row, J/(kg K)
            "delta_s_rotor_additional": None,  # Specific entropy rise from additional rotor losses, J/(kg K)
            "mdot_leak": None,  # Mass flow leaking through the rotor shroud seal (zero if unshrouded), kg/s
        }

    def size_turbine(self, gas, loading_coefficient, flow_coefficient, reaction_isentropic, RPM,
                     shaft_power, mdot, T_0, p_0, radial_clearance, no_blades_stator, no_blades_rotor,
                     chord_over_pitch_stator, t_TE_stator, t_TE_rotor, Ra_roughness,
                     loss_model: TraupelLossModel, admission_fraction=1, partial_admission_rotor="free",
                     chord_over_pitch_rotor=2.5,
                     s_ax_over_pitch_rotor=0.35, delta_s_estimate=[0, 0, 0], shrouded_rotor=False,
                     h_shroud_over_blade_length=0.008, s_ax_shroud_over_h_shroud=2, t_shroud=None,
                     seal_teeth_number=None):
        """A method to size the turbine based on given requirements. It changes properties of the object.
        This method calls calculate_entropy_rise, which is an outer numerical solution loop, to calculate consistent
        fluid and flow state at each station. This allows to calculate the geometry of the turbine.

        :param IdealGas gas: IdealGas object representing working gas of the turbine.
        :param float or integer loading_coefficient: Real, nonisentropic loading coefficient of the stage.
        :param float or integer flow_coefficient: Flow coefficient of the turbine at station 2.
        :param float reaction_isentropic: Enthalpic, isentropic reaction of the turbine. It should be noted enthalpic
            reaction is not the same as pressure reaction, although two quantities are similar.
        :param float or integer RPM: Rotations per minute of the turbine.
        :param float or integer shaft_power: Shaft power (W) that the turbine must deliver.
        :param float or integer mdot: Massflow (kg/s) through the turbine.
        :param float or integer T_0: Total temperature (K) at station 0. It is assumed it is approximately equal to
            static temperature there due to negligible velocity.
        :param float or integer p_0: Total pressure (Pa) at station 0. It is assumed it is approximately equal to
            static pressure there due to negligible velocity.
        :param float or integer radial_clearance: Radial clearance (m) of the rotor blade.
        :param integer no_blades_stator: Number of stator (-) blades/nozzles.
        :param integer no_blades_rotor: Number of rotor (-) blades.
        :param float or integer chord_over_pitch_stator: Ratio of stator chord to its blade/nozzle pitch (-).
        :param float or integer chord_over_pitch_rotor: Ratio of rotor chord to its blade pitch (-). By default,
         equal to 2.5, which is in the range of 2.5-3.3 recommended for supersonic impulse rotor cascades by Traupel
         in "Thermal Turbomachines".
        :param float or integer admission_fraction: Admission fraction (-) of the turbine stage. By default, 1.
        :param string partial_admission_rotor: Whether partial admission rotor (if used) is "enclosed" or "free".
         By default, "free".
        :param float or integer t_TE_stator: Tangential thickness (m) of the trailing edge for the stator.
         In other words, length of the projection of the thickness on the tangential axis.
        :param float or integer t_TE_rotor: Tangential thickness (m) of the trailing edge for the rotor.
         In other words, length of the projection of the thickness on the tangential axis.
        :param float or integer Ra_roughness: Roughness of the turbine flow surfaces in Ra.
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
          losses. By default, [0, 0, 0].
        :raises TypeError: If loss_model is not a TraupelLossModel instance.
        :raises ValueError: If the specified shaft work and outlet axial kinetic energy leave no positive outlet
            static enthalpy.
        :raises RuntimeError: If the entropy-rise, pressure-ratio or blade-row flow-coefficient solve does not converge.
        """

        if not isinstance(loss_model, TraupelLossModel):
            raise TypeError("loss_model must be a TraupelLossModel instance.")

        # Calculate the blade speed from given requirements
        delta_h = shaft_power / mdot  # J/kg
        u = np.sqrt(delta_h / loading_coefficient)  # m/s

        # Calculate angular speed and Euler diameter of the turbine
        omega = RPM * 2 * np.pi / 60  # rad/s
        self.D_Euler = 2 * u / omega  # m
        self.R_Euler = self.D_Euler / 2  # m

        # Calculate thermodynamic properties which are already known
        h_t0 = gas.Cp * T_0 # J/kg
        h_t2 = h_t0 - delta_h # J/kg
        T_t2 = h_t2 / gas.Cp # K

        # Calculate available geometry already
        self.p_stator = self.D_Euler * np.pi / no_blades_stator  # m
        self.p_rotor = self.D_Euler * np.pi / no_blades_rotor  # m
        self.c_stator = self.c_stator / chord_over_pitch_stator  # m
        self.c_rotor = self.c_rotor / chord_over_pitch_rotor  # m
        self.s_ax = self.p_rotor * s_ax_over_pitch_rotor  # m

        # Assign remaining geometry that is already known
        self.t_TE_rotor = t_TE_rotor  # m
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
        # Change Ra roughness to sand grain equivalent roughness
        self.sand_grain_roughness = 5.863 * Ra_roughness # m

        # Put known variables in analysis_results_at_design_point already
        self.analysis_results_at_design_point.update({"gas": gas,
                                                      "psi": loading_coefficient,  # -
                                                      "theta_2": flow_coefficient,  # -
                                                      "R_h_ideal": reaction_isentropic,  # -
                                                      "RPM": RPM,  # rpm
                                                      "omega": omega,  # rad/s
                                                      "u": u,  # m/s
                                                      "P_shaft": shaft_power,  # W
                                                      "mdot_total": mdot,  # kg/s
                                                      "delta_h": delta_h, # J/kg
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
        # Get an estimate of entropy rise scales for normalization of the residual. This estimate are the entropy
        # scales for assumed values equal to zero. This represents the result of loss model as if all input velocities
        # and flow angles were ideal.
        entropy_scales = get_entropy_residual([0, 0, 0])
        # Solve for entropy increase
        entropy_solution = root(get_entropy_residual, np.array(delta_s_estimate), method="hybr",
                                options={"xtol":1e-6, "eps": 1e-12, "maxfev": 1000, "factor": 1,
                                         "diag": 1/entropy_scales})
        # Raise an error if not converged
        if not entropy_solution.success:
            raise RuntimeError("Numerical solve for the stator and rotor entropy rises did not converge.")
        # Get entropy increases and the rest of the results
        delta_s_stator, delta_s_rotor, delta_s_rotor_additional = entropy_solution.x  # J/(kg K)
        _, analysis_results, blade_row_results, loss_model_results = \
            self.calculate_entropy_rise(delta_s_stator, delta_s_rotor, delta_s_rotor_additional, loss_model)


        # Update analysis_results_at_design_point and blade row results
        self.analysis_results_at_design_point = analysis_results
        self.blade_row_results_at_design_point = blade_row_results
        self.loss_model_results_at_design_point = loss_model_results

        # Get flow angles, which become metal angles
        self.alpha_1_metal_deg = self.analysis_results_at_design_point["alpha_1"]  # rad
        self.beta_1_metal_deg = self.analysis_results_at_design_point["beta_1"]  # rad
        self.beta_2_metal_deg = self.analysis_results_at_design_point["alpha_2"]  # rad

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
        model, together with the turbine analysis results. This method has two internal numerical loops - one to
         calculate consistent thermodynamic state at each station for assumed entropy increases through blade rows with
         _calculate_thermodynamic_variables, and another one to calculate flow states for blade rows alone without
         inclusion of additional losses like clearance, partial admission or disk friction.

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
        :raises ValueError: If shaft work and outlet axial kinetic energy leave no positive outlet static enthalpy.
        :raises RuntimeError: If the pressure-ratio or blade-row flow-coefficient solve does not converge.
        """

        if not isinstance(loss_model, TraupelLossModel):
            raise TypeError("loss_model must be a TraupelLossModel instance.")

        # In rocket turbines, there are no clearance losses in stators, while partial admission losses belong to the
        # rotor. Therefore, entropy generation in the stator due to aerodynamic losses constitute total entropy
        # generation through the rotor.
        delta_s_stator_total = delta_s_stator  # J/(kg K)
        # In the rotor, aside from aerodynamic losses, there are additional losses like clearance, partial admission
        # or disk friction losses. Total entropy generation is thus the sum of all of these.
        delta_s_rotor_total = delta_s_rotor + delta_s_rotor_additional  # J/(kg K)


        # Thermodynamic properties at each station must be calculated. Thermodynamic properties depend on pressure ratio
        # p_0_over_p_2. It needs to be numerically found, such that energy balance is satisfied and total enthalpy at
        # station 2 agrees with the already calculated one.
        # First, retrieve some variables from analysis_results_at_design_point that will be used to obtain solution
        # estimate
        gas = self.analysis_results_at_design_point["gas"]
        h_t0 = self.analysis_results_at_design_point["h_t0"]  # J/kg
        h_t2 = self.analysis_results_at_design_point["h_t2"]  # J/kg
        u = self.analysis_results_at_design_point["u"]  # m/s
        theta_2 = self.analysis_results_at_design_point["theta_2"]  # -
        # The known axial velocity sets the largest possible exit static enthalpy (zero exit swirl).
        h_2_max = h_t2 - (theta_2 * u)**2 / 2  # J/kg
        if h_2_max <= 0:
            raise ValueError("Shaft work and exit axial kinetic energy leave no positive exit static enthalpy.")
        # This static enthalpy can be used to calculate loss-free lower bound on p_0/p_2:
        p_0_over_p_2_ideal = (h_t0 / h_2_max)**(gas.Cp / gas.R)  # -
        # Now minimum pressure ratio that takes into account losses can be calculated
        delta_s_total = delta_s_stator_total + delta_s_rotor_total
        p_0_over_p_2_min = p_0_over_p_2_ideal * np.exp(delta_s_total / gas.R)

        # That minimum value is a solution estimate that can be used with the Newton scheme
        # Define a function that obtains residual
        def get_PR_residual(p_0_over_p_2_estimate):
            return self.__calculate_thermodynamic_properties(p_0_over_p_2_estimate, delta_s_stator_total,
                                                             delta_s_rotor_total, delta_s_rotor_additional)[1]
        # Solve it
        PR_solution = root_scalar(get_PR_residual, x0=p_0_over_p_2_min, method="newton", maxiter=1000, xtol=1e-10,
                                  rtol=1e-8)
        # Raise error if the solution is not converged
        if not PR_solution.converged:
            raise RuntimeError("Numerical solve for p_0/p_2 did not converge.")
        # Obtain solution and the remaining results
        p_0_over_p_2 = PR_solution.root  # -
        analysis_results, residual = self.__calculate_thermodynamic_properties(
            p_0_over_p_2, delta_s_stator_total, delta_s_rotor_total, delta_s_rotor_additional)  # residual: -


        # Flow angles can be now calculated. These are equal to metal angles. These are affected by the total entropy
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


        # Recalculate velocities for the rotor and its calculated angles, but without additional losses like friction,
        # clearance or partial admission, as the loss model may require separate blade row velocities. Such velocity
        # only takes into account aerodynamic losses / friction losses in the blade passage itself and is thus
        # representative of stationary test cascades. To reconstruct these blade row results, theta_2_blade is needed.
        # This is theta_2 for the blade angles above, if delta_s_rotor_additional was zero. However, it is also an
        # outcome of these calculations. Thus, the model is implicit and a numerical solve is again needed.
        # Bracketing scheme will be used, so first theta_2_max is estimated. This is theta_2 as T_2 approaches zero.
        h_1 = analysis_results["h_1"]  # J/kg
        w_1 = analysis_results["w_1"]  # m/s
        u = analysis_results["u"]  # m/s
        gas = analysis_results["gas"]
        theta_2_max = np.sqrt(2 * h_1 + w_1**2) / (u * np.sqrt(1 + np.tan(beta_2)**2))  # -
        # The bracket used can be supersonic or subsonic. theta_2_crit divides these two ranges. Whichever bracket is
        # used depends on the value of original theta_2.
        theta_2_crit = theta_2_max * np.sqrt((gas.gamma - 1) / (gas.gamma + 1))  # -
        subsonic_branch = [1e-3 * theta_2_crit, theta_2_crit]
        supersonic_branch = [theta_2_crit, (1 - 1e-3) * theta_2_max]
        branch = subsonic_branch if theta_2 < theta_2_crit else supersonic_branch
        # Define residual function to solve
        def get_theta_2_blade_residual(theta_2_blade):
            _, residual = self.__calculate_blade_row_velocities(
                alpha_1, beta_2, theta_2_blade, analysis_results)  # residual: -
            return residual
        residual_at_branch = [get_theta_2_blade_residual(theta) for theta in branch]
        # Use toms748 if bracket gives opposite results
        if residual_at_branch[0] * residual_at_branch[1] < 0:
            theta_2_blade_solution = root_scalar(get_theta_2_blade_residual, bracket=branch, method="toms748",
                                                 options={"k": 2}, maxiter=1000, xtol=1e-10, rtol=1e-8)
            # If the solution does not converge, attempt Newton scheme with theta_2 as initial estimate
            if not theta_2_blade_solution.converged:
                theta_2_blade_solution = root_scalar(get_theta_2_blade_residual, x0=theta_2, method="newton".upper(),
                                                     maxiter=1000, xtol=1e-10, rtol=1e-8)
        # If bracket does not give opposite results, also use Newton scheme with theta_2 as initial estimate
        else:
            theta_2_blade_solution = root_scalar(get_theta_2_blade_residual, x0=theta_2, method="newton",
                                                 maxiter=1000, xtol=1e-10, rtol=1e-8)
        # Raise an error if the solution is not converged
        if not theta_2_blade_solution.converged:
            raise RuntimeError("Numerical solve for theta_2_blade did not converge.")
        # Obtain remaining results
        theta_2_blade = theta_2_blade_solution.root  # -
        blade_row_results, _ = self.__calculate_blade_row_velocities(alpha_1, beta_2, theta_2_blade, analysis_results)

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
                            "t_TE_rotor": self.t_TE_rotor,  # m
                            "alpha_1_metal_deg": alpha_1 * 180 / np.pi,  # deg
                            "beta_1_metal_deg": beta_1 * 180 / np.pi,  # deg
                            "beta_2_metal_deg": beta_2 * 180 / np.pi,  # deg
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

    def __calculate_thermodynamic_properties(self, p_0_over_p_2, total_delta_s_stator, total_delta_s_rotor,
                                             delta_s_rotor_additional):
        """A method to calculate thermodynamic properties, velocities and Mach numbers at turbine stations 1 and 2 for
         the assumed pressure ratio. It also calculates energy residual at station 2 for that ratio.

        Definitions of the values read from analysis_results_at_design_point are in Turbine1D.__init__.

        :param float p_0_over_p_2: Total/static pressure at station 0 over static pressure at station 2.
        :param float total_delta_s_stator: Total specific entropy rise across the stator (J/kg/K).
        :param float total_delta_s_rotor: Total specific entropy rise across the rotor (J/kg/K).
        :param float delta_s_rotor_additional: Specific entropy rise due to clearance, partial admission and disk
            friction losses in the rotor (J/kg/K).
        :return: Dictionary with turbine analysis results and residual of the total enthalpy at station 2,
            respectively.
        :rtype: tuple[dict, float]
        """

        # Retrieve already known variables from analysis_results_at_design_point
        psi = self.analysis_results_at_design_point["psi"]  # -
        R_h_ideal = self.analysis_results_at_design_point["R_h_ideal"]  # -
        theta_2 = self.analysis_results_at_design_point["theta_2"]  # -
        h_t2 = self.analysis_results_at_design_point["h_t2"]  # J/kg
        T_0 = self.analysis_results_at_design_point["T_0"]  # K
        p_0 = self.analysis_results_at_design_point["p_0"]  # Pa
        gas = self.analysis_results_at_design_point["gas"] # IdealGas object
        u = self.analysis_results_at_design_point["u"]  # m/s
        delta_h = self.analysis_results_at_design_point["delta_h"]  # J/kg

        # Calculate total entropy increase
        delta_s = total_delta_s_rotor + total_delta_s_stator  # J/(kg K)
        # Calculate h_0
        h_0 = gas.Cp * T_0  # J/kg
        # Calculate static pressure, temperature, density and enthalpy at station 2
        p_2 = p_0 / p_0_over_p_2  # Pa
        T_2 = T_0 * p_0_over_p_2**(-gas.R / gas.Cp) * np.exp(delta_s / gas.Cp) # K
        rho_2 = gas.calculate_density(p_2, T_2)  # kg/m^3
        h_2 = gas.Cp * T_2 # J/kg

        # Calculate isentropic reference state at station 2 assuming that h_0 = h_t0
        h_2_ideal = h_2 * np.exp(-delta_s / gas.Cp)
        h_t2_ideal = h_t2 * np.exp(-delta_s / gas.Cp)
        # Calculate difference between total ideal enthalpies
        delta_h_t_ideal = h_0 - h_t2_ideal
        # Now using ideal reaction definition, find ratio of isentropic enthalpies at station 1 and 2
        h_1_over_h_2_isentropic = 1 + R_h_ideal * delta_h_t_ideal / h_2_ideal
        # Now the ratio of real enthalpies can be found
        h_1_over_h_2 = h_1_over_h_2_isentropic * np.exp(-total_delta_s_rotor / gas.Cp) # -

        # Calculate p_1, T_1, rho_1, h_1, h_t1
        h_1 = h_2 * h_1_over_h_2  # J/kg
        T_1 = h_1 / gas.Cp  # K
        p_1 = p_2 * np.exp(total_delta_s_rotor / gas.R) * h_1_over_h_2**(gas.Cp / gas.R)  # Pa
        rho_1 = gas.calculate_density(p_1, T_1)  # kg/m^3
        # Energy is conserved between station 0 and station 1, so:
        h_t1 = h_0  # J/kg

        # Calculate theta_1
        theta_1 = theta_2 * rho_2 / rho_1  # -
        # Calculate real reaction from its definition
        R_h = (h_1 - h_2) / delta_h # -
        # Calculate absolute velocity at the outlet
        v_2_over_u = np.sqrt(theta_2**2 + (1 - R_h - psi / 2 + (theta_2**2 - theta_1**2) / (2 * psi))**2)  # -
        v_2 = v_2_over_u * u  # m/s
        # Calculate total enthalpy at the outlet
        h_t2_calculated = h_2 + v_2**2 / 2 # J/kg
        # Finally, the total enthalpy residual can be calculated
        residual = h_t2_calculated - h_t2 # J/kg

        # Calculate remaining values of interest.
        # First calculate ideal psi
        h_t2_over_h_t0 = h_t2_calculated / h_0  # -
        psi_ideal = psi * (1 - h_t2_over_h_t0 * np.exp(-delta_s / gas.Cp)) / (1 - h_t2_over_h_t0)  # -

        # Now velocity and Mach number at station 1. First calculate velocity of sound there:
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
        dummy_a1 = 1 - R_h + psi / 2 + (theta_2**2 - theta_1**2) / (2 * psi)  # -
        dummy_b1 = dummy_a1 - 1  # -
        w_1 = u * np.sqrt(theta_1**2 + dummy_b1**2)  # m/s
        M_r1 = w_1 / a_1  # -

        # Now velocity and Mach number at station 2. Ideal values at station 2
        # still assume real values at station 1. First calculate velocity of sound at station 2.
        a_2 = gas.calculate_sound_velocity(T_2)  # m/s
        h_2_ideal_r_real_s = h_1 * (p_2 / p_1) ** (gas.R / gas.Cp)  # J/kg
        T_2_ideal_r_real_s = h_2_ideal_r_real_s / gas.Cp  # K
        a_2_ideal_r_real_s = gas.calculate_sound_velocity(T_2_ideal_r_real_s)  # m/s
        # Real values in stationary reference frame:
        M_s2 = v_2 / a_2  # -
        # Now real and ideal values in rotary reference frame:
        dummy_a2 = dummy_a1 - psi  # -
        dummy_b2 = dummy_a2 - 1  # -
        w_2 = u * np.sqrt(theta_2**2 + dummy_b2**2)  # m/s
        delta_h_loss_rotor = h_2 - h_2_ideal_r_real_s  # J/kg
        w_2_ideal_r_real_s = np.sqrt(w_2**2 + 2 * delta_h_loss_rotor)  # m/s
        M_r2 = w_2 / a_2  # -
        M_r2_ideal_r_real_s = w_2_ideal_r_real_s / a_2_ideal_r_real_s  # -

        # Calculate real pressure reaction, R_p
        R_p = (p_1 - p_2) / (p_0 - p_2)  # -

        # Calculate Reynolds numbers
        Re_s1 = rho_1 * v_1 * self.c_stator / gas.calculate_dynamic_viscosity(T_1)  # -
        Re_r2 = rho_2 * w_2 * self.c_rotor / gas.calculate_dynamic_viscosity(T_2)  # -

        # Package thermodynamic properties, velocities and loading coefficients into analysis_results dictionary.
        # Make it a copy of analysis_results_at_design_point, such that the values there stay constant during
        # iterations.
        analysis_results = self.analysis_results_at_design_point.copy()
        analysis_results.update({"p_0_over_p_2": p_0_over_p_2, # -
                                 "h_0": h_0,  # J/kg
                                 "p_1": p_1,  # Pa
                                 "T_1": T_1,  # K
                                 "rho_1": rho_1,  # kg/m^3
                                 "T_1_ideal": T_1_ideal,  # K
                                 "h_1": h_1,  # J/kg
                                 "h_t1": h_t1,  # J/kg
                                 "h_1_ideal": h_1_ideal,  # J/kg
                                 "delta_s_stator": total_delta_s_stator,  # J/(kg K)
                                 "a_1": a_1,  # m/s
                                 "a_1_ideal": a_1_ideal,  # m/s
                                 "v_1": v_1,  # m/s
                                 "v_1_ideal": v_1_ideal,  # m/s
                                 "w_1": w_1,  # m/s
                                 "M_s1": M_s1,  # -
                                 "M_s1_ideal": M_s1_ideal,  # -
                                 "M_r1": M_r1,  # -
                                 "p_2": p_2,  # Pa
                                 "T_2": T_2,  # K
                                 "rho_2": rho_2,  # kg/m^3
                                 "T_2_ideal_r_real_s": T_2_ideal_r_real_s,  # K
                                 "h_2": h_2,  # J/kg
                                 "h_t2": h_t2,  # J/kg
                                 "h_2_ideal_r_real_s": h_2_ideal_r_real_s,  # J/kg
                                 "delta_s_rotor": total_delta_s_rotor - delta_s_rotor_additional,  # J/(kg K)
                                 "delta_s_rotor_additional": delta_s_rotor_additional,  # J/(kg K)
                                 "a_2": a_2,  # m/s
                                 "a_2_ideal_r_real_s": a_2_ideal_r_real_s,  # m/s
                                 "v_2": v_2,  # m/s
                                 "w_2": w_2,  # m/s
                                 "w_2_ideal_r_real_s": w_2_ideal_r_real_s,  # m/s
                                 "M_s2": M_s2,  # -
                                 "M_r2": M_r2,  # -
                                 "M_r2_ideal_r_real_s": M_r2_ideal_r_real_s,  # -
                                 "R_p": R_p,  # -
                                 "psi_ideal": psi_ideal,  # -
                                 "psi_ideal_design": psi_ideal,  # -
                                 "R_h": R_h,  # -
                                 "theta_1": theta_1,  # -
                                 "Re_s1": Re_s1,  # -
                                 "Re_r2": Re_r2,  # -
                                 })

        # Return residual and analysis results
        return analysis_results, residual

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
        R_h = analysis_results["R_h"]  # -
        psi = analysis_results["psi"]  # -
        # Calculate support variable
        dummy_theta = (theta_2**2 - theta_1**2) / (2 * psi)  # -
        # Calculate flow angles
        alpha_1 = np.atan2(1 - R_h + psi / 2 + dummy_theta, theta_1)  # rad
        beta_1 = np.atan2(-R_h + psi / 2 + dummy_theta, theta_1)  # rad
        beta_2 = np.atan2(-R_h - psi / 2 + dummy_theta, theta_2)  # rad
        alpha_2 = np.atan2(1 - R_h - psi / 2 + dummy_theta, theta_2)  # rad
        # Return all flow angles
        return alpha_1, beta_1, alpha_2, beta_2

    def __calculate_blade_row_velocities(self, alpha_1_metal, beta_2_metal, theta_2_blade,
                                         analysis_results):
        """A method to calculate velocities and thermodynamic properties for the blade row alone, without additional
        rotor losses such as clearance, partial admission or disk friction losses, for the assumed theta_2 of the blade
        row alone. It also calculates velocity residual in the form of difference between assumed and calculated values
        of theta_2.

        :param float alpha_1_metal: Outlet metal angle of the stator blades/nozzles (rad).
        :param float beta_2_metal: Outlet metal angle of the rotor blades (rad).
        :param float theta_2_blade: Assumed flow coefficient (-) at station 2 for the blade row alone.
        :param dict analysis_results: Dictionary with turbine analysis results required to calculate the blade row
            velocities and thermodynamic properties. Variable definitions are in Turbine1D.__init__.
        :return: Tuple containing a dictionary with blade row results and the residual between calculated and assumed
            flow coefficient at station 2.
        :rtype: tuple[dict, float]
        """

        # The inlet velocity to the rotor blade row at station 1 is unchanged, since additional rotor losses only affect
        # results at the station 2.
        gas = analysis_results["gas"]
        p_1_blade = analysis_results["p_1"]  # Pa
        T_1_blade = analysis_results["T_1"]  # K
        rho_1_blade = analysis_results["rho_1"]  # kg/m^3
        h_1_blade = analysis_results["h_1"]  # J/kg
        w_1_blade = analysis_results["w_1"]  # m/s
        v_1_blade = analysis_results["v_1"]  # m/s
        M_s1_blade = analysis_results["M_s1"]  # -
        theta_1_blade = analysis_results["theta_1"]  # -
        h_0 = analysis_results["h_0"]  # J/kg
        p_0 = analysis_results["p_0"]  # Pa
        # Unpack entropy increase across the rotor due to aerodynamic losses
        delta_s_rotor = analysis_results["delta_s_rotor"]  # J/(kg K)
        # Blade velocity is also the same
        u = analysis_results["u"]  # m/s

        # Based on assumed theta_2_blade, calculate velocities for the blade row alone at station 2.
        w_2_blade = np.sqrt(u**2 * theta_2_blade**2 * (1 + np.tan(beta_2_metal)**2))  # m/s
        v_2_blade = np.sqrt(u**2 * (theta_2_blade**2 + (1 + theta_2_blade * np.tan(beta_2_metal))**2))  # m/s

        # From the conservation of rotalphy, get enthalpy and temperature at station 2
        h_2_blade = h_1_blade + (w_1_blade**2 - w_2_blade**2) / 2  # J/kg
        T_2_blade = h_2_blade / gas.Cp  # K
        # Entropy increase across the rotor is known, so pressure at station 2 can be calculated. It will be higher
        # than pressure at station 2 for the whole stage.
        p_2_blade = p_1_blade * np.exp((gas.Cp * np.log(T_2_blade / T_1_blade) - delta_s_rotor) / gas.R)  # Pa
        rho_2_blade = gas.calculate_density(p_2_blade, T_2_blade)  # kg/m^3

        # From continuity, theta_2_blade_output can be calculated. The difference between the assumed and calculated
        # values can be also calculated.
        theta_2_blade_output = rho_1_blade * theta_1_blade / rho_2_blade  # -
        residual = theta_2_blade_output - theta_2_blade  # -

        # Some other quantities can be also calculated for the blade row alone.
        alpha_2_blade = np.atan2(1 + theta_2_blade * np.tan(beta_2_metal), theta_2_blade)  # rad
        psi_blade = theta_1_blade * np.tan(alpha_1_metal) - theta_2_blade * np.tan(beta_2_metal) - 1  # -
        R_h_blade = (h_1_blade - h_2_blade) / (h_0 - h_2_blade)  # -
        p_0_over_p_2_blade = p_0 / p_2_blade  # -
        M_r2_blade = w_2_blade / gas.calculate_sound_velocity(T_2_blade)  # -
        M_r1_blade = w_1_blade / gas.calculate_sound_velocity(T_1_blade)  # -

        # Calculate Reynolds numbers
        Re_s1_blade = rho_1_blade * v_1_blade * self.c_stator / gas.calculate_dynamic_viscosity(T_1_blade)  # -
        Re_r2_blade = rho_2_blade * v_2_blade * self.c_rotor / gas.calculate_dynamic_viscosity(T_2_blade)  # -

        # Now pack the results and return them together with residual
        blade_row_results = {"v_1_blade": v_1_blade,  # m/s
                             "w_1_blade": w_1_blade,  # m/s
                             "p_1_blade": p_1_blade,  # Pa
                             "T_1_blade": T_1_blade,  # K
                             "rho_1_blade": rho_1_blade,  # kg/m^3
                             "M_s1_blade": M_s1_blade,  # -
                             "v_2_blade": v_2_blade,  # m/s
                             "w_2_blade": w_2_blade,  # m/s
                             "p_2_blade": p_2_blade,  # Pa
                             "T_2_blade": T_2_blade,  # K
                             "rho_2_blade": rho_2_blade,  # kg/m^3
                             "M_r1_blade": M_r1_blade,  # -
                             "M_r2_blade": M_r2_blade,  # -
                             "alpha_2_blade": alpha_2_blade,  # rad
                             "psi_blade": psi_blade,  # -
                             "R_h_blade": R_h_blade,  # -
                             "p_0_over_p_2_blade": p_0_over_p_2_blade,  # -
                             "Re_s1_blade": Re_s1_blade,  # -
                             "Re_r2_blade": Re_r2_blade,  # -
                             }
        return blade_row_results, residual


    # def assign_geometry(self):
    #     #TODO Create a function to manually assign all geometry
    #     ...
    #
    # def analyse_turbine(self):
    #     # TODO Create a function to analyze the turbine off-desing
    #     ...
