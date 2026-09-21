import numpy as np
from scipy.optimize import root_scalar, root

class Turbine1D:
    def __init__(self):
        """A class representing 1D, nonisentropic turbine. Geometry can be either sized based on given requirements or
         assigned directly. The analysis of the turbine can be performed off-design too. This class was created with
          supersonic turbines in mind, but can be also used for subsonic ones. It takes into account nonisentropic
           processes when calculating fluid states and flow properties."""

        # Nomenclature:
        # Total properties have _t prefix. Static properties are assumed by default, they do not have any prefix.
        # Properties in stationary and rotary reference frame have _s and _r prefixes respectively.
        # Real properties are assumed by default, while ideal/isentropic ones have _ideal prefix.
        # Station 0 is station before stator, station 1 is after stator and station 2 is after rotor.
        # _b specifies that the results were calculated only for the blade row and its passage aerodynamic losses,
        # without an inclusion of additional losses like clearances, disk friction or partial admission. In other words,
        # _b represent velocities or thermodynamic properties representative of flow in stationary test cascade.

        # Assumptions:
        # 1. This class utilizes ideal gas model for calculations.
        # 2. Velocity at station 0 is assumed to be negligable.
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
        self.c_stator = None # Stator chord, m
        self.p_stator = None # Stator pitch at Euler radius, m
        self.c_rotor = None # Rotor chord, m
        self.p_rotor = None # Rotor pitch at Euler radius, m
        self.t_TE_stator = None # Stator trailing edge thickness at Euler radius, m.
        self.t_TE_rotor = None # Rotor trailing edge thickness, m.
        self.alpha_1_metal_deg = None  # Metal outlet angle of the stator blades/nozzle, deg
        self.beta_1_metal_deg = None  # Metal inlet angle of the rotor blades, deg
        self.beta_2_metal_deg = None  # Metal outlet angle of the rotor blades, deg
        self.no_blades_rotor = None # Number of blades in the rotor, -
        self.no_blades_stator = None # Number of nozzles/blades in the stator, -
        self.admission_fraction = None # Admission fraction of the turbine stage, -
        # Analysis results at the design point
        self.analysis_results_at_design_point = {"gas": None, # IdealGas object
                                                 "mdot_total": None, # Total massflow through the turbine, kg/s
                                                 "RPM": None, # Rotations Per Minute, -
                                                 "omega": None, # Angular velocity, rad/s
                                                 "T_0": None,   # Total/static temperature at station 0, K
                                                 "p_0": None,   # Total/static pressure at station 0, Pa
                                                 "loss_model": None,    # LossModel object used for calculations, -
                                                 "h_0": None,   # Total/static enthalpy at station 0, J/kg
                                                 "psi": None, # Real loading coefficient, -
                                                 "psi_ideal": None, # Ideal loading coefficient, -
                                                 "theta_1": None, # Flow coefficient at station 1, -
                                                 "theta_2": None, # Flow coefficient at station 2, -
                                                 "R_h_ideal": None, # Ideal enthalpic reaction of the stage, -
                                                 "R_h": None,   # Real enthalpic reaction of the stage, -
                                                 "R_p": None,   # Real pressure reaction of the stage, -
                                                 "p_0_over_p_2": None,  # Pressure ratio of the stage, -
                                                 "u": None, # Blade velocity at Euler radius, u/s
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
                                                 "M_s1": None, # Mach number in stationary reference frame at station 1, -
                                                 "M_s1_ideal": None, # Mach number in stationary reference frame
                                                 # assuming ideal expansion at station 1, -
                                                 "M_r1": None, # Mach number in rotary reference frame at station 1, -
                                                 "p_2": None, # Static pressure at station 2, Pa
                                                 "T_2": None, # Static temperature at station 2, K
                                                 "rho_2": None, # Static density at station 2, kg/m3
                                                 "T_2_ideal": None, # Static temperature assuming ideal expansion at
                                                 # station 2, K
                                                 "h_2": None, # Static enthalpy at station 2, J/kg
                                                 "h_t2": None, # Total enthalpy at station 2, J/kg
                                                 "h_2_ideal": None, # Static enthalpy assuming ideal expansion at
                                                 # station 2, J/kg
                                                 "delta_s_rotor": None, # Total entropy increase at rotor, J/kg/K
                                                 "delta_s_rotor_additional": None,  # Total entropy increase at rotor
                                                 # due to additional losses like clearances, disk friction or partial
                                                 # admission, J/kg/K
                                                 "a_2": None, # Sound velocity at station 2, m/s
                                                 "a_2_ideal": None, # Sound velocity assuming ideal expansion at
                                                 # station 2, m/s
                                                 "v_2": None, # Absolute velocity at station 2, m/s
                                                 "v_2_ideal": None, # Absolute velocity assuming ideal expansion at
                                                 # station 2, m/s
                                                 "w_2": None, # Relative velocity at station 2, m/s
                                                 "w_2_ideal": None, # Relative velocity assuming ideal expansion at
                                                 # station 2, m/s
                                                 "M_s2": None, # Mach number in stationary reference frame at station 2, -
                                                 "M_s2_ideal": None, # Mach number in stationary reference frame
                                                 # assuming ideal expansion at station 2, -
                                                 "M_r2": None, # Mach number in rotary reference frame at station 2, -
                                                 "M_r2_ideal": None, # Mach number in rotary reference frame assuming
                                                 # ideal expansion at station 2, -
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
            "v_2_blade": None,  # Absolute velocity at station 2 for the blade row alone, m/s
            "w_2_blade": None,  # Relative velocity at station 2 for the blade row alone, m/s
            "p_2_blade": None,  # Static pressure at station 2 for the blade row alone, Pa
            "T_2_blade": None,  # Static temperature at station 2 for the blade row alone, K
            "rho_2_blade": None,  # Static density at station 2 for the blade row alone, kg/m3
            "alpha_2_blade": None,  # Absolute flow angle at station 2 for the blade row alone, rad
            "psi_blade": None,  # Loading coefficient for the blade row alone, -
            "R_h_blade": None,  # Enthalpic reaction for the blade row alone, -
            "p_0_over_p_2_blade": None,  # Pressure ratio for the blade row alone, -
        }

    def size_turbine(self, gas, loading_coefficient, flow_coefficient, reaction_isentropic, pressure_ratio, RPM,
                     shaft_power, mdot, T_0, p_0, radial_clearance, no_blades_stator, no_blades_rotor,
                     chord_over_pitch_stator, t_TE_stator, t_TE_rotor,
                     loss_model, admission_fraction=1, chord_over_pitch_rotor=2.5, s_ax_over_pitch_rotor=0.35,
                     delta_s_estimate=[0, 0, 0], shrouded_rotor=False, h_shroud_over_blade_length=0.008,
                     s_ax_shroud_over_h_shroud=2, t_shroud=None):
        """A method to size the turbine based on given requirements. It changes properties of the object.

        :param IdealGas gas: IdealGas object representing working gas of the turbine.
        :param float or integer loading_coefficient: Real, nonisentropic loading coefficient of the stage.
        :param float or integer flow_coefficient: Flow coefficient of the turbine at station 2.
        :param float reaction_isentropic: Enthalpic, isentropic reaction of the turbine. It should be noted enthalpic
            reaction is not the same as pressure reaction, although two quantities are similar.
        :param float or integer pressure_ratio: Pressure ratio across the turbine; ratio of p_t0 to p_s2.
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
        :param float or integer t_TE_stator: Thickness (m) of the trailing edge for the stator.
        :param float or integer t_TE_rotor: Thickness (m) of the trailing edge for the rotor.
        :param float or integer s_ax_over_pitch_rotor: Axial distance between stator and rotor over rotor pitch (-).
         By default, 0.35, which is in the range of 0.3-0.35 given by Traupel.
        :param boolean shrouded_rotor: Boolean whether the rotor is shrouded. By default, False.
        :param float h_shroud_over_blade_length: Overlap of the blade with the casing over rotor blade length.
         By default, 0.008, which is below the value of 0.01 recommended by Traupel.
        :param float or integer s_ax_shroud_over_h_shroud: Axial distance between shroud and the casing over blade
         overlap with it. By default, 2, which is above the value of 1.5 recommended by Traupel.
        :param float or integer t_shroud: Shroud thickness, m. By default, None.
        :param loss_model: LossModel object to be used in the analysis.
        :param list delta_s_estimate: List of initial estimates of the entropy rises (J/kg/K) due to stator, rotor and
         additional rotor losses. Additional rotor losses represent clearance, partial admission or disk friction
          losses. By default, [0, 0, 0].
        """

        # Calculate the blade speed from given requirements
        delta_h = shaft_power / mdot
        u = np.sqrt(delta_h / loading_coefficient)

        # Calculate angular speed and Euler diameter of the turbine
        omega = RPM * 2 * np.pi / 60
        self.D_Euler = 2 * u / omega
        self.R_Euler = self.D_Euler / 2

        # Calculate available geometry already
        self.c_stator = self.D_Euler * np.pi / no_blades_stator
        self.c_rotor = self.D_Euler * np.pi / no_blades_rotor
        self.p_stator = self.c_stator / chord_over_pitch_stator
        self.p_rotor = self.c_rotor / chord_over_pitch_rotor
        self.s_ax = self.p_rotor * s_ax_over_pitch_rotor

        # Assign remaining geometry that is already known
        self.t_TE_rotor = t_TE_rotor
        self.t_TE_stator = t_TE_stator
        self.admission_fraction = admission_fraction
        self.s_r = radial_clearance
        self.no_blades_rotor = no_blades_rotor
        self.no_blades_stator = no_blades_stator
        self.shrouded_rotor = shrouded_rotor
        self.t_shroud = t_shroud
        self.h_shroud_over_blade_length = h_shroud_over_blade_length
        self.s_ax_shroud_over_h_shroud = s_ax_shroud_over_h_shroud

        # Put design variables in analysis_results_at_design_point already
        self.analysis_results_at_design_point.update({"gas": gas,
                                                      "psi": loading_coefficient,
                                                      "theta_2": flow_coefficient,
                                                      "R_h_ideal": reaction_isentropic,
                                                      "p_0_over_p_2": pressure_ratio,
                                                      "RPM": RPM,
                                                      "omega": omega,
                                                      "u": u,
                                                      "P_shaft": shaft_power,
                                                      "mdot_total": mdot,
                                                      "T_0": T_0,
                                                      "p_0": p_0,
                                                      "loss_model": loss_model.name})


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
        entropy_solution = root(get_entropy_residual, delta_s_estimate, method="hybr",
                                options={"xtol":1e-6, "eps": 1e-12, "maxfev": 1000, "factor": 1,
                                         "diag": 1/entropy_scales})
        # Raise an error if not converged
        if not entropy_solution.success:
            raise RuntimeError("Numerical solve for the stator and rotor entropy rises did not converge.")
        # Get entropy increases and the rest of the results
        delta_s_stator, delta_s_rotor, delta_s_rotor_additional = entropy_solution.x
        _, analysis_results, blade_row_results = \
            self.calculate_entropy_rise(delta_s_stator, delta_s_rotor, delta_s_rotor_additional, loss_model)


        # Update analysis_results_at_design_point and blade row results
        self.analysis_results_at_design_point = analysis_results
        self.blade_row_results_at_design_point = blade_row_results

        # Get flow angles, which become metal angles
        self.alpha_1_metal_deg = self.analysis_results_at_design_point["alpha_1"]
        self.beta_1_metal_deg = self.analysis_results_at_design_point["beta_1"]
        self.beta_2_metal_deg = self.analysis_results_at_design_point["alpha_2"]

        # Calculate and assign remaining properties related to geometry
        self.D_hub, self.D_tip, self.D_mean, self.l_rotor, self.h_shroud, self.s_ax_shroud =\
            self.__calculate_geometry(self.analysis_results_at_design_point)
        self.l_stator = self.l_rotor
        self.R_mean = self.D_mean / 2
        self.R_hub = self.D_hub / 2
        self.R_tip = self.D_tip / 2

    def calculate_entropy_rise(self, delta_s_stator, delta_s_rotor, delta_s_rotor_additional, loss_model):
        """A method to calculate residuals between the assumed entropy rises and the entropy rises returned by the loss
        model, together with the turbine analysis results.

        :param float delta_s_stator: Specific entropy rise due to aerodynamic losses in the stator (J/kg/K).
        :param float delta_s_rotor: Specific entropy rise due to aerodynamic losses in the rotor (J/kg/K).
        :param float delta_s_rotor_additional: Specific entropy rise due to additional rotor losses such as clearance,
            partial admission or disk friction losses (J/(kg K)).
        :param loss_model: LossModel object to be used to calculate the entropy rises.
        :return: Entropy-rise residuals, turbine analysis results and blade row analysis results, respectively.
        :rtype: tuple[list, dict, dict]
        """

        # In rocket turbines, there are no clearance losses in stators, while partial admission losses belong to the
        # rotor. Therefore, entropy generation in the stator due to aerodynamic losses constitute total entropy
        # generation through the rotor.
        delta_s_stator_total = delta_s_stator
        # In the rotor, aside from aerodynamic losses, there are additional losses like clearance, partial admission
        # or disk friction losses. Total entropy generation is thus the sum of all of these.
        delta_s_rotor_total = delta_s_rotor + delta_s_rotor_additional


        # Thermodynamic properties at each station must be calculated. Thermodynamic properties depend on rotational
        # Mach number M_u, which itself depends on thermodynamic properties. The model is thus implicit and requires
        # numerical solve.
        # First, retrieve some variables from analysis_results_at_design_point
        gas = self.analysis_results_at_design_point["gas"]
        p_0_over_p_2 = self.analysis_results_at_design_point["p_0_over_p_2"]
        psi = self.analysis_results_at_design_point["psi"]
        theta_2 = self.analysis_results_at_design_point["theta_2"]
        R_h_ideal = self.analysis_results_at_design_point["R_h_ideal"]
        # Now, calculate maximum possible M_u
        delta_s_total = delta_s_stator_total + delta_s_rotor_total
        h_0_over_h_2 = p_0_over_p_2**((gas.gamma - 1) / gas.gamma) * \
                       np.exp(-delta_s_total * (gas.gamma - 1) / (gas.gamma * gas.R))
        M_u_max = np.sqrt((h_0_over_h_2 - 1) / (psi * (gas.gamma - 1)))
        # Calculate also initial estimate (approximately analytical solution for loss free calculations and
        # constant flow coefficient)
        v_2_over_u_2_estimate = np.sqrt(theta_2**2 + (1 - R_h_ideal - psi / 2)**2)
        M_u_estimate = M_u_max * np.sqrt(2 * psi / (h_0_over_h_2 * v_2_over_u_2_estimate + 2 * psi))
        # First, bracketing scheme will be attempted. If bracket does not return opposite signs,
        # Newton's method is used.
        # Create a bracket
        bracket = [1e-3 * M_u_max, (1 - 1e-3) * M_u_max]
        # Define a function that obtains residual
        def get_M_u_residual(M_u):
            return self.__calculate_thermodynamic_properties(M_u, delta_s_stator_total, delta_s_rotor_total,
                                                             delta_s_rotor_additional)[1]
        # Calculate residuals for the bracket
        residual_at_bracket = [get_M_u_residual(M_u) for M_u in bracket]
        # If it returns opposite signs, use toms748 bracketing scheme
        if residual_at_bracket[0] * residual_at_bracket[1] < 0:
            M_u_solution = root_scalar(get_M_u_residual, bracket=bracket, method="toms748", options={"k": 2},
                                       maxiter=1000, xtol=1e-10, rtol=1e-8)
            # If the solution does not converge, attempt Newton scheme
            if not M_u_solution.converged:
                M_u_solution = root_scalar(get_M_u_residual, x0=M_u_estimate, method="newton",
                                           maxiter=1000, xtol=1e-10, rtol=1e-8)
        # If the bracket does not return opposite results, also use Newton scheme
        else:
            M_u_solution = root_scalar(get_M_u_residual, x0=M_u_estimate, method="newton",
                                       maxiter=1000, xtol=1e-10, rtol=1e-8)
        # Raise error if the solution is not converged
        if not M_u_solution.converged:
            raise RuntimeError("Numerical solve for M_u did not converge.")
        # Obtain solution and the remaining results
        M_u = M_u_solution.root
        analysis_results, residual = self.__calculate_thermodynamic_properties(
            M_u, delta_s_stator_total, delta_s_rotor_total, delta_s_rotor_additional)


        # Flow angles can be now calculated. These are equal to metal angles. These are affected by the total entropy
        # generation in the rotor, as it affects theta_1 through rho_2.
        alpha_1, beta_1, alpha_2, beta_2 = self.__calculate_flow_angles(analysis_results)
        # Append analysis results with these flow angles
        analysis_results.update({"alpha_1": alpha_1,
                                 "alpha_2": alpha_2,
                                 "beta_1": beta_1,
                                 "beta_2": beta_2,
                                 "alpha_1_deg": alpha_1 * 180 / np.pi,
                                 "alpha_2_deg": alpha_2 * 180 / np.pi,
                                 "beta_1_deg": beta_1 * 180 / np.pi,
                                 "beta_2_deg": beta_2 * 180 / np.pi,
                                 "incidence": 0,
                                 "incidence_deg": 0})


        # Recalculate velocities for the rotor and its calculated angles, but without additional losses like friction,
        # clearance or partial admission, as the loss model may require separate blade row velocities. Such velocity
        # only takes into account aerodynamic losses / friction losses in the blade passage itself and is thus
        # representative of stationary test cascades. To reconstruct these blade row results, theta_2_blade is needed.
        # This is theta_2 for the blade angles above, if delta_s_rotor_additional was zero. However, it is also an
        # outcome of these calculations. Thus, the model is implicit and a numerical solve is again needed.
        # Bracketing scheme will be used, so first theta_2_max is estimated. This is theta_2 as T_2 approaches zero.
        h_1 = analysis_results["h_1"]
        w_1 = analysis_results["w_1"]
        u = analysis_results["u"]
        gas = analysis_results["gas"]
        theta_2_max = np.sqrt(2 * h_1 + w_1**2) / (u * np.sqrt(1 + np.tan(beta_2)**2))
        # The bracket used can be supersonic or subsonic. theta_2_crit divides these two ranges. Whichever bracket is
        # used depends on the value of original theta_2.
        theta_2_crit = theta_2_max * np.sqrt((gas.gamma - 1) / (gas.gamma + 1))
        subsonic_branch = [1e-3 * theta_2_crit, theta_2_crit]
        supersonic_branch = [theta_2_crit, (1 - 1e-3) * theta_2_max]
        branch = subsonic_branch if theta_2 < theta_2_crit else supersonic_branch
        # Define residual function to solve
        def get_theta_2_blade_residual(theta_2_blade):
            _, residual = self.__calculate_blade_row_velocities(
                alpha_1, beta_1, beta_2, theta_2_blade, analysis_results)
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
        theta_2_blade = theta_2_blade_solution.root
        blade_row_results = self.__calculate_blade_row_velocities(alpha_1, beta_1, beta_2, theta_2_blade,
                                                                  analysis_results)

        # Some additional geometry must be calculated for the loss model. Pack to dictionary to pass to the loss model.
        D_hub, D_tip, D_mean, l_rotor, h_shroud, s_ax_shroud = self.__calculate_geometry(analysis_results)
        l_stator = l_rotor
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
                            "admission_fraction": self.admission_fraction}  # -




        # Call loss model, calculate entropy rise and loss coefficients.
        delta_s_stator_output, delta_s_rotor_output, delta_s_rotor_additional_output =\
            loss_model.calculate_entropy_increase(analysis_results, turbine_geometry)


        # Calculate residual
        residual = [delta_s_stator_output - delta_s_stator, delta_s_rotor_output - delta_s_rotor,
                    delta_s_rotor_additional_output - delta_s_rotor_additional]


        # Return all calculated results
        return residual, analysis_results, blade_row_results

    def __calculate_geometry(self, analysis_results):
        """A method to calculate some of the turbine geometry."""

        # First obtain design blade velocity and theta_2.
        u = analysis_results["u"]
        theta_2 = analysis_results["theta_2"]

        # Calculate axial velocity.
        v_ax = theta_2 * u

        # From massflow, density and axial velocity, calculate annulus area.
        mdot = analysis_results["mdot"]
        rho_2 = analysis_results["rho_2"]
        area = mdot / (v_ax * rho_2 * self.admission_fraction)

        # From annulus area, calculate blade length.
        D_hub = np.sqrt(self.D_Euler**2 - (2 * area / np.pi))
        D_tip = np.sqrt(self.D_Euler**2 + (2 * area / np.pi))
        D_mean = (D_hub + D_tip) / 2
        l_blade = (D_tip - D_hub) / 2

        # If shroud is used, calculate remaining variables. Otherwise, these are None.
        if self.shrouded_rotor:
            h_shroud = self.h_shroud_over_blade_length * l_blade
            s_ax_shroud = self.s_ax_shroud_over_h_shroud * h_shroud
        else:
            h_shroud = None
            s_ax_shroud = None

        # Return the results
        return D_hub, D_tip, D_mean, l_blade, h_shroud, s_ax_shroud

    def __calculate_thermodynamic_properties(self, M_u, total_delta_s_stator, total_delta_s_rotor,
                                             delta_s_rotor_additional):
        """A method to calculate thermodynamic properties, velocities and Mach numbers at turbine stations 1 and 2.

        :param float M_u: Mach number based on blade velocity at the Euler radius.
        :param float total_delta_s_stator: Total specific entropy rise across the stator (J/kg/K).
        :param float total_delta_s_rotor: Total specific entropy rise across the rotor (J/kg/K).
        :param float delta_s_rotor_additional: Specific entropy rise due to clearance, partial admission and disk
            friction losses in the rotor (J/kg/K).
        :return: Dictionary with turbine analysis results and residual of the rotational Mach number equation,
            respectively.
        :rtype: tuple[dict, float]
        """

        # Retrieve already known variables from analysis_results_at_design_point
        psi = self.analysis_results_at_design_point["psi"]
        R_h_ideal = self.analysis_results_at_design_point["R_h_ideal"]
        theta_2 = self.analysis_results_at_design_point["theta_2"]
        p_0_over_p_2 = self.analysis_results_at_design_point["p_0_over_p_2"]
        T_0 = self.analysis_results_at_design_point["T_0"]
        p_0 = self.analysis_results_at_design_point["p_0"]
        gas = self.analysis_results_at_design_point["gas"]
        u = self.analysis_results_at_design_point["u"]

        # Calculate total entropy increase
        delta_s = total_delta_s_rotor + total_delta_s_stator
        # Calculate h_0
        h_0 = gas.Cp * T_0
        # Calculate h_0_over_h_2
        h_0_over_h_2 = p_0_over_p_2**((gas.gamma - 1)/ gas.gamma) \
                       * np.exp(-delta_s * (gas.gamma - 1) / (gas.gamma * gas.R))
        # Calculate h_t0_over_h_t2
        h_t0_over_h_t2 = psi * (gas.gamma - 1) * M_u**2 + 1
        # Calculate h_t2_over_h_2
        h_t2_over_h_2 = h_0_over_h_2 / h_t0_over_h_t2
        # Calculate h_1_over_h_2_isentropic
        h_1_over_h_2_isentropic = R_h_ideal * (h_t0_over_h_t2 - 1) * h_t2_over_h_2 + 1
        # Calculate h_1_over_h_2
        h_1_over_h_2 = h_1_over_h_2_isentropic * np.exp(-total_delta_s_rotor * (gas.gamma - 1) / (gas.R * gas.gamma))
        # Calculate nonisentropic degree of reaction, R_h
        R_h = ((h_1_over_h_2 - 1) / (h_t0_over_h_t2 - 1)) / h_1_over_h_2_isentropic

        # Calculate p_2, T_2, rho_2, h_2, h_t2
        p_2 = p_0 / p_0_over_p_2
        h_2 = h_0 / h_0_over_h_2
        T_2 = h_2 / gas.Cp
        rho_2 = gas.calculate_density(p_2, T_2)
        h_t2 = h_0 / h_t0_over_h_t2

        # Calculate p_1, T_1, rho_1, h_1, h_t1
        h_1 = h_2 * h_1_over_h_2
        T_1 = h_1 / gas.Cp
        p_1 = p_2 * np.exp(total_delta_s_rotor / gas.R) * h_1_over_h_2**(gas.gamma / (gas.gamma - 1))
        rho_1 = gas.calculate_density(p_1, T_1)
        # Energy is conserved between station 0 and station 1, so:
        h_t1 = h_0

        # Calculate theta_1
        theta_1 = theta_2 * rho_2 / rho_1

        # Calculate velocity_ratio_squared
        v_2_over_u = np.sqrt(theta_2**2 + (1 - R_h - psi / 2 + (theta_2**2 - theta_1**2) / (2 * psi))**2)

        # Calculate residual that must be zero
        dummy_1 = M_u**2 * h_t2_over_h_2 * v_2_over_u**2
        dummy_2 = (2 / (gas.gamma - 1)) * (h_t2_over_h_2 - 1)
        residual = dummy_1 - dummy_2

        # Calculate remaining values of interest.
        # First calculate ideal psi
        h_t2_over_h_t0 = 1 / h_t0_over_h_t2
        psi_ideal = psi * (1 - h_t2_over_h_t0 * np.exp(-delta_s * (gas.gamma - 1) / (gas.gamma * gas.R))) \
                    / (1 - h_t2_over_h_t0)

        # Now velocity and Mach number at station 1. First calculate velocity of sound there:
        a_1 = gas.calculate_sound_velocity(T_1)
        h_1_ideal = h_0 * (p_1 / p_0) ** ((gas.gamma - 1) / gas.gamma)
        T_1_ideal = h_1_ideal / gas.Cp
        a_1_ideal = gas.calculate_sound_velocity(T_1_ideal)
        # Real and ideal velocity and Mach number in stationary reference frame:
        v_1 = np.sqrt(2 * (h_0 - h_1))
        delta_h_loss_stator = h_1 - h_1_ideal
        v_1_ideal = np.sqrt(v_1**2 + 2 * delta_h_loss_stator)
        M_s1 = v_1 / a_1
        M_s1_ideal = v_1_ideal / a_1_ideal
        # Now real values in rotary reference frame:
        dummy_a1 = 1 - R_h + psi / 2 + (theta_2**2 - theta_1**2) / (2 * psi)
        dummy_b1 = dummy_a1 - 1
        w_1 = u * np.sqrt(theta_1**2 + dummy_b1**2)
        M_r1 = w_1 / a_1

        # Now velocity and Mach number at station 2. Ideal values at station 2
        # still assume real values at station 1. First calculate velocity of sound at station 2.
        a_2 = gas.calculate_sound_velocity(T_2)
        h_2_ideal = h_1 * (p_2 / p_1) ** ((gas.gamma - 1) / gas.gamma)
        T_2_ideal = h_2_ideal / gas.Cp
        a_2_ideal = gas.calculate_sound_velocity(T_2_ideal)
        # Real values in stationary reference frame:
        v_2 = v_2_over_u * u
        M_s2 = v_2 / a_2
        # Ideal values in stationary reference frame:
        v_2_ideal = np.sqrt(2*(h_t1 - psi_ideal * u**2 - h_2_ideal))
        M_s2_ideal = v_2_ideal / a_2_ideal
        # Now values in rotary reference frame:
        dummy_a2 = dummy_a1 - psi
        dummy_b2 = dummy_a2 - 1
        w_2 = u * np.sqrt(theta_2**2 + dummy_b2**2)
        delta_h_loss_rotor = h_2 - h_2_ideal
        w_2_ideal = np.sqrt(w_2**2 + 2 * delta_h_loss_rotor)
        M_r2 = w_2 / a_2
        M_r2_ideal = w_2_ideal / a_2_ideal

        # Calculate real pressure reaction, R_p
        R_p = (p_1 - p_2) / (p_0 - p_2)

        # Package thermodynamic properties, velocities and loading coefficients into analysis_results dictionary.
        # Make it a copy of analysis_results_at_design_point, such that the values there stay constant during
        # iterations.
        analysis_results = self.analysis_results_at_design_point.copy()
        analysis_results.update({"h_0": h_0,
                                 "p_1": p_1,
                                 "T_1": T_1,
                                 "rho_1": rho_1,
                                 "T_1_ideal": T_1_ideal,
                                 "h_1": h_1,
                                 "h_t1": h_t1,
                                 "h_1_ideal": h_1_ideal,
                                 "delta_s_stator": total_delta_s_stator,
                                 "a_1": a_1,
                                 "a_1_ideal": a_1_ideal,
                                 "v_1": v_1,
                                 "v_1_ideal": v_1_ideal,
                                 "w_1": w_1,
                                 "M_s1": M_s1,
                                 "M_s1_ideal": M_s1_ideal,
                                 "M_r1": M_r1,
                                 "p_2": p_2,
                                 "T_2": T_2,
                                 "rho_2": rho_2,
                                 "T_2_ideal": T_2_ideal,
                                 "h_2": h_2,
                                 "h_t2": h_t2,
                                 "h_2_ideal": h_2_ideal,
                                 "delta_s_rotor": total_delta_s_rotor - delta_s_rotor_additional,
                                 "delta_s_rotor_additional": delta_s_rotor_additional,
                                 "a_2": a_2,
                                 "a_2_ideal": a_2_ideal,
                                 "v_2": v_2,
                                 "v_2_ideal": v_2_ideal,
                                 "w_2": w_2,
                                 "w_2_ideal": w_2_ideal,
                                 "M_s2": M_s2,
                                 "M_s2_ideal": M_s2_ideal,
                                 "M_r2": M_r2,
                                 "M_r2_ideal": M_r2_ideal,
                                 "R_p": R_p,
                                 "psi_ideal": psi_ideal,
                                 "R_h": R_h,
                                 "theta_1": theta_1,
                                 })

        # Return residual
        return analysis_results, residual

    def __calculate_flow_angles(self, analysis_results):
        """A method to calculate absolute and relative flow angles at stations 1 and 2.

        :param dict analysis_results: Dictionary with turbine analysis results required to calculate the flow angles.
        :return: Absolute flow angle at station 1 (rad), relative flow angle at station 1 (rad),
         absolute flow angle at station 2 (rad) and relative flow angle at station 2 (rad), respectively.
        :rtype: tuple[float, float, float, float]
        """

        # Retrieve variables from analysis_results
        theta_1 = analysis_results["theta_1"]
        theta_2 = analysis_results["theta_2"]
        R_h = analysis_results["R_h"]
        psi = analysis_results["psi"]
        # Calculate support variable
        dummy_theta = (theta_2**2 - theta_1**2) / (2 * psi)
        # Calculate flow angles
        alpha_1 = np.atan2(1 - R_h + psi / 2 + dummy_theta, theta_1)
        beta_1 = np.atan2(-R_h + psi / 2 + dummy_theta, theta_1)
        beta_2 = np.atan2(-R_h - psi / 2 + dummy_theta, theta_2)
        alpha_2 = np.atan2(1 - R_h - psi / 2 + dummy_theta, theta_2)
        # Return all flow angles
        return alpha_1, beta_1, alpha_2, beta_2

    def __calculate_blade_row_velocities(self, alpha_1_metal, beta_1_metal, beta_2_metal, theta_2_blade,
                                         analysis_results):
        """A method to calculate velocities and thermodynamic properties for the blade row alone, without additional
        rotor losses such as clearance, partial admission or disk friction losses.

        :param float alpha_1_metal: Outlet metal angle of the stator blades/nozzles (rad).
        :param float beta_1_metal: Inlet metal angle of the rotor blades (rad).
        :param float beta_2_metal: Outlet metal angle of the rotor blades (rad).
        :param float theta_2_blade: Assumed flow coefficient (-) at station 2 for the blade row alone.
        :param dict analysis_results: Dictionary with turbine analysis results required to calculate the blade row
            velocities and thermodynamic properties.
        :return: Tuple containing a dictionary with blade row results and the residual between calculated and assumed
            flow coefficient at station 2.
        :rtype: tuple[dict, float]
        """

        # The inlet velocity to the rotor blade row at station 1 is unchanged, since additional rotor losses only affect
        # results at the station 2.
        gas = analysis_results["gas"]
        p_1_blade = analysis_results["p_1"]
        T_1_blade = analysis_results["T_1"]
        rho_1_blade = analysis_results["rho_1"]
        h_1_blade = analysis_results["h_1"]
        w_1_blade = analysis_results["w_1"]
        v_1_blade = analysis_results["v_1"]
        theta_1_blade = analysis_results["theta_1"]
        h_0 = analysis_results["h_0"]
        p_0 = analysis_results["p_0"]
        # Unpack entropy increase across the rotor due to aerodynamic losses
        delta_s_rotor = analysis_results["delta_s_rotor"]
        # Blade velocity is also the same
        u = analysis_results["u"]

        # Based on assumed theta_2_blade, calculate velocities for the blade row alone at station 2.
        w_2_blade = np.sqrt(u**2 * theta_2_blade**2 * (1 + np.tan(beta_2_metal)**2))
        v_2_blade = np.sqrt(u**2 * (theta_2_blade**2 + (1 + theta_2_blade * np.tan(beta_2_metal))**2))

        # From the conservation of rotalphy, get enthalpy and temperature at station 2
        h_2_blade = h_1_blade + (w_1_blade**2 - w_2_blade**2) / 2
        T_2_blade = h_2_blade / gas.Cp
        # Entropy increase across the rotor is known, so pressure at station 2 can be calculated. It will be higher
        # than pressure at station 2 for the whole stage.
        p_2_blade = p_1_blade * np.exp((gas.Cp * np.log(T_2_blade / T_1_blade) - delta_s_rotor) / gas.R)
        rho_2_blade = gas.calculate_density(p_2_blade, T_2_blade)

        # From continuity, theta_2_blade_output can be calculated. The difference between the assumed and calculated
        # values can be also calculated.
        theta_2_blade_output = rho_1_blade * theta_1_blade / rho_2_blade
        residual = theta_2_blade_output - theta_2_blade

        # Some other quantities can be also calculated for the blade row alone.
        alpha_2_blade = np.atan2(1 + theta_2_blade * np.tan(beta_2_metal), theta_2_blade_output)
        psi_blade = theta_1_blade * np.tan(alpha_1_metal) - theta_2_blade_output * np.tan(beta_2_metal) - 1
        R_h_blade = (h_1_blade - h_2_blade) / (h_0 - h_2_blade)
        p_0_over_p_2_blade = p_0 / p_2_blade

        # Now pack the results and return them together with residual
        blade_row_results = {"v_1_blade": v_1_blade,
                             "w_1_blade": w_1_blade,
                             "p_1_blade": p_1_blade,
                             "T_1_blade": T_1_blade,
                             "rho_1_blade": rho_1_blade,
                             "v_2_blade": v_2_blade,
                             "w_2_blade": w_2_blade,
                             "p_2_blade": p_2_blade,
                             "T_2_blade": T_2_blade,
                             "rho_2_blade": rho_2_blade,
                             "alpha_2_blade": alpha_2_blade,
                             "psi_blade": psi_blade,
                             "R_h_blade": R_h_blade,
                             "p_0_over_p_2_blade": p_0_over_p_2_blade}
        return blade_row_results, residual


    # def assign_geometry(self):
    #     #TODO Create a function to manually assign all geometry
    #     ...
    #
    # def analyse_turbine(self):
    #     # TODO Create a function to analyze the turbine off-desing
    #     ...
