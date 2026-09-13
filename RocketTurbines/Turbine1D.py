import numpy as np
from scipy.optimize import root_scalar, root

class Turbine1D:
    def __init__(self):
        """A class representing 1D, nonisentropic turbine. Geometry can be either sized based on given requirements or
         assigned directly. The analysis of the turbine can be performed off-design too. This class was created with
          supersonic turbines in mind, but can be also used for subsonic ones. It takes into account nonisentropic
           processes when calculating fluid states and flow properties."""

        #TODO: Add nomenclature

        # Geometry stored in the object properties
        self.D_hub = None
        self.D_tip = None
        self.R_hub = None
        self.R_tip = None
        self.R_mean = None
        self.R_Euler = None
        self.l_blade = None
        self.s_ax = None
        self.s_r = None
        self.alfa_1 = None
        self.alfa_2 = None
        self.beta_1 = None
        self.beta_2 = None
        self.no_blades_rotor = None
        self.no_blades_stator = None
        self.admission_fraction = None
        # Analysis results at the design point
        self.analysis_results_at_design_point = {}

    def size_turbine(self, gas, loading_coefficient, flow_coefficient, reaction_isentropic, pressure_ratio, RPM,
                     shaft_power, mdot, T_0, p_0, radial_clearance, no_blades_stator, no_blades_rotor,
                     admission_fraction, loss_model, delta_s_estimate=[0, 0]):
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
        :param float or integer radial_clearance: Radial clearance of the rotor blade.
        :param integer no_blades_stator: Number of stator blades/nozzles.
        :param integer no_blades_rotor: Number of rotor blades.
        :param float admission_fraction: Admission fraction of the turbine stage.
        :param loss_model: Loss model to be used in the analysis.
        :param list delta_s_estimate: Initial estimates of the stator and rotor entropy rises, respectively.
        """

        # Calculate the blade speed from given requirements
        delta_h = shaft_power / mdot
        u = np.sqrt(delta_h / loading_coefficient)

        # Calculate angular speed and Euler diameter of the turbine
        omega = RPM * 2 * np.pi / 60
        D_Euler = 2 * u / omega

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
            return self.calculate_entropy_rise(delta_s[0], delta_s[1], loss_model)[0]
        # Solve for entropy increase
        entropy_solution = root(get_entropy_residual, delta_s_estimate, method="hybr")
        # Raise an error if not converged
        if not entropy_solution.success:
            raise RuntimeError("Numerical solve for the stator and rotor entropy rises did not converge.")
        # Get entropy increases and the rest of the results
        delta_s_stator, delta_s_rotor = entropy_solution.x
        _, analysis_results = self.calculate_entropy_rise(delta_s_stator, delta_s_rotor, loss_model)

        # Update analysis_results_at_design_point

        # Get flow angles, which become metal angles

        # Calculate and assign object properties related to geometry

        ...

    def calculate_entropy_rise(self, delta_s_stator, delta_s_rotor, loss_model):
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
        delta_s = delta_s_rotor + delta_s_stator
        h_0_over_h_2 = p_0_over_p_2**((gas.gamma - 1) / gas.gamma) * \
                       np.exp(-delta_s * (gas.gamma - 1) / (gas.gamma * gas.R))
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
            return self.__calculate_thermodynamic_properties(M_u, delta_s_stator, delta_s_rotor)[1]
        # Calculate residuals for the bracket
        residual_at_bracket = [get_M_u_residual(M_u) for M_u in bracket]
        # If it returns opposite signs, use toms748 bracketing scheme
        if residual_at_bracket[0] * residual_at_bracket[1] < 0:
            M_u_solution = root_scalar(get_M_u_residual, bracket=bracket, method="toms748")
        # Otherwise, use Newton scheme
        else:
            M_u_solution = root_scalar(get_M_u_residual, x0=M_u_estimate, method="newton")
        # Raise error if the solution is not converged
        if not M_u_solution.converged:
            raise RuntimeError("Numerical solve for M_u did not converge.")
        # Obtain solution and the remaining results
        M_u = M_u_solution.root
        analysis_results, residual = self.__calculate_thermodynamic_properties(
            M_u, delta_s_stator, delta_s_rotor)

        # Flow angles can be now calculated.
        alpha_1, beta_1, alpha_2, beta_2 = self.__calculate_flow_angles(analysis_results)

        # Call loss model, calculate entropy rise and loss coefficients.
        delta_s_stator_output, delta_s_rotor_output = loss_model.calculate_entropy_increase(...)

        # Calculate residual
        residual = [delta_s_stator_output - delta_s_stator, delta_s_rotor_output - delta_s_rotor]

        # Return all calculated results
        return residual, analysis_results

    def __calculate_thermodynamic_properties(self, M_u, delta_s_stator, delta_s_rotor):
        # It is assumed that velocity at station 0 is negligable, therefore static conditions there are approximately
        # total ones.

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
        delta_s = delta_s_rotor + delta_s_stator
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
        h_1_over_h_2 = h_1_over_h_2_isentropic * np.exp(-delta_s_rotor * (gas.gamma - 1) / (gas.R * gas.gamma))
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
        p_1 = p_2 * np.exp(delta_s_rotor / gas.R) * h_1_over_h_2**(gas.gamma / (gas.gamma - 1))
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
        v_1_ideal = np.sqrt(v_1**2 + 2 * delta_s_stator)
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
        w_2_ideal = np.sqrt(w_2**2 + 2 * delta_s_rotor)
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
                                 "delta_s_stator": delta_s_stator,
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
                                 "delta_s_rotor": delta_s_rotor,
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
        return alpha_1, beta_1, beta_2, alpha_2

    def assign_geometry(self):
        ...

    def analyse_turbine(self):
        ...
