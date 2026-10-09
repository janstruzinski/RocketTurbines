import warnings
import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

class TraupelLossModel:
    def __init__(self, extrapolation_method, warn_on_extrapolation=True, incidence_loss="medium",
                 stator="regular", rotor="regular"):
        """A class to calculate turbine losses with Traupel meanline loss model presented in
         "Thermische Turbomaschinen", which is very suitable for steam, supersonic turbines. All graphs are digitalized
          as ready to use interpolators in class properties. Class methods allow to calculate specific loss
           coefficients. Aerodynamic blade-row loss coefficients are defined as dissipated enthalpy over the
           corresponding ideal outlet kinetic energy in the absolute frame for the stator or relative frame for
           the rotor. Additional stage-loss coefficients are defined as dissipated enthalpy over the static-to-static
           isentropic enthalpy drop of the stage.

        :param str extrapolation_method: Method used outside the digitized data region, either "linear" or "closest".
            The latter returns the value at the closest point in the digitized region. "closest" was found to be more
            accurate for conventional designs.
        :param bool warn_on_extrapolation: Whether to warn on each out-of-bounds interpolation call.
        :param str incidence_loss: Incidence-loss level: "high" uses curve a, "low" uses curve b and "medium"
            averages the two curves. By default, "medium".
        :param str stator: Stator Mach-correction curve, either "regular" or "near_sonic_outlet". By default, "regular".
        :param str rotor: Rotor Mach-correction curve, either "regular", "near_sonic_outlet", "impulse_low_M" or
            "impulse_high_M". By default, "regular".
        """

        # Use linear interpolation throughout and check the selected extrapolation option.
        interpolation_method = "slinear"
        extrapolation_method = extrapolation_method.lower()
        if extrapolation_method not in ("linear", "closest"):
            raise ValueError("extrapolation_method must be either 'linear' or 'closest'.")
        # Select the incidence-loss curve once for all later z calculations.
        if not isinstance(incidence_loss, str) or incidence_loss.lower() not in ("high", "medium", "low"):
            raise ValueError("incidence_loss must be 'high', 'medium' or 'low'.")
        # The stator uses only the two accelerating-cascade curves; the rotor can also use either impulse curve.
        if not isinstance(stator, str) or stator not in ("regular", "near_sonic_outlet"):
            raise ValueError("stator must be 'regular' or 'near_sonic_outlet'.")
        if not isinstance(rotor, str) or rotor not in ("regular", "near_sonic_outlet", "impulse_low_M",
                                                      "impulse_high_M"):
            raise ValueError("rotor must be 'regular', 'near_sonic_outlet', 'impulse_low_M' or 'impulse_high_M'.")
        self.interpolation_method = interpolation_method
        self.extrapolation_method = extrapolation_method
        self.warn_on_extrapolation = warn_on_extrapolation
        self.incidence_loss = incidence_loss.lower()
        self.stator = stator
        self.rotor = rotor

        # AERODYNAMIC LOSS: PROFILE LOSS DATA
        # Data for chi_R
        # Figure 8.4.6 has logarithmic Reynolds-number abscissa. Reynolds number is therefore transformed before it is
        # passed to the gridded interpolator. The second coordinate is the equivalent sand roughness over blade chord.
        Re_chi_R = np.array([1e4, 1.7e4, 3e4, 4.2e4, 7e4, 1.35e5, 2e5, 2.5e5, 1e7])
        log_Re_chi_R = np.log10(Re_chi_R)
        k_s_over_s = np.array([1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 2e-3])
        chi_R = np.array([
            [4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50],
            [3.50, 3.50, 3.50, 3.50, 3.50, 3.50, 3.50],
            [2.60, 2.60, 2.60, 2.60, 2.60, 2.60, 3.50],
            [2.25, 2.25, 2.25, 2.25, 2.25, 2.60, 3.50],
            [1.70, 1.70, 1.70, 1.70, 2.25, 2.60, 3.50],
            [1.15, 1.15, 1.15, 1.70, 2.25, 2.60, 3.50],
            [1.00, 1.00, 1.15, 1.70, 2.25, 2.60, 3.50],
            [0.95, 1.00, 1.15, 1.70, 2.25, 2.60, 3.50],
            [0.95, 1.00, 1.15, 1.70, 2.25, 2.60, 3.50],
        ])
        interpolator_chi_R = RegularGridInterpolator(
            (log_Re_chi_R, k_s_over_s), chi_R, method=interpolation_method, bounds_error=False, fill_value=None)
        chi_R_bounds = ((log_Re_chi_R[0], log_Re_chi_R[-1]), (k_s_over_s[0], k_s_over_s[-1]))
        self.interpolator_chi_R = lambda Re, roughness: self.__interpolate_data(
            np.log10(Re), roughness, bounds=chi_R_bounds, interpolator=interpolator_chi_R,
            coefficient_name="chi_R")

        # Data for chi_M
        # Curves 1-4 in Figure 8.4.5 describe different cascades.
        # Curve 1 refers to strongly accelerating cascades like stators and reaction rotors.
        # Curve 2 represent strongly accelerating cascades where uncovered turning region is adapted to near-sonic flow
        # Curve 3 and 4 represent impulse-rotor cascades with rounded and sharp LE respectively. The latter is
        # suitable for high Mach numbers.
        M_curve_1 = np.array([
            0.00, 0.20, 0.40, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20,
            1.25, 1.30])
        chi_M_curve_1 = np.array([
            1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.98, 0.96, 0.94, 0.90, 0.85, 0.84, 0.89, 1.08,
            1.32, 1.60])
        M_curve_2 = np.array([
            0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35,
            1.40])
        chi_M_curve_2 = np.array([
            0.98, 0.95, 0.91, 0.87, 0.83, 0.79, 0.75, 0.72, 0.73, 0.75, 0.78, 0.81, 0.84, 0.88, 0.93,
            1.00])
        M_curve_3 = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75])
        chi_M_curve_3 = np.array([1.00, 0.95, 0.91, 0.88, 0.86, 0.88, 0.94, 1.08, 1.20])
        M_curve_4 = np.array([
            0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30])
        chi_M_curve_4 = np.array([
            0.98, 0.92, 0.88, 0.84, 0.80, 0.76, 0.73, 0.71, 0.75, 0.81, 0.90, 1.00, 1.13, 1.30])

        # Construct one gridded interpolator per curve, each with its own Mach range.
        interpolator_chi_M_curve_1 = RegularGridInterpolator(
            (M_curve_1,), chi_M_curve_1, method=interpolation_method, bounds_error=False, fill_value=None)
        interpolator_chi_M_curve_2 = RegularGridInterpolator(
            (M_curve_2,), chi_M_curve_2, method=interpolation_method, bounds_error=False, fill_value=None)
        interpolator_chi_M_curve_3 = RegularGridInterpolator(
            (M_curve_3,), chi_M_curve_3, method=interpolation_method, bounds_error=False, fill_value=None)
        interpolator_chi_M_curve_4 = RegularGridInterpolator(
            (M_curve_4,), chi_M_curve_4, method=interpolation_method, bounds_error=False, fill_value=None)
        chi_M_curve_1_bounds = ((M_curve_1[0], M_curve_1[-1]),)
        chi_M_curve_2_bounds = ((M_curve_2[0], M_curve_2[-1]),)
        chi_M_curve_3_bounds = ((M_curve_3[0], M_curve_3[-1]),)
        chi_M_curve_4_bounds = ((M_curve_4[0], M_curve_4[-1]),)
        self.interpolator_chi_M_curve_1 = lambda mach_number: self.__interpolate_data(
            mach_number, bounds=chi_M_curve_1_bounds, interpolator=interpolator_chi_M_curve_1,
            coefficient_name="chi_M_curve_1")
        self.interpolator_chi_M_curve_2 = lambda mach_number: self.__interpolate_data(
            mach_number, bounds=chi_M_curve_2_bounds, interpolator=interpolator_chi_M_curve_2,
            coefficient_name="chi_M_curve_2")
        self.interpolator_chi_M_curve_3 = lambda mach_number: self.__interpolate_data(
            mach_number, bounds=chi_M_curve_3_bounds, interpolator=interpolator_chi_M_curve_3,
            coefficient_name="chi_M_curve_3")
        self.interpolator_chi_M_curve_4 = lambda mach_number: self.__interpolate_data(
            mach_number, bounds=chi_M_curve_4_bounds, interpolator=interpolator_chi_M_curve_4,
            coefficient_name="chi_M_curve_4")

        # Data for zeta_p0
        # Both angles use Traupel's original definition from the vertical direction. The 10 degree outlet-angle curve
        # is omitted because its short visible range would exclude the well-resolved low inlet-angle part of the graph.
        # Additional support points are used in the low-angle region where the curves separate in quick succession.
        inlet_angle_deg = np.array([20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120])
        outlet_angle_deg = np.array([15, 20, 25, 30, 35, 45])
        # At low inlet angles the outlet-angle curves successively merge into one common curve. Equal entries below
        # retain that feature of the graph instead of treating the thickness of the blended line as separate data.
        zeta_p0 = np.array([
            [0.0700, 0.0700, 0.0700, 0.0700, 0.0700, 0.0700],
            [0.0550, 0.0550, 0.0550, 0.0550, 0.0550, 0.0550],
            [0.0500, 0.0450, 0.0450, 0.0450, 0.0450, 0.0450],
            [0.0470, 0.0400, 0.0390, 0.0390, 0.0390, 0.0390],
            [0.0440, 0.0370, 0.0340, 0.0340, 0.0340, 0.0340],
            [0.0420, 0.0350, 0.0315, 0.0290, 0.0290, 0.0290],
            [0.0405, 0.0330, 0.0290, 0.0260, 0.0260, 0.0260],
            [0.0380, 0.0305, 0.0260, 0.0240, 0.0210, 0.0210],
            [0.0365, 0.0290, 0.0245, 0.0215, 0.0195, 0.0170],
            [0.0345, 0.0270, 0.0225, 0.0195, 0.0175, 0.0145],
            [0.0315, 0.0240, 0.0195, 0.0165, 0.0140, 0.0105],
            [0.0285, 0.0210, 0.0165, 0.0130, 0.0105, 0.0060],
        ])
        interpolator_zeta_p0 = RegularGridInterpolator(
            (inlet_angle_deg, outlet_angle_deg), zeta_p0, method=interpolation_method, bounds_error=False,
            fill_value=None)
        zeta_p0_bounds = ((inlet_angle_deg[0], inlet_angle_deg[-1]),
                          (outlet_angle_deg[0], outlet_angle_deg[-1]))
        self.interpolator_zeta_p0 = lambda inlet_angle, outlet_angle: self.__interpolate_data(
            inlet_angle, outlet_angle, bounds=zeta_p0_bounds, interpolator=interpolator_zeta_p0,
            coefficient_name="zeta_p0")

        # Data for zeta_h
        # The upper-left part of Figure 8.4.5 is a nomogram. The left curve supplies the transfer ordinate from
        # Delta_a / (chi_M * chi_R * zeta_p0), while the straight rays on the right convert it to zeta_h for Delta_a.
        delta_a_over_corrected_zeta_p0 = np.array([1, 2, 3, 4, 5, 6, 8, 10])
        delta_a = np.array([0.04, 0.08, 0.12, 0.16, 0.20])
        # Rows follow the delta_a ratio above and columns follow delta_a. The straight rays are continued past the
        # right edge of the nomogram for the last two values on the delta_a = 0.20 ray.
        zeta_h = np.array([
            [0.00330, 0.00660, 0.00990, 0.01320, 0.01650],
            [0.00534, 0.01068, 0.01602, 0.02136, 0.02670],
            [0.00732, 0.01464, 0.02196, 0.02928, 0.03660],
            [0.00851, 0.01702, 0.02552, 0.03403, 0.04254],
            [0.00913, 0.01826, 0.02740, 0.03653, 0.04566],
            [0.00976, 0.01951, 0.02927, 0.03902, 0.04878],
            [0.01073, 0.02146, 0.03218, 0.04291, 0.05364],
            [0.01157, 0.02314, 0.03470, 0.04627, 0.05784],
        ])
        interpolator_zeta_h = RegularGridInterpolator(
            (delta_a_over_corrected_zeta_p0, delta_a), zeta_h, method=interpolation_method, bounds_error=False,
            fill_value=None)
        zeta_h_bounds = ((delta_a_over_corrected_zeta_p0[0], delta_a_over_corrected_zeta_p0[-1]),
                         (delta_a[0], delta_a[-1]))
        self.interpolator_zeta_h = lambda delta_a_ratio, trailing_edge_blockage: self.__interpolate_data(
            delta_a_ratio, trailing_edge_blockage, bounds=zeta_h_bounds, interpolator=interpolator_zeta_h,
            coefficient_name="zeta_h")

        # AERODYNAMIC LOSS: FANNING LOSS DATA
        # Data for zeta_f
        blade_length_over_mean_diameter = np.array([0.0, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20])
        speed_parameter = np.array([0.5, 0.7, 0.9])
        # All curves pass through the origin. The 0.9 curve starts just beyond 0.05, so its value there is estimated
        # from the first visible part of the curve.
        zeta_f = np.transpose([
            [0.0, 0.00102, 0.00196, 0.00336, 0.00515, 0.00742, 0.01020, 0.01350],
            [0.0, 0.00062, 0.00140, 0.00259, 0.00410, 0.00597, 0.00836, 0.01125],
            [0.0, 0.00012, 0.00070, 0.00159, 0.00282, 0.00446, 0.00645, 0.00879],
        ])
        interpolator_zeta_f = RegularGridInterpolator(
            (blade_length_over_mean_diameter, speed_parameter), zeta_f, method="slinear",
            bounds_error=False, fill_value=None)
        zeta_f_bounds = ((blade_length_over_mean_diameter[0], blade_length_over_mean_diameter[-1]),
                         (speed_parameter[0], speed_parameter[-1]))
        self.interpolator_zeta_f = lambda length_ratio, nu: self.__interpolate_data(
            length_ratio, nu, bounds=zeta_f_bounds, interpolator=interpolator_zeta_f,
            coefficient_name="zeta_f")

        # AERODYNAMIC LOSS: ENDWALL & SECONDARY FLOWS LOSS DATA
        # Data for F factor
        turning_angle_deg = np.array([10, 20, 40, 60, 80, 100, 120, 140])
        velocity_ratio = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        F = np.array([
            [0.0137, 0.0177, 0.0232, 0.0287, 0.0335, 0.0383, 0.0442, 0.0511, 0.0599],
            [0.0170, 0.0238, 0.0306, 0.0377, 0.0446, 0.0509, 0.0593, 0.0683, 0.0784],
            [0.0226, 0.0312, 0.0399, 0.0494, 0.0591, 0.0683, 0.0782, 0.0897, 0.1035],
            [0.0274, 0.0377, 0.0480, 0.0591, 0.0700, 0.0818, 0.0940, 0.1087, 0.1235],
            [0.0318, 0.0441, 0.0556, 0.0684, 0.0808, 0.0953, 0.1100, 0.1268, 0.1455],
            [0.0362, 0.0502, 0.0641, 0.0778, 0.0927, 0.1093, 0.1268, 0.1470, 0.1709],
            [0.0406, 0.0562, 0.0727, 0.0880, 0.1057, 0.1246, 0.1455, 0.1692, 0.1993],
            [0.0450, 0.0622, 0.0811, 0.0990, 0.1192, 0.1411, 0.1659, 0.1943, 0.2320],
        ])
        interpolator_F = RegularGridInterpolator((turning_angle_deg, velocity_ratio), F,
                                                 method=interpolation_method, bounds_error=False, fill_value=None)
        F_bounds = ((turning_angle_deg[0], turning_angle_deg[-1]), (velocity_ratio[0], velocity_ratio[-1]))
        self.interpolator_F = lambda turning_angle, inlet_over_outlet_velocity: self.__interpolate_data(
            turning_angle, inlet_over_outlet_velocity, bounds=F_bounds, interpolator=interpolator_F,
            coefficient_name="F")

        # Data for c_f factor
        Re_c_f = np.array([1e4, 2e4, 5e4, 7e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7])
        log_Re_c_f = np.log10(Re_c_f)
        # The 1e-4 curve is omitted. The 2e-4 and 5e-4 curves start near Re = 7e4;
        # below that, their values are estimated from linear extrapolation.
        k_s_over_d_h = np.array([0, 2e-4, 5e-4, 1e-3, 2e-3])
        c_f = np.array([
            [0.00769, 0.00780, 0.00798, 0.00828, 0.00883],
            [0.00641, 0.00652, 0.00669, 0.00697, 0.00748],
            [0.00518, 0.00533, 0.00556, 0.00594, 0.00655],
            [0.00483, 0.00505, 0.00539, 0.00570, 0.00637],
            [0.00448, 0.00477, 0.00511, 0.00546, 0.00619],
            [0.00394, 0.00431, 0.00473, 0.00516, 0.00600],
            [0.00332, 0.00395, 0.00439, 0.00494, 0.00592],
            [0.00293, 0.00373, 0.00427, 0.00492, 0.00589],
            [0.00262, 0.00358, 0.00416, 0.00492, 0.00589],
            [0.00228, 0.00347, 0.00410, 0.00492, 0.00583],
            [0.00203, 0.00339, 0.00409, 0.00492, 0.00581],
        ])
        interpolator_c_f = RegularGridInterpolator(
            (log_Re_c_f, k_s_over_d_h), c_f, method=interpolation_method, bounds_error=False, fill_value=None)
        c_f_bounds = ((log_Re_c_f[0], log_Re_c_f[-1]), (k_s_over_d_h[0], k_s_over_d_h[-1]))
        self.interpolator_c_f = lambda Re, relative_roughness: self.__interpolate_data(
            np.log10(Re), relative_roughness, bounds=c_f_bounds, interpolator=interpolator_c_f,
            coefficient_name="c_f")

        # CLEARANCE LOSS DATA
        # Data for K_sigma
        # Figure 8.4.16 uses sin(alpha_1) or sin(beta_2), the circumferential-velocity change over the normal velocity,
        # and x = delta / s - 0.002. The tabulated curves retain that order in the property arguments.
        sin_outlet_angle = np.array([0.30, 0.35, 0.40, 0.50])
        x_clearance = np.array([0.010, 0.020, 0.030, 0.045, 0.060])
        # Keep the original sampling ranges of the four panels. In each table, columns follow x_clearance above.
        velocity_change_ratio_030 = np.array([1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00])
        K_sigma_030 = np.array([
            [3.11, 2.91, 2.79, 2.52, 2.30],
            [3.50, 3.25, 3.10, 2.84, 2.61],
            [3.93, 3.63, 3.45, 3.17, 2.91],
            [4.41, 4.04, 3.83, 3.52, 3.19],
            [4.91, 4.49, 4.21, 3.87, 3.46],
            [5.43, 4.95, 4.59, 4.20, 3.71],
            [5.95, 5.39, 4.95, 4.52, 3.95],
            [6.46, 5.81, 5.29, 4.83, 4.17],
            [6.94, 6.23, 5.63, 5.11, 4.37],
            [7.40, 6.64, 5.95, 5.38, 4.55],
        ])
        velocity_change_ratio_035 = np.array([1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50])
        K_sigma_035 = np.array([
            [2.97, 2.74, 2.57, 2.33, 2.11],
            [3.47, 3.15, 2.96, 2.72, 2.45],
            [3.98, 3.56, 3.36, 3.09, 2.77],
            [4.50, 4.00, 3.74, 3.44, 3.08],
            [5.00, 4.46, 4.10, 3.77, 3.37],
            [5.52, 4.91, 4.45, 4.07, 3.62],
            [6.03, 5.34, 4.79, 4.33, 3.84],
            [6.54, 5.76, 5.11, 4.55, 4.03],
            [7.04, 6.14, 5.43, 4.71, 4.14],
        ])
        velocity_change_ratio_040 = np.array([1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00])
        K_sigma_040 = np.array([
            [2.61, 2.34, 2.15, 1.94, 1.74],
            [3.10, 2.76, 2.53, 2.29, 2.09],
            [3.62, 3.20, 2.90, 2.65, 2.42],
            [4.15, 3.65, 3.28, 2.98, 2.72],
            [4.70, 4.11, 3.65, 3.31, 2.99],
            [5.27, 4.59, 4.02, 3.61, 3.24],
            [5.79, 5.03, 4.35, 3.89, 3.46],
            [6.18, 5.42, 4.62, 4.11, 3.64],
        ])
        velocity_change_ratio_050 = np.array([1.00, 1.25, 1.50, 1.75, 2.00, 2.25])
        K_sigma_050 = np.array([
            [2.34, 2.13, 1.90, 1.72, 1.56],
            [2.81, 2.55, 2.29, 2.10, 1.94],
            [3.24, 2.95, 2.65, 2.45, 2.26],
            [3.62, 3.31, 2.99, 2.75, 2.55],
            [3.99, 3.62, 3.29, 3.02, 2.78],
            [4.29, 3.90, 3.53, 3.24, 2.98],
        ])

        # Extend each measured curve to a common velocity-change range. PCHIP estimates the endpoint tangents
        # from the first/last three samples; the added samples lie on straight lines attached at those endpoints.
        # The common grid has one slice per sine-angle panel, one row per velocity ratio and one column per clearance.
        velocity_change_over_normal_velocity = np.arange(1.0, 4.01, 0.25)
        K_sigma = np.empty((len(sin_outlet_angle), len(velocity_change_over_normal_velocity), len(x_clearance)))
        # Pair each panel's original velocity samples with its measured K_sigma values in sine-angle order.
        K_sigma_panels = ((velocity_change_ratio_030, K_sigma_030), (velocity_change_ratio_035, K_sigma_035),
                          (velocity_change_ratio_040, K_sigma_040), (velocity_change_ratio_050, K_sigma_050))
        for panel_index, (curve_velocity_ratio, curve_K_sigma) in enumerate(K_sigma_panels):
            # PCHIP only prepares the common grid; the final coefficient lookup below still uses slinear.
            # axis=0 fits velocity ratio separately for every clearance column of this panel.
            curve_interpolator = PchipInterpolator(curve_velocity_ratio, curve_K_sigma, axis=0)
            # Hold positions outside the measured interval at an endpoint before adding straight-line extensions.
            clipped_velocity_ratio = np.clip(
                velocity_change_over_normal_velocity, curve_velocity_ratio[0], curve_velocity_ratio[-1])
            K_sigma[panel_index] = curve_interpolator(clipped_velocity_ratio)
            # The two derivative rows contain the lower and upper endpoint slope for each clearance column.
            endpoint_slopes = curve_interpolator.derivative()(curve_velocity_ratio[[0, -1]])
            below_curve = velocity_change_over_normal_velocity < curve_velocity_ratio[0]
            above_curve = velocity_change_over_normal_velocity > curve_velocity_ratio[-1]
            # [:, None] makes each endpoint distance a column, so it multiplies all clearance slopes at once.
            K_sigma[panel_index, below_curve] += (
                velocity_change_over_normal_velocity[below_curve] - curve_velocity_ratio[0])[:, None] \
                                                 * endpoint_slopes[0]
            K_sigma[panel_index, above_curve] += (
                velocity_change_over_normal_velocity[above_curve] - curve_velocity_ratio[-1])[:, None] \
                                                 * endpoint_slopes[1]
        # The four extended panels now form one rectangular grid for the public three-input interpolator.
        interpolator_K_sigma = RegularGridInterpolator(
            (sin_outlet_angle, velocity_change_over_normal_velocity, x_clearance), K_sigma,
            method=interpolation_method, bounds_error=False, fill_value=None)
        # These are the bounds of the extended grid, not the shorter measured range of each individual panel.
        K_sigma_bounds = ((sin_outlet_angle[0], sin_outlet_angle[-1]),
                          (velocity_change_over_normal_velocity[0], velocity_change_over_normal_velocity[-1]),
                          (x_clearance[0], x_clearance[-1]))
        # The shared wrapper applies the selected extrapolation rule and names K_sigma in any warning.
        self.interpolator_K_sigma = lambda sine_angle, velocity_change_ratio, x: self.__interpolate_data(
            sine_angle, velocity_change_ratio, x, bounds=K_sigma_bounds, interpolator=interpolator_K_sigma,
            coefficient_name="K_sigma")

        # DISK FRICTION LOSS DATA
        # Data for C_M
        # Figure 8.4.6 has logarithmic axes. Only the start, corner and end of its two straight segments are needed.
        Re_C_M = np.array([7e4, 2e5, 1e7])
        log_Re_C_M = np.log10(Re_C_M)
        C_M = np.array([4.35e-4, 2.60e-4, 1.18e-4])
        interpolator_C_M = RegularGridInterpolator(
            (log_Re_C_M,), np.log10(C_M), method="slinear", bounds_error=False, fill_value=None)
        # Convert back after interpolation or extrapolation on the graph's logarithmic axes.
        interpolator_C_M_value = lambda points: 10 ** interpolator_C_M(points)
        C_M_bounds = ((log_Re_C_M[0], log_Re_C_M[-1]),)
        self.interpolator_C_M = lambda Re: self.__interpolate_data(
            np.log10(Re), bounds=C_M_bounds, interpolator=interpolator_C_M_value, coefficient_name="C_M")

        # INCIDENCE LOSS
        # Factor z data
        # Figure 8.4.21 uses inlet-angle deviation in degrees, with Traupel's original angles measured from vertical.
        # Curve a reaches the upper edge at about +43.7 degrees; curve b remains visible through +60 degrees.
        incidence_angle_deg_a = np.array([
            -50, -45, -40, -35, -30, -25, -20, -15, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 43.7])
        z_a = np.array([
            0.1768, 0.1597, 0.1404, 0.1189, 0.0944, 0.0696, 0.0476, 0.0288, 0.0133, 0.0000,
            0.0090, 0.0305, 0.0641, 0.1061, 0.1572, 0.2160, 0.2797, 0.3461, 0.4000])
        incidence_angle_deg_b = np.array([
            -50, -45, -40, -35, -30, -25, -20, -15, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
        z_b = np.array([
            0.1135, 0.0985, 0.0820, 0.0643, 0.0458, 0.0302, 0.0177, 0.0077, 0.0028, 0.0000,
            0.0022, 0.0075, 0.0189, 0.0350, 0.0567, 0.0838, 0.1183, 0.1610, 0.2095, 0.2655, 0.3276, 0.3928])
        interpolator_z_a = RegularGridInterpolator(
            (incidence_angle_deg_a,), z_a, method=interpolation_method, bounds_error=False, fill_value=None)
        interpolator_z_b = RegularGridInterpolator(
            (incidence_angle_deg_b,), z_b, method=interpolation_method, bounds_error=False, fill_value=None)
        z_a_bounds = ((incidence_angle_deg_a[0], incidence_angle_deg_a[-1]),)
        z_b_bounds = ((incidence_angle_deg_b[0], incidence_angle_deg_b[-1]),)
        self.interpolator_z_a = lambda incidence_angle: self.__interpolate_data(
            incidence_angle, bounds=z_a_bounds, interpolator=interpolator_z_a,
            coefficient_name="z_a")
        self.interpolator_z_b = lambda incidence_angle: self.__interpolate_data(
            incidence_angle, bounds=z_b_bounds, interpolator=interpolator_z_b,
            coefficient_name="z_b")
        # Average of the interpolated curves
        self.interpolator_z_average = lambda incidence_angle: (
            self.interpolator_z_a(incidence_angle) + self.interpolator_z_b(incidence_angle)) / 2

        # HALF LABIRYNTH FLOW FUNCTION
        # Uncapped Phi^2 data from the manually extended Figure 10.4.2, sampled down to p_2/p_1 = 0.3.
        # The upper plot selects a common abscissa from p_2/p_1 and the tooth-count curve; the lower plot gives Phi^2.
        p_2_over_p_1 = np.array([0.3, 0.4, 0.5, 0.525, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0])
        teeth_spacing_over_seal_clearance = np.array([5, 6, 7, 8, 10, 13])
        teeth_number = np.array([4, 5, 6, 8, 10, 12, 15])
        # Each pressure-ratio slice has one row per spacing curve and one column per tooth-count curve.
        # These values follow the curves beyond g; calculate_phi applies the physical flow limit separately.
        phi_squared = np.array([
            # p_2_over_p_1 = 0.3
            [
                [0.2914, 0.2551, 0.2337, 0.1982, 0.1721, 0.1508, 0.1262],  # teeth_spacing_over_seal_clearance = 5
                [0.2706, 0.2380, 0.2170, 0.1823, 0.1571, 0.1374, 0.1146],  # teeth_spacing_over_seal_clearance = 6
                [0.2552, 0.2224, 0.2019, 0.1681, 0.1440, 0.1255, 0.1041],  # teeth_spacing_over_seal_clearance = 7
                [0.2458, 0.2121, 0.1910, 0.1579, 0.1340, 0.1163, 0.0959],  # teeth_spacing_over_seal_clearance = 8
                [0.2206, 0.1884, 0.1694, 0.1388, 0.1174, 0.1016, 0.0838],  # teeth_spacing_over_seal_clearance = 10
                [0.2007, 0.1721, 0.1536, 0.1243, 0.1046, 0.0895, 0.0732],  # teeth_spacing_over_seal_clearance = 13
            ],
            # p_2_over_p_1 = 0.4
            [
                [0.2778, 0.2459, 0.2254, 0.1917, 0.1672, 0.1430, 0.1215],  # teeth_spacing_over_seal_clearance = 5
                [0.2589, 0.2289, 0.2090, 0.1762, 0.1525, 0.1301, 0.1100],  # teeth_spacing_over_seal_clearance = 6
                [0.2436, 0.2136, 0.1940, 0.1621, 0.1398, 0.1187, 0.0999],  # teeth_spacing_over_seal_clearance = 7
                [0.2334, 0.2030, 0.1831, 0.1519, 0.1299, 0.1099, 0.0922],  # teeth_spacing_over_seal_clearance = 8
                [0.2087, 0.1801, 0.1623, 0.1336, 0.1138, 0.0961, 0.0805],  # teeth_spacing_over_seal_clearance = 10
                [0.1909, 0.1640, 0.1468, 0.1193, 0.1012, 0.0845, 0.0702],  # teeth_spacing_over_seal_clearance = 13
            ],
            # p_2_over_p_1 = 0.5
            [
                [0.2641, 0.2350, 0.2115, 0.1782, 0.1541, 0.1305, 0.1105],  # teeth_spacing_over_seal_clearance = 5
                [0.2466, 0.2182, 0.1954, 0.1632, 0.1402, 0.1187, 0.0995],  # teeth_spacing_over_seal_clearance = 6
                [0.2311, 0.2032, 0.1805, 0.1497, 0.1284, 0.1079, 0.0905],  # teeth_spacing_over_seal_clearance = 7
                [0.2206, 0.1922, 0.1699, 0.1399, 0.1189, 0.0993, 0.0834],  # teeth_spacing_over_seal_clearance = 8
                [0.1964, 0.1704, 0.1504, 0.1227, 0.1040, 0.0869, 0.0725],  # teeth_spacing_over_seal_clearance = 10
                [0.1798, 0.1546, 0.1349, 0.1092, 0.0919, 0.0760, 0.0633],  # teeth_spacing_over_seal_clearance = 13
            ],
            # p_2_over_p_1 = 0.525
            [
                [0.2595, 0.2302, 0.2066, 0.1737, 0.1493, 0.1266, 0.1070],  # teeth_spacing_over_seal_clearance = 5
                [0.2423, 0.2136, 0.1906, 0.1587, 0.1361, 0.1149, 0.0963],  # teeth_spacing_over_seal_clearance = 6
                [0.2266, 0.1985, 0.1759, 0.1455, 0.1242, 0.1044, 0.0876],  # teeth_spacing_over_seal_clearance = 7
                [0.2163, 0.1877, 0.1655, 0.1355, 0.1151, 0.0962, 0.0805],  # teeth_spacing_over_seal_clearance = 8
                [0.1923, 0.1664, 0.1461, 0.1188, 0.1005, 0.0841, 0.0700],  # teeth_spacing_over_seal_clearance = 10
                [0.1758, 0.1507, 0.1310, 0.1058, 0.0885, 0.0735, 0.0612],  # teeth_spacing_over_seal_clearance = 13
            ],
            # p_2_over_p_1 = 0.55
            [
                [0.2538, 0.2242, 0.2006, 0.1684, 0.1442, 0.1225, 0.1032],  # teeth_spacing_over_seal_clearance = 5
                [0.2367, 0.2078, 0.1847, 0.1536, 0.1312, 0.1110, 0.0929],  # teeth_spacing_over_seal_clearance = 6
                [0.2211, 0.1928, 0.1703, 0.1407, 0.1197, 0.1008, 0.0844],  # teeth_spacing_over_seal_clearance = 7
                [0.2109, 0.1819, 0.1600, 0.1309, 0.1108, 0.0929, 0.0775],  # teeth_spacing_over_seal_clearance = 8
                [0.1874, 0.1612, 0.1409, 0.1147, 0.0969, 0.0812, 0.0674],  # teeth_spacing_over_seal_clearance = 10
                [0.1710, 0.1458, 0.1262, 0.1020, 0.0852, 0.0709, 0.0589],  # teeth_spacing_over_seal_clearance = 13
            ],
            # p_2_over_p_1 = 0.6
            [
                [0.2400, 0.2092, 0.1865, 0.1561, 0.1325, 0.1131, 0.0946],  # teeth_spacing_over_seal_clearance = 5
                [0.2231, 0.1931, 0.1711, 0.1419, 0.1204, 0.1020, 0.0851],  # teeth_spacing_over_seal_clearance = 6
                [0.2081, 0.1783, 0.1572, 0.1300, 0.1095, 0.0926, 0.0771],  # teeth_spacing_over_seal_clearance = 7
                [0.1971, 0.1678, 0.1472, 0.1204, 0.1010, 0.0855, 0.0704],  # teeth_spacing_over_seal_clearance = 8
                [0.1747, 0.1483, 0.1294, 0.1054, 0.0885, 0.0745, 0.0614],  # teeth_spacing_over_seal_clearance = 10
                [0.1589, 0.1330, 0.1153, 0.0932, 0.0775, 0.0649, 0.0536],  # teeth_spacing_over_seal_clearance = 13
            ],
            # p_2_over_p_1 = 0.7
            [
                [0.1994, 0.1716, 0.1510, 0.1254, 0.1053, 0.0904, 0.0747],  # teeth_spacing_over_seal_clearance = 5
                [0.1836, 0.1567, 0.1377, 0.1138, 0.0948, 0.0809, 0.0671],  # teeth_spacing_over_seal_clearance = 6
                [0.1693, 0.1436, 0.1257, 0.1034, 0.0862, 0.0735, 0.0605],  # teeth_spacing_over_seal_clearance = 7
                [0.1590, 0.1336, 0.1165, 0.0953, 0.0792, 0.0671, 0.0550],  # teeth_spacing_over_seal_clearance = 8
                [0.1399, 0.1171, 0.1017, 0.0833, 0.0689, 0.0584, 0.0476],  # teeth_spacing_over_seal_clearance = 10
                [0.1253, 0.1043, 0.0896, 0.0728, 0.0602, 0.0508, 0.0413],  # teeth_spacing_over_seal_clearance = 13
            ],
            # p_2_over_p_1 = 0.8
            [
                [0.1436, 0.1223, 0.1071, 0.0879, 0.0729, 0.0623, 0.0510],  # teeth_spacing_over_seal_clearance = 5
                [0.1307, 0.1108, 0.0964, 0.0786, 0.0654, 0.0559, 0.0458],  # teeth_spacing_over_seal_clearance = 6
                [0.1192, 0.1006, 0.0877, 0.0714, 0.0589, 0.0504, 0.0412],  # teeth_spacing_over_seal_clearance = 7
                [0.1104, 0.0928, 0.0806, 0.0652, 0.0536, 0.0460, 0.0378],  # teeth_spacing_over_seal_clearance = 8
                [0.0965, 0.0810, 0.0701, 0.0566, 0.0463, 0.0398, 0.0324],  # teeth_spacing_over_seal_clearance = 10
                [0.0849, 0.0708, 0.0613, 0.0493, 0.0402, 0.0343, 0.0278],  # teeth_spacing_over_seal_clearance = 13
            ],
            # p_2_over_p_1 = 0.9
            [
                [0.0754, 0.0651, 0.0560, 0.0435, 0.0371, 0.0312, 0.0255],  # teeth_spacing_over_seal_clearance = 5
                [0.0676, 0.0584, 0.0503, 0.0388, 0.0329, 0.0277, 0.0232],  # teeth_spacing_over_seal_clearance = 6
                [0.0610, 0.0526, 0.0453, 0.0351, 0.0298, 0.0252, 0.0206],  # teeth_spacing_over_seal_clearance = 7
                [0.0555, 0.0480, 0.0414, 0.0319, 0.0269, 0.0225, 0.0184],  # teeth_spacing_over_seal_clearance = 8
                [0.0480, 0.0416, 0.0356, 0.0275, 0.0232, 0.0195, 0.0159],  # teeth_spacing_over_seal_clearance = 10
                [0.0417, 0.0359, 0.0307, 0.0234, 0.0196, 0.0165, 0.0137],  # teeth_spacing_over_seal_clearance = 13
            ],
            # p_2_over_p_1 = 1.0
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # teeth_spacing_over_seal_clearance = 5
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # teeth_spacing_over_seal_clearance = 6
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # teeth_spacing_over_seal_clearance = 7
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # teeth_spacing_over_seal_clearance = 8
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # teeth_spacing_over_seal_clearance = 10
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # teeth_spacing_over_seal_clearance = 13
            ],
        ])
        interpolator_phi_squared = RegularGridInterpolator(
            (p_2_over_p_1, teeth_spacing_over_seal_clearance, teeth_number), phi_squared,
            method=interpolation_method, bounds_error=False, fill_value=None)
        phi_squared_bounds = ((p_2_over_p_1[0], p_2_over_p_1[-1]),
                              (teeth_spacing_over_seal_clearance[0], teeth_spacing_over_seal_clearance[-1]),
                              (teeth_number[0], teeth_number[-1]))
        self.interpolator_phi_squared = lambda pressure_ratio, spacing_ratio, tooth_count: self.__interpolate_data(
            pressure_ratio, spacing_ratio, tooth_count, bounds=phi_squared_bounds,
            interpolator=interpolator_phi_squared, coefficient_name="phi_squared")

        # Solid g curves in the upper plot determine critical pressure ratios. Below these pressure ratios, phi does
        # not increase. Only teeth_spacing_over_seal_clearance = 5, 7, 10 and 13 have g curves.
        # Critical pressure ratio interpolator supplies intermediate ratios.
        teeth_spacing_over_seal_clearance_g = np.array([5, 7, 10, 13])
        # Rows follow the four g curves and columns follow teeth_number, as in the Phi^2 table.
        critical_pressure_ratio = np.array([
            [0.5626, 0.5348, 0.5142, 0.4824, 0.4562, 0.4244, 0.3960],  # teeth_spacing_over_seal_clearance = 5
            [0.5338, 0.5084, 0.4863, 0.4525, 0.4260, 0.3953, 0.3646],  # teeth_spacing_over_seal_clearance = 7
            [0.5104, 0.4823, 0.4588, 0.4229, 0.3951, 0.3637, 0.3300],  # teeth_spacing_over_seal_clearance = 10
            [0.4954, 0.4665, 0.4429, 0.4054, 0.3750, 0.3453, 0.3093],  # teeth_spacing_over_seal_clearance = 13
        ])
        # Transpose the table to accept tooth count first and spacing ratio second.
        interpolator_critical_pressure_ratio = RegularGridInterpolator(
            (teeth_number, teeth_spacing_over_seal_clearance_g), critical_pressure_ratio.T,
            method=interpolation_method, bounds_error=False, fill_value=None)
        critical_pressure_ratio_bounds = ((teeth_number[0], teeth_number[-1]),
                                          (teeth_spacing_over_seal_clearance_g[0],
                                           teeth_spacing_over_seal_clearance_g[-1]))
        self.interpolator_critical_pressure_ratio = lambda tooth_count, spacing_ratio: self.__interpolate_data(
            tooth_count, spacing_ratio, bounds=critical_pressure_ratio_bounds,
            interpolator=interpolator_critical_pressure_ratio, coefficient_name="critical_pressure_ratio")

    def __interpolate_data(self, *coordinates, bounds, interpolator, coefficient_name):
        """Interpolate one point and apply the selected rule outside the digitized graph."""

        # This wrapper accepts one scalar per graph axis, not arrays of evaluation points.
        if any(np.ndim(coordinate) != 0 for coordinate in coordinates):
            raise TypeError("Interpolation inputs must be scalars.")
        point = tuple(np.float64(coordinate) for coordinate in coordinates)

        # Keep the original point for extrapolation and a clipped copy for the closest-value option.
        clipped_point = tuple(np.clip(coordinate, lower_bound, upper_bound)
                              for coordinate, (lower_bound, upper_bound) in zip(point, bounds))
        outside_bounds = any(coordinate < lower_bound or coordinate > upper_bound
                             for coordinate, (lower_bound, upper_bound) in zip(point, bounds))

        # Report which coefficient is outside its graph region on every such call.
        if outside_bounds and self.warn_on_extrapolation:
            boundary_rule = ("The closest value is used instead of extrapolation"
                             if self.extrapolation_method == "closest"
                             else "SciPy linear extrapolation is used")
            warnings.warn(f"Traupel coefficient {coefficient_name} is outside its digitized graph bounds; "
                          f"{boundary_rule}.",
                          RuntimeWarning, stacklevel=3)

        # SciPy extrapolates linearly outside the grid; closest instead clips inputs to the grid boundary.
        evaluation_point = clipped_point if self.extrapolation_method == "closest" else point
        # Pass one row of coordinates, which also works for SciPy's one-dimensional grids.
        return interpolator([evaluation_point])[0]

    def calculate_chi_R(self, Re, relative_roughness):
        """A method to calculate the Reynolds-number and roughness correction factor chi_R.

        :param float or integer Re: Blade-row Reynolds number (-).
        :param float relative_roughness: Equivalent sand roughness over blade chord, k_s/s (-).
        :return: Reynolds-number and roughness correction factor chi_R (-).
        :rtype: float
        """

        # Sometimes extrapolation beyond data given by Traupel can return negative values. Therefore, the minimum
        # values the interpolator can return is zero.
        return max(self.interpolator_chi_R(Re, relative_roughness), 0)

    def calculate_chi_M(self, outlet_Mach_number, blade_row):
        """A method to calculate the Mach-number correction factor chi_M for the configured blade row.

        :param float outlet_Mach_number: Blade-row Mach number (-).
        :param str blade_row: Blade row to evaluate, either "stator" or "rotor". Its chi_M curve is selected by the
            stator or rotor setting given when the object was initialized.
        :return: Mach-number correction factor chi_M (-).
        :rtype: float
        """

        # Select the row configuration first, then its distinct Mach-correction curve.
        if not isinstance(blade_row, str) or blade_row not in ("stator", "rotor"):
            raise ValueError("blade_row must be 'stator' or 'rotor'.")
        configuration = self.stator if blade_row == "stator" else self.rotor
        interpolators = {"regular": self.interpolator_chi_M_curve_1,
                         "near_sonic_outlet": self.interpolator_chi_M_curve_2,
                         "impulse_low_M": self.interpolator_chi_M_curve_3,
                         "impulse_high_M": self.interpolator_chi_M_curve_4}
        # Sometimes extrapolation beyond data given by Traupel can return negative values. Therefore, the minimum
        # values the interpolator can return is zero.
        return max(interpolators[configuration](outlet_Mach_number), 0)

    def calculate_zeta_p0(self, inlet_angle, outlet_angle):
        """A method to calculate the uncorrected profile-loss coefficient zeta_p0.

        :param float inlet_angle: Blade-row inlet angle measured from the vertical direction (deg).
        :param float outlet_angle: Blade-row outlet angle measured from the vertical direction (deg).
        :return: Uncorrected profile-loss coefficient zeta_p0 (-).
        :rtype: float
        """

        # zeta_p0 minimum value is assumed to be 0.001. It cannot be zero like for other coefficients, since division
        # by zeta_p0 occurs for other coefficients.
        return max(self.interpolator_zeta_p0(inlet_angle, outlet_angle), 0.001)

    def calculate_zeta_h(self, delta_a_ratio, trailing_edge_blockage, Re):
        """A method to calculate the Reynolds-corrected loss due to underpressure behind the trailing edge.
        The nomogram value is multiplied by a quarter-ellipse transition from Re = 8e4 to Re = 1.5e5.

        :param float delta_a_ratio: Ratio delta_a / (chi_M * chi_R * zeta_p0) (-).
        :param float trailing_edge_blockage: Trailing-edge blockage delta_a (-).
        :param float or integer Re: Blade-row Reynolds number (-).
        :return: Reynolds-corrected trailing-edge underpressure loss coefficient zeta_h (-).
        :rtype: float
        """

        # Below Re = 8e4 the underpressure effect disappears, so the nomogram contribution is zero.
        if Re <= 8e4:
            return 0.0
        zeta_h = self.interpolator_zeta_h(delta_a_ratio, trailing_edge_blockage)  # -
        if Re >= 1.5e5:
            # Sometimes extrapolation beyond data given by Traupel can return negative values. Therefore, the minimum
            # values the interpolator can return is zero.
            return max(zeta_h, 0)
        # Approximate Traupel's elliptical transition with a quarter ellipse that levels off at full strength.
        relative_Re = (Re - 8e4) / (1.5e5 - 8e4)  # -
        transition_factor = np.sqrt(1 - (1 - relative_Re)**2)  # -
        # Sometimes extrapolation beyond data given by Traupel can return negative values. Therefore, the minimum
        # values the interpolator can return is zero.
        return max(zeta_h * transition_factor, 0)

    def calculate_zeta_f(self, length_ratio, speed_parameter):
        """A method to calculate the blade-row fanning-loss coefficient zeta_f.

        :param float length_ratio: Blade length over mean turbine diameter (-).
        :param float speed_parameter: Speed parameter of the blade row (-).
        :return: Fanning-loss coefficient zeta_f (-).
        :rtype: float
        """

        # Sometimes extrapolation beyond data given by Traupel can return negative values. Therefore, the minimum
        # values the interpolator can return is zero.
        return max(self.interpolator_zeta_f(length_ratio, speed_parameter), 0)

    def calculate_F(self, turning_angle, inlet_over_outlet_velocity):
        """A method to calculate factor F for endwall and secondary-flow losses.

        :param float turning_angle: Blade-row turning angle (deg).
        :param float inlet_over_outlet_velocity: Inlet-to-outlet velocity ratio (-).
        :return: Endwall and secondary-flow factor F (-).
        :rtype: float
        """

        # Sometimes extrapolation beyond data given by Traupel can return negative values. Therefore, the minimum
        # values the interpolator can return is zero.
        return max(self.interpolator_F(turning_angle, inlet_over_outlet_velocity), 0)

    def calculate_c_f(self, Re, relative_roughness):
        """A method to calculate the friction factor c_f.

        :param float or integer Re: Reynolds number for the friction-factor graph (-).
        :param float relative_roughness: Equivalent sand roughness over hydraulic diameter, k_s/d_h (-).
        :return: Friction factor c_f (-).
        :rtype: float
        """

        # Sometimes extrapolation beyond data given by Traupel can return negative values. Therefore, the minimum
        # values the interpolator can return is zero.
        return max(self.interpolator_c_f(Re, relative_roughness), 0)

    def calculate_K_sigma(self, outlet_angle, velocity_change_ratio, normalized_clearance):
        """A method to calculate the clearance-loss factor K_sigma.

        :param float outlet_angle: Blade-row outlet angle measured from the vertical direction (rad).
        :param float velocity_change_ratio: Circumferential-velocity change over normal velocity (-).
        :param float normalized_clearance: Clearance ratio delta/s minus 0.002 (-).
        :return: Clearance-loss factor K_sigma (-).
        :rtype: float
        """

        # Figure 8.4.16 uses the sine of Traupel's outlet angle as its first coordinate.
        sine_angle = np.sin(outlet_angle)  # -
        # Sometimes extrapolation beyond data given by Traupel can return negative values. Therefore, the minimum
        # values the interpolator can return is zero.
        return max(self.interpolator_K_sigma(sine_angle, velocity_change_ratio, normalized_clearance), 0)

    def calculate_C_M(self, Re):
        """A method to calculate the disk-friction coefficient C_M.

        :param float or integer Re: Reynolds number for the disk-friction graph (-).
        :return: Disk-friction coefficient C_M (-).
        :rtype: float
        """

        # Sometimes extrapolation beyond data given by Traupel can return negative values. Therefore, the minimum
        # values the interpolator can return is zero.
        return max(self.interpolator_C_M(Re), 0)

    def calculate_z(self, incidence_angle):
        """A method to calculate incidence-loss factor z using the curve selected at initialization.

        :param float incidence_angle: Deviation of the inlet angle from the design angle (deg).
        :return: Incidence-loss factor z (-).
        :rtype: float
        """

        # Medium incidence loss is the mean of the two interpolated curves, not a separate digitized curve.
        interpolators = {"low": self.interpolator_z_b,
                         "medium": self.interpolator_z_average,
                         "high": self.interpolator_z_a}
        # Sometimes extrapolation beyond data given by Traupel can return negative values. Therefore, the minimum
        # values the interpolator can return is zero.
        return max(interpolators[self.incidence_loss](incidence_angle), 0)

    def calculate_phi(self, p_2_over_p_1, teeth_spacing_over_seal_clearance, teeth_number):
        """A method to calculate the half-labyrinth flow function phi from Figure 10.4.2.
        The flow remains constant below the critical pressure ratio given by the corresponding g curve.

        :param float p_2_over_p_1: Outlet-to-inlet pressure ratio of the seal (-).
        :param float teeth_spacing_over_seal_clearance: Tooth spacing over seal clearance, a/delta (-).
        :param int or float teeth_number: Number of seal teeth (-).
        :return: Half-labyrinth flow function phi (-).
        :rtype: float
        """

        # Read the g limit for this geometry and stop reducing the pressure ratio once the flow is at its maximum.
        critical_pressure_ratio = self.interpolator_critical_pressure_ratio(
            teeth_number, teeth_spacing_over_seal_clearance)  # -
        pressure_ratio = max(p_2_over_p_1, critical_pressure_ratio)  # -
        # The zero limit is applied to the interpolator before taking a square root
        phi_squared = max(self.interpolator_phi_squared(pressure_ratio, teeth_spacing_over_seal_clearance,
                                                        teeth_number), 0)  # -
        return np.sqrt(phi_squared)

    def calculate_blade_row_aerodynamic_loss(self, blade_row, inlet_angle, outlet_angle, t_TE_over_pitch,
                                             roughness_over_chord, roughness_over_hydraulic_diameter,
                                             blade_length_over_mean_diameter, chord_over_blade_length, chord_over_pitch,
                                             outlet_Mach_number, Reynolds_number, speed_parameter, blade_velocity_ratio,
                                             axial_clearance_over_blade_length, shrouded_row=False,
                                             shroud_axial_clearance_over_blade_length=None):
        """A method to calculate the total aerodynamic loss coefficient of a stator or rotor blade row.

        :param str blade_row: Blade row being evaluated, either "stator" or "rotor".
        :param float inlet_angle: Blade-row inlet flow angle measured from the positive vertical direction (rad).
        :param float outlet_angle: Blade-row outlet flow angle measured from the positive vertical direction for the
         stator and the negative vertical direction for the rotor (rad).
        :param float t_TE_over_pitch: Projection of the trailing-edge thickness on tangential axis over blade pitch (-).
        :param float roughness_over_chord: Equivalent sand roughness over blade chord (-).
        :param float roughness_over_hydraulic_diameter: Equivalent sand roughness over hydraulic diameter (-).
        :param float blade_length_over_mean_diameter: Blade length over mean turbine diameter (-).
        :param float chord_over_blade_length: Blade chord over blade length (-).
        :param float chord_over_pitch: Blade chord over blade pitch (-).
        :param float outlet_Mach_number: Blade-row outlet Mach number (-).
        :param float or integer Reynolds_number: Blade-row Reynolds number (-).
        :param float speed_parameter: Blade-row speed parameter for the fanning-loss graph (-).
        :param float blade_velocity_ratio: Blade-row inlet-to-outlet velocity ratio (-).
        :param float axial_clearance_over_blade_length: Axial clearance over blade length (-).
        :param bool shrouded_row: Whether the blade row has a shroud. By default, False.
        :param float or None shroud_axial_clearance_over_blade_length: Axial shroud clearance over blade length,
            required for a shrouded row. By default, None.
        :return: Total blade-row aerodynamic loss coefficient (-).
        :rtype: float
        """

        # Aerodynamic loss coefficient of the blade row has the form of zeta = zeta_p + zeta_f + zeta_rest
        # First calculate profiles loss, zeta_p = chi_R * chi_M * zeta_p0 + zeta_h + zeta_C
        # Calculate chi_R
        chi_R = self.calculate_chi_R(Reynolds_number, roughness_over_chord)  # -
        # Calculate chi_M
        chi_M = self.calculate_chi_M(outlet_Mach_number, blade_row)  # -
        # Calculate zeta_p0
        zeta_p0 = self.calculate_zeta_p0(inlet_angle * 180 / np.pi, outlet_angle * 180 / np.pi)  # -
        corrected_zeta_p0 = chi_R * chi_M * zeta_p0  # -
        # Calculate zeta_h
        zeta_h = self.calculate_zeta_h(t_TE_over_pitch/corrected_zeta_p0, t_TE_over_pitch, Reynolds_number)  # -
        # Calcuale zeta C
        zeta_C = (t_TE_over_pitch / (1 - t_TE_over_pitch))**2 * np.sin(outlet_angle)**2  # -
        # Sum individual coefficients
        zeta_p = chi_R * chi_M * zeta_p0 + zeta_h + zeta_C  # -

        # Calculate fanning loss coefficients
        zeta_f = self.calculate_zeta_f(blade_length_over_mean_diameter, speed_parameter)  # -

        # Calculate approximation of the zeta_p over the whole blade
        integrated_zeta_p = zeta_p + zeta_f  # -

        # Calculate the residual loss due to endwall effects. First calculate critical length ratio.
        if blade_row == "stator":
            coefficient = 7  # -
        else:
            coefficient = 10 if self.rotor in ("impulse_low_M", "impulse_high_M") else 7  # -
        critical_blade_length_to_pitch_ratio = np.sqrt(integrated_zeta_p) * coefficient  # -
        blade_length_to_pitch_ratio = chord_over_pitch / chord_over_blade_length  # -
        # The residual loss calculations depend on if the blade_length_to_pitch_ratio is above or below the critical
        # ratio. However, both branches require zeta_a.
        if shrouded_row:
            zeta_a = (0.04 / np.sin(outlet_angle)) * shroud_axial_clearance_over_blade_length  # -
        else:
            # If the blade row is the rotor, zeta_a is zero, because there is no blade row afterwards
            if blade_row == "rotor": zeta_a = 0  # -
            # If the blade row is the stator, calculate zeta_a
            elif blade_row == "stator":
                Reynolds_number_for_c_f = Reynolds_number * (2 / chord_over_blade_length)
                c_f = self.calculate_c_f(Reynolds_number_for_c_f, roughness_over_hydraulic_diameter)  # -
                zeta_a = (c_f / np.sin(outlet_angle)) * (1 + blade_length_over_mean_diameter) * \
                         axial_clearance_over_blade_length  # -
        # Also calculate F factor which is needed for both branches
        flow_angle_change = 180 - (inlet_angle + outlet_angle) * 180 / np.pi  # deg
        F_coefficient = self.calculate_F(flow_angle_change, blade_velocity_ratio)  # -
        # Calculate the residual loss if length ratio is above the critical value
        if blade_length_to_pitch_ratio >= critical_blade_length_to_pitch_ratio:
            zeta_rest = zeta_a + (zeta_p / zeta_p0) * F_coefficient / blade_length_to_pitch_ratio  # -
        # Calculate the residual loss if length ratio is below the critical value
        else:
            # Choose A coefficient. According to Traupel, it has avalue of 0.02 for strongly accelerating cascades,
            # or 0.035 for impulse rotors.
            if blade_row == "stator":
                A_coefficient = 0.02  # -
            else:
                A_coefficient = 0.035 if self.rotor in ("impulse_low_M", "impulse_high_M") else 0.02  # -
            zeta_rest = zeta_a + (zeta_p / zeta_p0) * F_coefficient / critical_blade_length_to_pitch_ratio \
                        + A_coefficient * chord_over_pitch * \
                        (1 / blade_length_to_pitch_ratio - 1 / critical_blade_length_to_pitch_ratio)  # -

        # Calculate and return total aerodynamic loss coefficient
        zeta_total_aerodynamic = integrated_zeta_p + zeta_rest  # -
        return zeta_total_aerodynamic

    def calculate_shrouded_rotor_clearance_loss(self, tip_diameter, seal_clearance, teeth_number, teeth_spacing,
                                                admission_fraction, p_2_over_p_1, mdot, p_1, rho_1,
                                                design_isentropic_work_coefficient_ss,
                                                operation_isentropic_work_coefficient_ss):
        """A method to calculate shrouded-rotor clearance loss and the flow leaking through its half-labyrinth seal.

        :param float tip_diameter: Rotor tip diameter (m).
        :param float seal_clearance: Radial clearance of the seal (m).
        :param int or float teeth_number: Number of seal teeth (-).
        :param float teeth_spacing: Axial distance between seal teeth (m).
        :param float admission_fraction: Admitted fraction of the rotor circumference (-).
        :param float p_2_over_p_1: Seal outlet-to-inlet pressure ratio (-).
        :param float mdot: Main mass flow rate (kg/s).
        :param float p_1: Seal inlet pressure (Pa).
        :param float rho_1: Seal inlet fluid density (kg/m^3).
        :param float design_isentropic_work_coefficient_ss: Static-to-static isentropic loading coefficient at design
         point (-).
        :param float operation_isentropic_work_coefficient_ss: Static-to-static isentropic loading coefficient at
         operating point (-).
        :return: Shrouded-rotor clearance loss coefficient (-) and seal leakage mass flow rate (kg/s), respectively.
        :rtype: tuple[float, float]
        """

        # First calculate seal flow area
        A_seal = tip_diameter * np.pi * seal_clearance  # m^2
        # Calculate flow coefficient phi
        phi = self.calculate_phi(p_2_over_p_1, teeth_spacing / seal_clearance, teeth_number)  # -
        # Calculate leak rate
        mdot_leak_rotor = admission_fraction * A_seal * phi * np.sqrt(p_1 * rho_1)  # kg/s
        # If it is higher than the main flow, raise an error
        if mdot_leak_rotor >= mdot:
            raise ValueError("Massflow through the seal is higher than through the rotor.")
        # Calculate relative leak flow rate, mu coefficient
        mu = mdot_leak_rotor / (mdot - mdot_leak_rotor)  # -
        # Calculate speed parameters
        v_operation = 1 / np.sqrt(2 * operation_isentropic_work_coefficient_ss)  # -
        v_design = 1 / np.sqrt(2 * design_isentropic_work_coefficient_ss)  # -
        # Calculate clearance loss and return it together with the leak rate
        zeta_clearance_rotor = mu * (1 - ((v_operation - v_design) / v_design)**2)  # -
        return zeta_clearance_rotor, mdot_leak_rotor


    def calculate_unshrouded_rotor_clearance_loss(self, normalized_inlet_velocity,
                                                  isentropic_work_coefficient_ss, clearance_over_blade_length,
                                                  chord_over_blade_length, tip_over_mean_diameter, isentropic_reaction,
                                                  outlet_angle, circumferential_velocity_change, axial_velocity):
        """A method to calculate the clearance loss coefficient of an unshrouded rotor.

        :param float normalized_inlet_velocity: Inlet velocity normalized by the blade velocity (-).
        :param float isentropic_work_coefficient_ss: Static-to-static isentropic loading coefficient of the stage (-).
        :param float clearance_over_blade_length: Radial clearance over blade length (-).
        :param float chord_over_blade_length: Blade chord over blade length (-).
        :param float tip_over_mean_diameter: Rotor tip diameter over mean turbine diameter (-).
        :param float isentropic_reaction: Isentropic reaction of the turbine stage according to
         Traupel's definition (-).
        :param float outlet_angle: Blade row outlet flow angle for rotor measured from the negative vertical
         direction (rad).
        :param float circumferential_velocity_change: Change in circumferential velocity across the rotor (m/s).
        :param float axial_velocity: Axial velocity used to normalize the circumferential velocity change (m/s).
        :return: Unshrouded rotor clearance loss coefficient (-).
        :rtype: float
        """

        # First calculate normalized clerance and K_sigma
        normalized_clearance = max(clearance_over_blade_length / chord_over_blade_length - 0.002, 0)  # -
        velocity_change_ratio = circumferential_velocity_change / axial_velocity  # -
        K_sigma = self.calculate_K_sigma(outlet_angle, velocity_change_ratio, normalized_clearance)  # -

        # Now calculate zeta_clearance_rotor
        zeta_clearance_rotor = \
            K_sigma * (2 * isentropic_reaction * isentropic_work_coefficient_ss + normalized_inlet_velocity**2) * \
            max(clearance_over_blade_length - 0.002 * chord_over_blade_length, 0) * tip_over_mean_diameter / \
            (2 * isentropic_work_coefficient_ss)  # -
        return max(zeta_clearance_rotor, 0)

    def calculate_admission_loss(self, admission_fraction, isentropic_work_coefficient_ss,
                                 flow_coefficient, blade_length_over_mean_diameter, blade_width_over_mean_diameter,
                                 outlet_angle, partial_admission_rotor):
        """A method to calculate the rotor partial-admission loss coefficient.

        :param float admission_fraction: Partial admission fraction of the rotor circumference (-).
        :param float isentropic_work_coefficient_ss: Static-to-static isentropic loading coefficient of the stage (-)
        calculated using the mean diameter.
        :param float flow_coefficient: Turbine flow coefficient (-)  calculated using the mean diameter..
        :param float blade_length_over_mean_diameter: Rotor blade length over mean turbine diameter (-).
        :param float blade_width_over_mean_diameter: Rotor blade width over mean turbine diameter (-).
        :param float outlet_angle: Blade row outlet flow angle for rotor, measured from the negative vertical direction
         (rad).
        :param str partial_admission_rotor: Rotor configuration, either "free" or "enclosed".
        :return: Partial-admission loss coefficient (-).
        :rtype: float
        """

        # Calculate C factor. If rotor is free:
        if partial_admission_rotor == "free":
            if self.rotor in ("impulse_low_M", "impulse_high_M"): correction = 0.8
            else: correction = 1
            C_coefficient = correction * (0.045 + 0.58 * blade_length_over_mean_diameter) * np.sin(outlet_angle)  # -
        # If rotor is partially enclosed:
        elif partial_admission_rotor == "enclosed":
            # Enclosed row formula works only for blade_length_over_mean_diameter < 0.125. However, for the lack of
            # other alternatives, it is also used beyond that limit. Partial admission turbines in rocket engines
            # usually have a low l/D, so such extrapolation is deemed acceptable.
            C_coefficient = 0.0095 + 0.55 * max(0.125 - blade_length_over_mean_diameter, 0)**2  # -
        # Calculate and return zeta_admission. It is assumed there is only one partial admission sector, which is
        # common in rocket engine turbines.
        if admission_fraction < 1:
            zeta_admission = \
                C_coefficient * (1 - admission_fraction) / \
                (admission_fraction * flow_coefficient * isentropic_work_coefficient_ss) + \
                0.21 * blade_width_over_mean_diameter / \
                (admission_fraction * np.sqrt(isentropic_work_coefficient_ss))  # -
        elif admission_fraction == 1:
            zeta_admission = 0
        return zeta_admission

    def calculate_disk_friction_loss(self, admission_fraction, hub_over_mean_diameter, hub_diameter_over_blade_length,
                                     isentropic_work_coefficient_ss, flow_coefficient, Reynolds_number):
        """A method to calculate the rotor disk-friction loss coefficient.

        :param float admission_fraction: Admitted fraction of the rotor circumference (-).
        :param float hub_over_mean_diameter: Rotor hub diameter over mean turbine diameter (-).
        :param float hub_diameter_over_blade_length: Rotor hub diameter over blade length (-).
        :param float isentropic_work_coefficient_ss: Static-to-static isentropic loading coefficient of the stage (-)
         calculated using the mean diameter.
        :param float flow_coefficient: Turbine flow coefficient (-) calculated using the mean diameter.
        :param float or integer Reynolds_number: Reynolds number used for the disk-friction coefficient (-).
        :return: Rotor disk-friction loss coefficient (-).
        :rtype: float
        """

        # Calculate C_M
        C_M = self.calculate_C_M(Reynolds_number)  # -
        # Calculate and return zeta_friction
        zeta_friction = 1.27 * C_M * hub_over_mean_diameter**4 * hub_diameter_over_blade_length / \
                        (admission_fraction * flow_coefficient * isentropic_work_coefficient_ss)  # -
        return zeta_friction

    def calculate_incidence_losses(self, incidence_angle, inlet_to_outlet_velocity_ratio, inlet_Mach_number):
        """A method to calculate blade-row incidence loss with a Mach-number correction to the incidence angle.

        :param float incidence_angle: Deviation of the inlet flow angle from the design angle (rad).
        :param float inlet_to_outlet_velocity_ratio: Inlet-to-outlet blade-row velocity ratio (-).
        :param float inlet_Mach_number: Blade-row inlet Mach number; correction uses values from 0.5 to 0.8 (-).
        :return: Incidence loss coefficient (-).
        :rtype: float
        """

        # First calculate corrected incidence angle for Mach number. Clip inlet Mach number so that correction is always
        # done over a valid range of Mach numbers (from 0.5 to 0.8)
        inlet_Mach_number = max(0.5, min(0.8, inlet_Mach_number))  # -
        corrected_incidence_angle = incidence_angle / (2 * (1 - inlet_Mach_number))  # rad
        # Get z coefficient
        z_coefficient = self.calculate_z(corrected_incidence_angle * 180 / np.pi)  # -
        # Calculate zeta_incidence and return it
        zeta_incidence = z_coefficient * inlet_to_outlet_velocity_ratio**2  # -
        return zeta_incidence

    def calculate_entropy_increase(self, analysis_results, turbine_geometry, blade_row_results):
        """A method to calculate entropy generation from stator and rotor losses.

        Exact definitions of the variables in the input dictionaries are in Turbine1D.__init__.

        :param dict analysis_results: Turbine flow analysis results that account for all losses.
        :param dict turbine_geometry: Stator, rotor, and seal geometry and roughness.
        :param dict blade_row_results: Blade-row flow analysis results that account only for profile losses.
        :return: Stator entropy increase, rotor blade-row entropy increase, additional rotor entropy increase
                 (J/(kg K)), and a dictionary of the calculated losses and leakage mass flow rate.
        :rtype: tuple[float, float, float, dict]
        """

        # Get inlet and outlet flow angles (measured from meridional axis) and change them to Traupel convention
        # (measured from circumferential axis). Traupel measures stator outlet and rotor inlet
        # from the positive circumferential direction, but rotor outlet from the negative direction.
        alpha_1 = np.pi / 2 - analysis_results["alpha_1"]  # rad
        beta_1 = np.pi / 2 - analysis_results["beta_1"]  # rad
        # Rotor outlet angle before additional losses occur. This represents local flow angle at Euler diameter for the
        # flow immediately after leaving rotor blade profile.
        beta_2 = np.pi / 2 + blade_row_results["beta_2_blade"]  # rad

        # Also calculate reaction according to Traupel's definition
        delta_h_stator_ideal = analysis_results["delta_h_stator_ideal"]
        delta_h_rotor_ideal = analysis_results["delta_h_rotor_ideal"]
        R_h_Traupel = delta_h_rotor_ideal / (delta_h_rotor_ideal + delta_h_stator_ideal)

        # Calculate aerodynamic loss coefficient for the stator. First retrieve some variables and calculate
        # normalized inputs. Quantities in analysis_results and blade_row_results at station 1 are the same,
        # so analysis results are used here.
        t_TE_stator = turbine_geometry["t_TE_stator"]  # m
        p_stator = turbine_geometry["p_stator"]  # m
        t_TE_over_p_stator = t_TE_stator / p_stator  # -
        c_stator = turbine_geometry["c_stator"]  # m
        c_over_p_stator = c_stator / p_stator  # -
        k_s = turbine_geometry["sand_grain_roughness"]  # m
        k_s_over_c_stator = k_s / c_stator  # -
        l_stator = turbine_geometry["l_stator"]  # m
        c_over_l_stator = c_stator / l_stator  # -
        hydraulic_diameter_stator = 2 * l_stator  # m
        k_s_over_d_h_stator = k_s / hydraulic_diameter_stator  # -
        D_m = turbine_geometry["D_mean"]  # m
        l_stator_over_D_m = l_stator / D_m  # -
        s_ax = turbine_geometry["s_ax"]  # m
        s_ax_over_l_stator = s_ax / l_stator  # -
        M_s1 = analysis_results["M_s1"]  # -
        Re_s1 = analysis_results["Re_s1"]  # -
        psi_ss_ideal = analysis_results["psi_ss_ideal"]  # -
        speed_parameter = 1 / np.sqrt(2 * psi_ss_ideal)  # -
        zeta_aerodynamic_stator = \
            self.calculate_blade_row_aerodynamic_loss(blade_row="stator", inlet_angle=np.pi/2, outlet_angle=alpha_1,
                                                      t_TE_over_pitch=t_TE_over_p_stator,
                                                      roughness_over_chord=k_s_over_c_stator,
                                                      roughness_over_hydraulic_diameter=k_s_over_d_h_stator,
                                                      blade_length_over_mean_diameter=l_stator_over_D_m,
                                                      chord_over_pitch=c_over_p_stator,
                                                      chord_over_blade_length=c_over_l_stator,
                                                      outlet_Mach_number=M_s1, Reynolds_number=Re_s1,
                                                      speed_parameter=speed_parameter, blade_velocity_ratio=0,
                                                      axial_clearance_over_blade_length=s_ax_over_l_stator)  # -
        # Calculate incidence loss. For the stator, it is assumed to be zero.
        zeta_incidence_stator = 0  # -
        # Calculate dissipated enthalpy in the stator
        eta_stator = 1 - (zeta_aerodynamic_stator + zeta_incidence_stator)  # -
        v_1 = analysis_results["v_1"]  # m/s
        delta_h_loss_stator = ((1 - eta_stator) / eta_stator) * v_1**2 / 2  # J/kg
        # Calculate entropy generation in the stator
        gas = analysis_results["gas"]
        T_1 = analysis_results["T_1"]  # K
        delta_s_stator = -gas.Cp * np.log(1 - delta_h_loss_stator / (gas.Cp * T_1))  # J/(kg K)

        # Now calculate aerodynamic loss coefficient for the rotor
        t_TE_rotor = turbine_geometry["t_TE_rotor"]  # m
        p_rotor = turbine_geometry["p_rotor"]  # m
        t_TE_over_p_rotor = t_TE_rotor / p_rotor  # -
        c_rotor = turbine_geometry["c_rotor"]  # m
        c_over_p_rotor = c_rotor / p_rotor  # -
        k_s_over_c_rotor = k_s / c_rotor  # -
        l_rotor = turbine_geometry["l_rotor"]  # m
        c_over_l_rotor = c_rotor / l_rotor  # -
        hydraulic_diameter_rotor = 2 * l_rotor  # m
        k_s_over_d_h_rotor = k_s / hydraulic_diameter_rotor  # -
        l_rotor_over_D_m = l_rotor / D_m  # -
        s_ax_over_l_rotor = s_ax / l_rotor  # -
        M_r2_blade = blade_row_results["M_r2_blade"]  # -
        Re_r2_blade = blade_row_results["Re_r2_blade"]  # -
        shrouded_rotor = turbine_geometry["shrouded_rotor"]
        if shrouded_rotor:
            s_ax_shroud = turbine_geometry["s_ax_shroud"]  # m
            s_ax_shroud_over_l_rotor = s_ax_shroud / l_rotor  # -
        else:
            s_ax_shroud_over_l_rotor = None
        w_1_blade = blade_row_results["w_1_blade"]  # m/s
        w_2_blade = blade_row_results["w_2_blade"]  # m/s
        velocity_ratio_rotor = w_1_blade / w_2_blade  # -
        zeta_aerodynamic_rotor = \
            self.calculate_blade_row_aerodynamic_loss(blade_row="rotor", inlet_angle=beta_1, outlet_angle=beta_2,
                                                      t_TE_over_pitch=t_TE_over_p_rotor,
                                                      roughness_over_chord=k_s_over_c_rotor,
                                                      roughness_over_hydraulic_diameter=k_s_over_d_h_rotor,
                                                      blade_length_over_mean_diameter=l_rotor_over_D_m,
                                                      chord_over_pitch=c_over_p_rotor,
                                                      chord_over_blade_length=c_over_l_rotor,
                                                      outlet_Mach_number=M_r2_blade, Reynolds_number=Re_r2_blade,
                                                      speed_parameter=speed_parameter,
                                                      blade_velocity_ratio=velocity_ratio_rotor,
                                                      axial_clearance_over_blade_length=s_ax_over_l_rotor,
                                                      shrouded_row=shrouded_rotor,
                                                      shroud_axial_clearance_over_blade_length=
                                                      s_ax_shroud_over_l_rotor)  # -
        # Calculate incidence loss for the rotor
        incidence = analysis_results["incidence"]  # rad
        M_r1_blade = blade_row_results["M_r1_blade"]  # -
        zeta_incidence_rotor = self.calculate_incidence_losses(incidence, velocity_ratio_rotor, M_r1_blade)  # -
        # Calculate dissipated enthalpy in the rotor for the blade row alone
        eta_rotor = 1 - (zeta_aerodynamic_rotor + zeta_incidence_rotor)  # -
        delta_h_loss_rotor = ((1 - eta_rotor) / eta_rotor) * w_2_blade**2 / 2  # J/kg

        # Calculate dissipated enthalpy in the rotor for the additional losses
        # Disk friction loss
        admission_fraction = turbine_geometry["admission_fraction"]  # -
        D_hub = turbine_geometry["D_hub"]  # m
        D_hub_over_D_mean = D_hub / D_m  # -
        D_hub_over_l_blade = D_hub / l_rotor  # -
        theta_2 = analysis_results["theta_2"]  # -
        # Disk friction and partial admission require coefficients normalized at the mean diameter.
        u_mean = analysis_results["omega"] * D_m / 2  # m/s
        psi_ss_ideal_mean = psi_ss_ideal * (analysis_results["u"] / u_mean)**2  # -
        theta_2_mean = theta_2 * analysis_results["u"] / u_mean  # -
        rho_1 = analysis_results["rho_1"]  # kg/m^3
        Re_disk = analysis_results["omega"] * D_hub**2 * rho_1 / (2 * gas.calculate_dynamic_viscosity(T_1))  # -
        zeta_disk_friction = self.calculate_disk_friction_loss(admission_fraction, D_hub_over_D_mean, D_hub_over_l_blade,
                                                           psi_ss_ideal_mean, theta_2_mean, Re_disk)  # -
        # Clearance loss
        # First consider the case if the rotor is unshrouded:
        s_r = turbine_geometry["s_r"]  # m
        D_tip = turbine_geometry["D_tip"]  # m
        if not shrouded_rotor:
            u = analysis_results["u"]  # m/s
            w_1_normalized = w_1_blade / u  # -
            s_r_over_l_rotor = s_r / l_rotor  # -
            D_tip_over_D_mean = D_tip / D_m  # -
            # Use blade row flow results to calculate remaining inputs
            v_ax = w_2_blade * np.cos(blade_row_results["beta_2_blade"])  # m/s
            delta_w_circumferential = abs(w_1_blade * np.sin(analysis_results["beta_1"]) - \
                                      w_2_blade * np.sin(blade_row_results["beta_2_blade"]))  # m/s
            zeta_clearance_rotor = \
                self.calculate_unshrouded_rotor_clearance_loss(w_1_normalized, psi_ss_ideal, s_r_over_l_rotor,
                                                               c_over_l_rotor, D_tip_over_D_mean, R_h_Traupel, beta_2,
                                                               delta_w_circumferential, v_ax)  # -
            mdot_leak = 0  # kg/s
        # Now consider the case if the rotor is shrouded:
        else:
            seal_teeth_number = turbine_geometry["seal_teeth_number"]  # -
            seal_teeth_spacing = turbine_geometry["seal_teeth_spacing"]  # m
            p_2 = analysis_results["p_2"]  # Pa
            p_1 = analysis_results["p_1"]  # Pa
            p_2_over_p_1 = p_2 / p_1  # -
            mdot = analysis_results["mdot_total"]  # kg/s
            psi_ss_ideal_design = analysis_results["psi_ss_ideal_design"]  # -
            zeta_clearance_rotor, mdot_leak = \
                self.calculate_shrouded_rotor_clearance_loss(D_tip, s_r, seal_teeth_number, seal_teeth_spacing,
                                                             admission_fraction, p_2_over_p_1, mdot, p_1, rho_1,
                                                             psi_ss_ideal_design, psi_ss_ideal)  # (-, kg/s)
        # Partial admission loss
        partial_admission_rotor = turbine_geometry["partial_admission_rotor"]
        c_rotor_over_D_m = c_rotor / D_m  # -
        zeta_admission = self.calculate_admission_loss(admission_fraction, psi_ss_ideal_mean, theta_2_mean,
                                                       l_rotor_over_D_m, c_rotor_over_D_m, beta_2,
                                                       partial_admission_rotor)  # -
        # Sum these losses and get the sum dissipated enthalpy
        zeta_rotor_additional = zeta_disk_friction + zeta_admission + zeta_clearance_rotor  # -
        delta_h_ideal = analysis_results["delta_h_ideal"]  # J/kg
        delta_h_loss_rotor_additional = zeta_rotor_additional * delta_h_ideal  # J/kg

        # Calculate the sum of dissipated enthalpy in the rotor blade row and total work lost due to additional losses
        delta_h_loss_rotor_sum = delta_h_loss_rotor + delta_h_loss_rotor_additional
        # The entropy formulas below compare static enthalpies at the same outlet pressure.
        # delta_h_loss_rotor is a static enthalpy loss, but delta_h_loss_rotor_additional
        # is a work loss (an outlet total enthalpy increment), so their sum cannot be used directly.
        # The reconstructed blade-row outlet includes only aerodynamic losses. The stage
        # outlet also includes additional losses. Their pressures match, but their absolute
        # velocities can differ. Since h_t = h + v**2 / 2, the static enthalpy increment due to additional losses is
        # h_2 - h_2_blade = delta_h_loss_rotor_additional + outlet_kinetic_energy_blade - outlet_kinetic_energy_stage.
        # Adding this increment to the aerodynamic static loss gives the difference between stage outlet's
        # static enthalpy and the rotor's isentropic outlet static enthalpy. This difference can be used to calculate
        # total entropy increase through the rotor.
        outlet_kinetic_energy_blade = blade_row_results["v_2_blade"]**2 / 2  # J/kg
        outlet_kinetic_energy_stage = analysis_results["v_2"]**2 / 2  # J/kg
        delta_h_loss_rotor_sum_static = \
            delta_h_loss_rotor_sum + outlet_kinetic_energy_blade - outlet_kinetic_energy_stage # J/kg
        # Calculate total entropy increase through the rotor
        T_2 = analysis_results["T_2"]  # K
        delta_s_rotor_sum = -gas.Cp * np.log(1 - delta_h_loss_rotor_sum_static / (gas.Cp * T_2))  # J/(kg K)
        # Calculate entropy increase for the blade row alone such that it is consistent with the blade row flow state
        # at convergence
        T_2_blade = blade_row_results["T_2_blade"]
        delta_s_rotor = -gas.Cp * np.log(1 - delta_h_loss_rotor / (gas.Cp * T_2_blade))
        # Calculate entropy increase due to additional losses.
        delta_s_rotor_additional = delta_s_rotor_sum - delta_s_rotor  # J/(kg K)

        # Pack Traupel loss analysis results into a dictionary.
        Traupel_loss_analysis_results = {
            "zeta_aerodynamic_stator": zeta_aerodynamic_stator,  # Stator blade-row aerodynamic loss coefficient, -
            "zeta_incidence_stator": zeta_incidence_stator,  # Stator incidence loss coefficient, -
            "zeta_aerodynamic_rotor": zeta_aerodynamic_rotor,  # Rotor blade-row aerodynamic loss coefficient, -
            "zeta_incidence_rotor": zeta_incidence_rotor,  # Rotor incidence loss coefficient, -
            "zeta_disk_friction": zeta_disk_friction,  # Rotor disk-friction loss coefficient, -
            "zeta_clearance_rotor": zeta_clearance_rotor,  # Rotor clearance loss coefficient, -
            "zeta_admission": zeta_admission,  # Rotor partial-admission loss coefficient, -
            "zeta_rotor_additional": zeta_rotor_additional,  # Sum of rotor disk-friction, clearance and admission
            # coefficients, -
            "delta_h_loss_stator": delta_h_loss_stator,  # Specific enthalpy dissipated in the stator, J/kg
            "delta_h_loss_rotor": delta_h_loss_rotor,  # Specific enthalpy dissipated in the rotor blade row, J/kg
            "delta_h_loss_rotor_additional": delta_h_loss_rotor_additional,  # Specific work lost due to
            # additional rotor losses, J/kg
            "delta_h_loss_rotor_sum": delta_h_loss_rotor_sum,  # The sum of dissipated enthalpy in the rotor blade row
            # and total work lost due to additional losses , J/kg
            "delta_h_loss_rotor_sum_static": delta_h_loss_rotor_sum_static,  # The sum of dissipated enthalpy in the
            # rotor blade row and static enthalpy increment through the rotor due to additional losses, J/kg
            "delta_s_stator": delta_s_stator,  # Specific entropy rise across the stator, J/(kg K)
            "delta_s_rotor": delta_s_rotor,  # Specific entropy rise across the rotor blade row alone, J/(kg K)
            "delta_s_rotor_additional": delta_s_rotor_additional,  # Specific entropy rise from additional rotor
            # losses, J/(kg K)
            "mdot_leak": mdot_leak,  # Mass flow leaking through the rotor shroud seal (zero if unshrouded), kg/s
        }

        # Return everything
        return delta_s_stator, delta_s_rotor, delta_s_rotor_additional, Traupel_loss_analysis_results
