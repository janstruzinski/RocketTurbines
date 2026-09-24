import warnings
import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from .LossModel import LossModel


class TraupelLossModel(LossModel):
    def __init__(self, extrapolation_method="linear", warn_on_extrapolation=True):
        """A class to calculate turbine losses with Traupel meanline loss model presented in
         "Thermische Turbomaschinen", which is very suitable for steam, supersonic turbines. All graphs are digitalized
          as ready to use interpolators in class properties. Class methods allow to calculate specific loss
           coefficients, which are defined as dissipated enthalpy over isentropic enthalpy drop through the rotor.

        :param str extrapolation_method: Method used outside the digitized data region, either "linear" or "closest".
            The latter returns the value at the closest point in the digitized region.
        :param bool warn_on_extrapolation: Whether to warn on each out-of-bounds interpolation call.
        """

        # Use linear interpolation throughout and check the selected extrapolation option.
        interpolation_method = "slinear"
        extrapolation_method = extrapolation_method.lower()
        if extrapolation_method not in ("linear", "closest"):
            raise ValueError("extrapolation_method must be either 'linear' or 'closest'.")
        self.interpolation_method = interpolation_method
        self.extrapolation_method = extrapolation_method
        self.warn_on_extrapolation = warn_on_extrapolation

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

    def calculate_entropy_increase(self, analysis_results, turbine_geometry):
        ...

    def calculate_blade_row_aerodynamic_loss(self):
        ...

    def calculate_clearance_loss(self):
        ...

    def calculate_partial_admission_loss(self):
        ...

    def calculate_disk_friction_loss(self):
        ...

    def calc_incidence_losses(self):
        ...
