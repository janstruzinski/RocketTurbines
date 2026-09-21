from .LossModel import LossModel


class TraupelLossModel(LossModel):
    def __init__(self):
        """A class to calculate turbine losses with Traupel meanline loss model presented in
         "Thermische Turbomaschinen", which is very suitable for steam, supersonic turbines. All graphs are digitalized
          as ready to use interpolators in class properties. Class methods allow to calculate specific loss
           coefficients, which are defined as dissipated enthalpy over isentropic enthalpy drop through the rotor."""

        # AERODYNAMIC LOSS: PROFILE LOSS DATA
        # Data for chi_R

        # Data for chi_M

        # Data for zeta_p0

        # Data for zeta_h

        # Data for zeta_C

        # AERODYNAMIC LOSS: FANNING LOSS DATA
        # Data for zeta_f

        # AERODYNAMIC LOSS: ENDWALL & SECONDARY FLOWS LOSS DATA
        # Data for F factor

        # Data for c_f factor

        # CLEARANCE LOSS DATA
        # Data for K_sigma

        # DISK FRICTION LOSS DATA
        # Data for C_M
        ...

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
