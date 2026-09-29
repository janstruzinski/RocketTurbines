import CoolProp.CoolProp as cp
from functools import lru_cache
import numpy as np


class IdealGas:
    def __init__(self, T_max, T_min, species, mass_fractions):
        """A class representing ideal, calorically perfect gas mixture. Specific heat is an average value from
        CoolProp for given temperature range.

        :param float T_max: Upper value (K) of the temperature range.
        :param float T_min: Lower value (K) of the temperature range.
        :param list species: List with CoolProp names of the species in the mixture.
        :param list mass_fractions: List with mass fractions of the species in the mixture.
        """

        # Check if mass fractions sum to 1.
        if abs(sum(mass_fractions) - 1) > 1e-12:
            raise ValueError("Mass fractions must sum to 1.")

        # Check if species are gasous, so that the specific heat is calculated correctly later on.
        gas_phases = ("gas", "supercritical_gas")
        for f in species:
            if cp.PhaseSI("P", 1e5, "T", T_max, f) not in gas_phases or\
                    cp.PhaseSI("P", 1e5, "T", T_min, f) not in gas_phases:
                raise ValueError(f"{f} must be gas at T_max and T_min.")

        # First get mole fractions of the mixture
        molar_masses = [cp.PropsSI("M", f) for f in species]
        dummy = [mf / M for mf, M in zip(mass_fractions, molar_masses)]
        molar_fractions = [x / sum(dummy) for x in dummy]
        mixture = "&".join(f"{fluid}[{xi}]" for fluid, xi in zip(species, molar_fractions))
        self.species = tuple(species)
        self.molar_masses = tuple(molar_masses)
        self.molar_fractions = tuple(molar_fractions)

        # Get specific gas constant of the mixture
        self.R = cp.PropsSI("GAS_CONSTANT", mixture) / cp.PropsSI("M", mixture)  # J/(kg K)

        # Now get Cp and Cv of the mixture
        delta_h = cp.PropsSI("Hmass", "P", 1e5, "T|gas", T_max, mixture) -\
                  cp.PropsSI("Hmass", "P", 1e5, "T|gas", T_min, mixture)  # J/kg
        self.Cp = delta_h / (T_max - T_min)  # J/(kg K)
        self.Cv = self.Cp - self.R  # J/(kg K)

        # Calculate its specific heat ratio
        self.gamma = self.Cp / self.Cv  # -

    @lru_cache(maxsize=1024)
    def calculate_density(self, p, T):
        """A method to calculate density of the ideal gas.

        :param float p: Pressure (Pa) of the gas.
        :param float T: Temperature (K) of the gas.
        :return: Gas density (kg/m3).
        """

        return p / (self.R * T)

    @lru_cache(maxsize=1024)
    def calculate_sound_velocity(self, T):
        """A method to calculate sound velocity of the ideal gas.

        :param float T: Temperature (K) of the gas.
        :return: Sound velocity (m/s).
        """

        return np.sqrt(self.gamma * self.R * T)

    @lru_cache(maxsize=1024)
    def calculate_dynamic_viscosity(self, T, p=1e5):
        """A method to calculate dynamic viscosity of the gas mixture using Wilke's rule.

        :param float T: Temperature (K) of the gas.
        :param float p: Pressure (Pa) of the gas. By default, 1e5, since dynamic viscosity is independent of pressure.
        :return: Gas mixture dynamic viscosity (Pa s).
        """

        species_viscosities = [cp.PropsSI("VISCOSITY", "P", p, "T|gas", T, f) for f in self.species]
        mixture_viscosity = 0  # Pa*s
        for xi, mu_i, M_i in zip(self.molar_fractions, species_viscosities, self.molar_masses):
            denominator = sum(xj * (1 + (mu_i / mu_j)**0.5 * (M_j / M_i)**0.25)**2 / (8 * (1 + M_i / M_j))**0.5
                for xj, mu_j, M_j in zip(self.molar_fractions, species_viscosities, self.molar_masses))  # -
            mixture_viscosity += xi * mu_i / denominator  # Pa*s

        return mixture_viscosity
