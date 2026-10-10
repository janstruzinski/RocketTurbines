import CoolProp.CoolProp as cp
from functools import lru_cache
import numpy as np
from thermoprop import CEA

class IdealGas:
    def __init__(self, T_max, T_min, species, mass_fractions, cea_species=None, P_max=1e5, P_min=1e5):
        """A class representing ideal, calorically perfect gas mixture. Specific heat is an average value from
        CoolProp for given temperature range.

        :param float T_max: Upper value (K) of the temperature range.
        :param float T_min: Lower value (K) of the temperature range.
        :param list species: List with CoolProp names of the species in the mixture.
        :param list mass_fractions: List with mass fractions of the species in the mixture.
        :param list cea_species: Optional list with NASA CEA gas names in the same order as species. If omitted,
         species must also be valid CEA names, for example "H2O" instead of "Water".
         The user must ensure that the CoolProp species and CEA species lists identify the same chemical species
         in the same order, even when their names differ between the two databases. Chemical correspondence
         between the lists is not validated automatically.
        :param int or float P_max: Maximum pressure (Pa), usually turbine inlet pressure. Used to check if species are
         gaseous at provided T_max. By default, 1e5.
        :param int or float P_min: Minimum pressure (Pa), usually turbine outlet pressure. Used to check if species are
         gaseous at provided T_min. By default, 1e5.
        :raises ValueError: If temperatures are not finite and positive, T_max <= T_min, or composition lists
         are empty, inconsistent, or contain invalid fractions.
        """

        self._validate_positive(T_min, "T_min")
        self._validate_positive(T_max, "T_max")
        if T_max <= T_min:
            raise ValueError("T_max must be greater than T_min.")

        species = tuple(species)
        mass_fractions = tuple(mass_fractions)
        if not species or len(species) != len(mass_fractions):
            raise ValueError("species and mass_fractions must be nonempty and have the same length.")
        if any(not np.isfinite(mf) or mf < 0 for mf in mass_fractions):
            raise ValueError("Mass fractions must be finite and nonnegative.")
        # Check if mass fractions sum to 1.
        if abs(sum(mass_fractions) - 1) > 1e-12:
            raise ValueError("Mass fractions must sum to 1.")

        # CEA names are used only for viscosity; all thermodynamic properties still use CoolProp names.
        # If CEA species are not given, use CoolProp species. Also check if CEA species list given has the same number
        # and order as CoolProp species list.
        cea_species = tuple(species if cea_species is None else cea_species)
        if len(cea_species) != len(species):
            raise ValueError("cea_species must have the same length and order as species.")

        # Remove absent species before checking phase or transport-data availability.
        active_indices = [i for i, mf in enumerate(mass_fractions) if mf > 0]
        species = tuple(species[i] for i in active_indices)
        mass_fractions = tuple(mass_fractions[i] for i in active_indices)
        cea_species = tuple(cea_species[i] for i in active_indices)

        # Set minimum temperature limit for viscosity evaluation to 300K, as this is the lower bound
        # to which many transport polynomials in NASA CEA were fitted.
        viscosity_temperature_min = 300.0
        # Check if CEA species exist.
        for f in cea_species:
            if not CEA.has_species(f) or not CEA.is_gas(f):
                raise ValueError(f"{f!r} is not a NASA CEA gas species. Provide cea_species with the "
                                 "corresponding CEA gas names in the same order as species.")
            # Check if NASA CEA has transport data and viscosity fits for selected species.
            if not CEA.has_transport(f):
                raise ValueError(f"NASA CEA transport data are not available for {f!r}.")
            viscosity_ranges = CEA.transport_temperature_ranges(f, "viscosity")
            if not viscosity_ranges:
                raise ValueError(f"NASA CEA viscosity fits are not available for {f!r}.")
            # If so, for each species, select lower bound used for fitting as capping limit for viscosity evaluation.
            # The capping limit cannot be lower than 300K.
            viscosity_temperature_min = max(viscosity_temperature_min, min(low_T for low_T, _ in viscosity_ranges))
        self.cea_species = cea_species
        self._viscosity_temperature_min = viscosity_temperature_min

        # Check if species are gasous, so that the specific heat is calculated correctly later on.
        gas_phases = ("gas", "supercritical_gas")
        for f in species:
            if cp.PhaseSI("P", P_max, "T", T_max, f) not in gas_phases or\
                    cp.PhaseSI("P", P_min, "T", T_min, f) not in gas_phases:
                raise ValueError(f"{f} must be gas at T_max and T_min at P_max and P_min respectively.")

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

    @staticmethod
    def _validate_positive(value, name):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")

    @lru_cache(maxsize=1024)
    def calculate_density(self, p, T):
        """A method to calculate density of the ideal gas.

        :param float p: Pressure (Pa) of the gas.
        :param float T: Temperature (K) of the gas.
        :return: Gas density (kg/m3).
        :raises ValueError: If pressure or temperature is not finite and positive.
        """

        self._validate_positive(p, "Pressure")
        self._validate_positive(T, "Temperature")
        return p / (self.R * T)

    @lru_cache(maxsize=1024)
    def calculate_sound_velocity(self, T):
        """A method to calculate sound velocity of the ideal gas.

        :param float T: Temperature (K) of the gas.
        :return: Sound velocity (m/s).
        :raises ValueError: If temperature is not finite and positive.
        """

        self._validate_positive(T, "Temperature")
        return np.sqrt(self.gamma * self.R * T)

    @lru_cache(maxsize=1024)
    def calculate_dynamic_viscosity(self, T):
        """A method to calculate dynamic viscosity of the gas mixture using Wilke's rule and NASA transport equations
        from CEA.

        The minimum evaluation temperature is the greater of 300 K and the highest of the minimum fitted
        temperatures for the active CEA species. For example, H2O sets a limit of 373.2 K. Requests below this
        limit return viscosity evaluated at the limiting temperature. This limit applies to all calls.

        :param float T: Temperature (K) of the gas.
        :return: Gas mixture dynamic viscosity (Pa s).
        :raises ValueError: If pressure or temperature is not finite and positive.
        """

        self._validate_positive(T, "Temperature")
        # It is possible that during some internal calculations, when the flow solution is not yet converged, a
        # very small temperature is reached. Keep the 300 K floor, but respect higher lower bounds of
        # individual CEA viscosity fits, for example 373.2 K for H2O.
        T = max(T, self._viscosity_temperature_min)

        species_viscosities = [CEA.viscosity(f, T) for f in self.cea_species]
        mixture_viscosity = 0  # Pa*s
        for xi, mu_i, M_i in zip(self.molar_fractions, species_viscosities, self.molar_masses):
            denominator = sum(xj * (1 + (mu_i / mu_j)**0.5 * (M_j / M_i)**0.25)**2 / (8 * (1 + M_i / M_j))**0.5
                for xj, mu_j, M_j in zip(self.molar_fractions, species_viscosities, self.molar_masses))  # -
            mixture_viscosity += xi * mu_i / denominator  # Pa*s

        return mixture_viscosity
