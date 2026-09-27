class LossModel:
    """A base class defining the interface for turbine loss models.
    """

    def calculate_entropy_increase(self, analysis_results, turbine_geometry, blade_row_results):
        """A method to calculate the stator, rotor and additional rotor specific entropy increases.

        :param dict analysis_results: Dictionary with turbine analysis results.
        :param dict turbine_geometry: Dictionary with turbine geometry.
        :param dict blade_row_results: Dictionary with results for the blade rows alone without additional rotor losses.
        :return: Stator, rotor and additional rotor specific entropy increases (J/kg/K), respectively.
        :rtype: tuple[float, float, float]
        """

        raise NotImplementedError("Loss model subclasses must implement calculate_entropy_increase().")
