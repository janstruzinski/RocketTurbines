class LossModel:
    def calculate_entropy_increase(self, analysis_results, turbine_geometry):
        """Calculate the stator, rotor and additional rotor entropy increases.

        :param dict analysis_results: Turbine analysis results.
        :param dict turbine_geometry: Turbine geometry.
        :return: Stator, rotor and additional rotor entropy increases, respectively.
        :rtype: tuple[float, float, float]
        """

        raise NotImplementedError("Loss model subclasses must implement calculate_entropy_increase().")
