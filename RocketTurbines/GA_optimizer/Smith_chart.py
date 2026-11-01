"""Efficiency maps for exploring turbine flow and work coefficients."""
from collections import Counter
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from ..Turbine1D import Turbine1D


def plot_Smith_chart(flow_coefficient_array, work_coefficient_array, efficiency_to_plot,
                     Turbine1D_arguments, gas, loss_model, *, stator_angle_constraint=None,
                     l_over_D_constraint=None, D_tip_constraint=None, l_blade_constraint=None):
    """Sweep turbine duty coefficients and plot the selected meanline efficiency as a Smith chart.

    Each grid point sizes a fresh Turbine1D with all other inputs fixed. Failed evaluations and efficiencies
    outside (0, 1] are not plotted. Designs outside any enabled engineering limit are treated the same way.

    :param array-like flow_coefficient_array: Increasing flow-coefficient sample values, with at least two entries.
     For example, numpy.linspace(0.25, 3., 21). These values form the horizontal axis.
    :param array-like work_coefficient_array: Increasing total-to-total work-coefficient sample values, with at
     least two entries. For example, numpy.linspace(1., 6., 21). These values form the vertical axis.
    :param string efficiency_to_plot: Meanline result key: "eta_tt" for isentropic total-to-total, "eta_ts" for
     isentropic total-to-static, "eta_poly_tt" for polytropic total-to-total, or "eta_poly_ts" for polytropic
      total-to-static efficiency. Contour labels and the colorbar display percentages.
    :param dict Turbine1D_arguments: Fixed arguments to Turbine1D.size_turbine, excluding flow_coefficient,
     work_coefficient_tt, gas and loss_model. Omitted optional arguments use the model defaults. This dictionary
      and its nested settings are not modified.
    :param IdealGas gas: Working-gas object supplied to every meanline evaluation.
    :param TraupelLossModel loss_model: Loss-model object supplied to every evaluation, retaining its settings.
    :param float or None stator_angle_constraint: Maximum signed stator-exit flow angle in degrees, using
     Turbine1D.analysis_results_at_design_point["alpha_1_deg"]. None (default) disables this limit.
    :param float or None l_over_D_constraint: Maximum rotor blade length divided by mean diameter (-), using
     Turbine1D.l_rotor / Turbine1D.D_mean. None (default) disables this limit.
    :param float or None D_tip_constraint: Maximum turbine tip diameter in metres, using Turbine1D.D_tip.
     None (default) disables this limit.
    :param float or None l_blade_constraint: Minimum rotor blade length in metres, using Turbine1D.l_rotor.
     None (default) disables this limit. Values equal to any of the four limits are accepted.
    :return: Dictionary with figure, axes, flow_coefficients, work_coefficients, efficiencies, best_design,
     best_efficiency and failure_reasons. Efficiencies are a masked array of fractions, indexed as [work, flow].
      best_design contains the two coefficients at the highest sampled efficiency; both best fields are None
       if no point succeeds. failure_reasons counts exact evaluation errors, constraint violations and unusable
        efficiencies. Only the first violated limit is counted for each excluded point.
        The plot remains open until its windows are closed, then the dictionary is returned. The sampled maximum
         is marked on the chart; its accuracy depends on grid resolution.
    :rtype: dict
    :raises ValueError: If either coefficient axis is not a one-dimensional array with at least two samples.
    :raises KeyError: If efficiency_to_plot is not one of the four supported result names.
    """
    # Translate the selected result name into the efficiency description used in the title and colorbar.
    efficiency_labels = {"eta_tt": "Isentropic total-to-total", "eta_ts": "Isentropic total-to-static",
                         "eta_poly_tt": "Polytropic total-to-total", "eta_poly_ts": "Polytropic total-to-static"}
    efficiency_label = efficiency_labels[efficiency_to_plot]
    # Convert the supplied coefficient sequences into numeric arrays for sizing and plotting.
    flow_coefficients = np.asarray(flow_coefficient_array, dtype=float)
    work_coefficients = np.asarray(work_coefficient_array, dtype=float)
    # If either axis cannot form a two-dimensional contour grid, report it before evaluating the turbine.
    if flow_coefficients.ndim != 1 or work_coefficients.ndim != 1 \
            or min(flow_coefficients.size, work_coefficients.size) < 2:
        raise ValueError("Provide at least two coefficient sample values per axis.")

    # Leave failed points as NaN and give every sizing call its own copy of the fixed model arguments.
    fixed_arguments = dict(Turbine1D_arguments)
    efficiencies = np.full((work_coefficients.size, flow_coefficients.size), np.nan)
    failure_reasons = Counter()
    # Sweep work coefficients one at a time; each work coefficient defines one row of the efficiency map.
    for work_index, work_coefficient in enumerate(work_coefficients):
        # At this work coefficient, evaluate every flow coefficient to fill the columns of the current row.
        for flow_index, flow_coefficient in enumerate(flow_coefficients):
            try:
                # Size a fresh turbine so each coefficient pair has its own meanline geometry and stored results.
                turbine = Turbine1D()
                turbine.size_turbine(gas=gas, loss_model=loss_model, flow_coefficient=flow_coefficient,
                                    work_coefficient_tt=work_coefficient, **deepcopy(fixed_arguments))
            except (ValueError, RuntimeError, ArithmeticError) as error:
                # Keep the exact construction error, but continue evaluating the remaining coefficient pairs.
                failure_reasons[f"{type(error).__name__}: {error}"] += 1
                continue
            # Read the completed meanline results before checking the optional engineering limits.
            results = turbine.analysis_results_at_design_point
            # If the stator angle exceeds its enabled limit, leave this point masked and move to the next design.
            if stator_angle_constraint is not None and results["alpha_1_deg"] > stator_angle_constraint:
                failure_reasons["stator_angle_constraint exceeded"] += 1
                continue
            # If blade length relative to mean diameter is too large, exclude this design from the chart.
            if l_over_D_constraint is not None and turbine.l_rotor / turbine.D_mean > l_over_D_constraint:
                failure_reasons["l_over_D_constraint exceeded"] += 1
                continue
            # If the tip diameter exceeds its enabled limit, exclude this design from the chart too.
            if D_tip_constraint is not None and turbine.D_tip > D_tip_constraint:
                failure_reasons["D_tip_constraint exceeded"] += 1
                continue
            # If the rotor blade is shorter than its enabled minimum, leave this point out of the chart.
            if l_blade_constraint is not None and turbine.l_rotor < l_blade_constraint:
                failure_reasons["l_blade_constraint not met"] += 1
                continue
            # Read the requested efficiency only after all enabled limits have been satisfied.
            selected_efficiency = results[efficiency_to_plot]
            # If a completed evaluation has a usable efficiency, retain it; otherwise leave this point masked.
            if selected_efficiency is not None and np.isfinite(selected_efficiency) and 0 < selected_efficiency <= 1:
                efficiencies[work_index, flow_index] = selected_efficiency
            else:
                failure_reasons["Nonfinite or out-of-range selected efficiency"] += 1

    # Now prepare to make the Smith diagram
    # Mask invalid samples without interpolating across failed designs or disconnected valid regions.
    efficiencies = np.ma.masked_invalid(efficiencies)
    valid = ~np.ma.getmaskarray(efficiencies)
    # A contour cell is usable only when all four surrounding samples have valid efficiencies.
    contour_cells = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]

    # Put flow coefficient on the horizontal axis and work coefficient on the vertical axis.
    figure, axes = plt.subplots(figsize=(9, 6), layout="compressed")
    axes.set(xlabel=r"Flow coefficient, $\phi$", ylabel=r"Work coefficient, $\psi_{tt}$",
             title=f"Smith chart: {efficiency_label.lower()} efficiency",
             xlim=(flow_coefficients[0], flow_coefficients[-1]),
             ylim=(work_coefficients[0], work_coefficients[-1]))
    # If at least one complete cell is valid, draw filled efficiency bands and labeled isolines.
    if contour_cells.any():
        minimum_efficiency, maximum_efficiency = float(efficiencies.min()), float(efficiencies.max())
        color_limits = [100 * minimum_efficiency, 100 * maximum_efficiency]
        flat_efficiency = np.isclose(minimum_efficiency, maximum_efficiency, rtol=0., atol=1e-12)
        # If efficiency is numerically constant, give the color scale a finite width without changing the data.
        if flat_efficiency:
            color_limits = [max(0., color_limits[0] - .5), min(100., color_limits[1] + .5)]
        # Use the same percentage levels for the filled bands, colorbar and efficiency isolines.
        levels = np.linspace(*color_limits, 16)
        contours = axes.contourf(flow_coefficients, work_coefficients, 100 * efficiencies,
                                 levels=levels, cmap="RdYlGn", corner_mask=False)
        figure.colorbar(contours, ax=axes, label=f"{efficiency_label} efficiency (%)")
        # If efficiency varies, add contour labels; a constant field has no distinct isolines to label.
        if not flat_efficiency:
            isolines = axes.contour(flow_coefficients, work_coefficients, 100 * efficiencies,
                                    levels=levels[1:-1:2], colors="0.25", linewidths=.6, corner_mask=False)
            axes.clabel(isolines, fmt="%.1f", fontsize=8)
    else:
        # Explain an empty contour map, including the case where valid points have no valid neighboring cells.
        message = "Too few adjacent valid designs for contours" if valid.any() else "No valid turbine evaluations"
        axes.text(.5, .5, message, transform=axes.transAxes, ha="center", va="center")

    # Leave the best-result fields empty when no usable turbine evaluation exists.
    best_design = best_efficiency = None
    # If any sample is valid, mark the highest sampled efficiency even when there are too few cells for contours.
    if valid.any():
        # Locate the maximum in the [work, flow] grid and retain its coefficient values and efficiency.
        work_index, flow_index = np.unravel_index(efficiencies.argmax(), efficiencies.shape)
        best_design = dict(flow_coefficient=float(flow_coefficients[flow_index]),
                           work_coefficient_tt=float(work_coefficients[work_index]))
        best_efficiency = float(efficiencies[work_index, flow_index])
        # Mark the actual sampled maximum separately from the interpolated efficiency contours.
        maximum_coordinates = (flow_coefficients[flow_index], work_coefficients[work_index])
        axes.scatter(*maximum_coordinates, facecolors="white", edgecolors="black", s=50, zorder=3)
        # Place the annotation toward the map interior so maxima at the domain edges remain readable.
        label_on_left = maximum_coordinates[0] > (flow_coefficients[0] + flow_coefficients[-1]) / 2
        label_above = maximum_coordinates[1] < (work_coefficients[0] + work_coefficients[-1]) / 2
        axes.annotate("max", maximum_coordinates, xytext=(-6 if label_on_left else 6, 6 if label_above else -6),
                      textcoords="offset points", ha="right" if label_on_left else "left",
                      va="bottom" if label_above else "top")
    # Add major and minor grids to help read the coefficient values from the efficiency map.
    axes.minorticks_on()
    axes.grid(which="major", alpha=.3)
    axes.grid(which="minor", alpha=.15, linestyle=":")
    # Display the completed diagram, then return its plot objects and sampled data for further inspection.
    plt.show()
    return dict(figure=figure, axes=axes, flow_coefficients=flow_coefficients, work_coefficients=work_coefficients,
                efficiencies=efficiencies, best_design=best_design, best_efficiency=best_efficiency,
                failure_reasons=dict(failure_reasons))
