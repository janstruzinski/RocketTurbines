import inspect
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from functools import partial
from numbers import Real as RealNumber
from threading import local
import numpy as np
from pymoo.core.mixed import MixedVariableGA, MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.core.variable import Integer, Real
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.operators.selection.rnd import RandomSelection
from ..Turbine1D import Turbine1D
from ..SupersonicTurbine import SupersonicTurbine
from . import scouting_functions


def optimize_turbine(gas, loss_model, Turbine1D_arguments, optimization_arguments, constraints=None, *,
                    mode="SupersonicTurbine", efficiency="total-to-total", efficiency_type="isentropic",
                    SupersonicTurbine_arguments=None, SupersonicStatorNozzle_arguments=None,
                    SupersonicRotorBlades_arguments=None,
                    n_threads=None, plot_axes=None, block_plot=True, population_size=16, scout_size=64,
                    perturbation_scales=(.035, .012), perturbation_standard_deviations=None, pymoo_arguments=None,
                    max_generations=None,
                    stagnation_generations=8, efficiency_tolerance=5e-4,
                    geometry_tolerance=1e-8, seed=1, verbose=True):
    """Maximize turbine efficiency using automatic prescouting and pymoo's mixed-variable genetic algorithm.

    Prescouting searches for a feasible design, then supplies local and global initial designs to GA.
    If scouting finds no feasible seed, GA starts from the design with the smallest constraint violation.
    SupersonicTurbine evaluations reuse the nearest compatible successful coupling from earlier batches,
    with one fallback initialization after a failed construction. Failed evaluations receive penalties.
     Scouting and proposal helpers are in GA_optimizer.scouting_functions.

    :param IdealGas gas: Working-gas object used by both turbine modes. Each worker receives its own copy.
    :param TraupelLossModel loss_model: Loss-model object used for meanline sizing. Its settings, including
     extrapolation and warning settings, are retained in the worker copies.
    :param dict Turbine1D_arguments: Fixed inputs to Turbine1D.size_turbine, excluding gas, loss_model and any names
     present in optimization_arguments. Required sizing inputs must be provided here or as bounded variables;
      omitted optional inputs use the model defaults. Fixed vector settings such as delta_s_estimate remain lists
       or tuples. s_ax_over_pitch_rotor is a fixed scalar here. In SupersonicTurbine mode, chord/pitch ratios, TE
        thicknesses and rotor_width_to_chord are fixed initial coupling estimates; constrain their final outputs.
         A single stator nozzle also solves admission internally, so its input is only an initial estimate.
    :param dict optimization_arguments: Bounded outer variables only, as [lower, upper] lists or tuples with
     strictly increasing bounds. Allowed names are RPM, work_coefficient_tt, flow_coefficient, no_blades_stator,
      no_blades_rotor, admission_fraction, separation_margin and reaction_isentropic_tt. Blade counts use integer
       variables. At least one bounded variable must remain active in the selected mode; empty searches are rejected.
        Separation margin is ignored in Turbine1D mode. Fixed values belong in the corresponding model dictionary.
         Inputs must not occur in both the fixed and bounded dictionaries. With one fixed stator nozzle, coupled mode
          uses admission bounds only to choose its initial estimate because admission is an internal unknown;
           another active variable is required in that case.
    :param dict or None constraints: Output names mapped to inclusive [lower, upper] ranges or boolean values.
     None at either end means no bound on that side. Names may refer to scalar meanline geometry, analysis,
      blade-row or loss results, and, in SupersonicTurbine mode, packing coefficients and thickness margins.
       Simple ratios such as "l_rotor/D_mean" are supported. Angles use the model's units; alpha_1_deg and
        other *_deg outputs use degrees. Lengths, including height_optimality, use metres. Default None adds no
         optional engineering constraints. Both meanline TE/pitch ratios are always strictly below 0.2;
          finite efficiency in (0, 1] is also required.
    :param string mode: Model to evaluate: "Turbine1D" or "SupersonicTurbine" (default).
    :param string efficiency: Efficiency basis: "total-to-total" (default) or "total-to-static".
    :param string efficiency_type: Reference process: "isentropic" (default) or "polytropic". Together with
     efficiency, selects eta_tt, eta_ts, eta_poly_tt or eta_poly_ts from the meanline results.
    :param dict or None SupersonicTurbine_arguments: Fixed SupersonicTurbine constructor settings, such as
     t_min_stator, t_min_rotor, stator_nozzle_chord_scale and print_user_warnings. A fixed separation_margin also
      belongs here; an optimized separation_margin belongs in optimization_arguments instead. Required constructor
       settings must be supplied in coupled mode; optional settings use the model defaults. None uses an empty
        dictionary. Ignored in Turbine1D mode. Meanline and blading dictionaries have their own separate inputs.
    :param dict or None SupersonicStatorNozzle_arguments: Fixed stator-blading settings, following
     SupersonicTurbine's dictionary interface. Must include contour_method in SupersonicTurbine mode;
      ignored in Turbine1D mode. Default None.
    :param dict or None SupersonicRotorBlades_arguments: Fixed rotor-blading settings, following
     SupersonicTurbine's dictionary interface. None uses the blading defaults; ignored in Turbine1D mode.
    :param integer or None n_threads: Positive number of model-evaluation threads, capped at the available
     logical CPU count. None (default) uses all available logical CPUs. A batch can contain fewer designs
      than threads. Plot updates and terminal output run on the calling thread.
    :param list or None plot_axes: Two numeric input names [x_name, y_name] for the live scatter plot.
     Coordinates are outer inputs, including when the final coupled values differ. Feasible designs are circles
      colored by the selected efficiency, with green at the top of the colorbar. Successfully constructed designs
       that violate constraints are squares on the same colormap and efficiency scale; failed constructions are omitted.
        The colorbar spans both classes; designs with unavailable or nonfinite efficiency are omitted.
         A legend below identifies both shapes. The best feasible point has a black outline and a "max" label.
          Updates occur during batches and after every completed GA generation. The status shows the phase/generation
           and indicates when no feasible designs exist. None (default) disables plotting. An interactive Matplotlib
            backend is needed for live display.
    :param bool block_plot: Keep the final plot open until its windows are closed before returning. Default True.
     False returns immediately after optimization while retaining the live plot and returned figure. Ignored when
      plotting is disabled, the plot was closed during optimization, or the backend is noninteractive. Time spent
       viewing the final plot is excluded from elapsed_seconds.
    :param integer population_size: GA population size, at least 2. Default 16. Half the initial population
     is near the scout leader and half uses global proposals. pymoo_arguments["pop_size"] overrides this.
    :param integer scout_size: Number of initial geometry-informed Sobol proposals, at least 2. Default 64.
     Scouting precedes GA and can require several evaluation batches; it is not a GA generation.
    :param tuple or list perturbation_scales: Nonempty sequence of nonnegative Gaussian standard deviations for
     continuous local proposals, expressed as fractions of each variable's bound span. Default (0.035, 0.012)
      alternates broad and tight proposals around the scout leader when initializing GA. Each local-proposal call
       starts at the first scale; additional scouting requests one local design per leader and uses that scale.
        These settings do not change global Sobol proposals or GA crossover/mutation.
    :param dict or None perturbation_standard_deviations: Optional absolute standard deviations by design-variable
     name, overriding the relative scales and integer defaults during local proposals. Units match each input:
      RPM uses rpm, coefficients use their own units, and blade counts use blades. Unspecified continuous inputs
       use perturbation_scales; unspecified blade counts use 1 stator nozzle or 3 rotor blades. Zero keeps
        that variable unchanged in local proposals. Bounds and integer rounding still apply. Fixed inputs are not
         perturbed. None (default) retains all previous perturbation settings; this dictionary is not modified.
    :param dict or None pymoo_arguments: Optional overrides of the tuned mixed-GA settings. None or {} retains
     population 16 (unless population_size is supplied), random parent selection, SBX crossover probability
      0.9/eta 15, polynomial mutation probability 1/eta 20, and integer rounding. Supported scalar overrides
       are pop_size, n_offsprings, crossover_probability, crossover_eta, mutation_probability and mutation_eta.
     Native operator overrides are selection, crossover, mutation, mating, survival, repair,
      eliminate_duplicates and advance_after_initial_infill. Mixed crossover/mutation maps use pymoo's Real
       and Integer types as keys. Do not combine a custom mating object with individual operator settings.
        Use this function's own seed, verbose and stopping inputs for run control.
    :param integer or None max_generations: Positive GA generation limit. Generation 1 evaluates the initial
     GA population; scouting does not count. None (default) imposes no generation limit.
    :param integer stagnation_generations: Improvement window, at least 2 generations. Default 8. GA stops
     when the best feasible efficiency improves by at most efficiency_tolerance over this window. Before
      feasibility, a constraint-violation improvement threshold of 1e-5 is used. Scouting uses the same
       window length with its violation threshold.
    :param float efficiency_tolerance: Positive absolute efficiency improvement threshold (-). Default 5e-4,
     equivalent to 0.05 percentage points. Used for GA stagnation after feasibility has been reached.
    :param float geometry_tolerance: Nonnegative roundoff tolerance (m) for the five packing/thickness
     coefficients only: height_optimality, width_optimality, t_TE_stator_min_margin, t_LE_rotor_min_margin,
      and t_TE_rotor_min_margin. Default 1e-8 m; zero requests exact comparisons. Other constraints do not
       receive this tolerance, including the two strict TE/pitch limits.
    :param integer or None seed: Seed for Sobol proposals, local perturbations and pymoo. Default 1.
     None requests an unseeded search.
    :param bool verbose: Print flushed start and completion lines for every scout iteration and GA
     generation. Default True. Completion rows give elapsed time (s), cumulative unique evaluations,
      successful/failed construction attempts, feasible designs, best feasible efficiency (%) and smallest
       normalized constraint violation (CV). Scouting is reported separately before GA starts. False
        silences this function's progress; model warnings are controlled separately.
    :returns: Dictionary containing success, best_turbine, best_design (outer inputs), best_results (actual
     outputs), best_efficiency (fraction), best_infeasible, termination, elapsed_seconds, statistics,
      evaluations, history and figure. If no feasible design exists, best_turbine, best_design, best_results
       and best_efficiency are None; best_infeasible describes the closest attempt. Construction counts
        include initialization retries, but exclude the internal meanline calls inside SupersonicTurbine.
         history stores one row per completed scout iteration or GA generation. figure is None when plotting
          is disabled; otherwise it is a Matplotlib Figure that the caller can save or keep open.
    :rtype: dict
    :raises TypeError: If dictionary inputs cannot be converted to dictionaries.
    :raises ValueError: For unsupported input/variable/constraint/optimizer names, overlapping fixed and bounded
     inputs, non-increasing optimization bounds, no active bounded variables, conflicting operators or unusable model
      outputs. Other numeric input requirements are left to Python, pymoo and the models.
    :raises KeyError: If a required input, model name or efficiency selection is missing or unknown.
    """
    # --- GA settings and objective selection ---

    # Copy overrides so constructing pymoo operators does not change the caller's dictionary.
    ga_arguments = deepcopy(dict(pymoo_arguments or {}))
    # Keep the previously tuned defaults for settings that the user did not override.
    operator_defaults = dict(crossover_probability=.9, crossover_eta=15., mutation_probability=1., mutation_eta=20.)
    supported_ga_arguments = {"pop_size", "n_offsprings", "selection", "crossover", "mutation", "mating",
                              "survival", "repair", "eliminate_duplicates", "advance_after_initial_infill"}
    unknown_ga_arguments = ga_arguments.keys() - supported_ga_arguments - operator_defaults.keys()
    # If an override name is unsupported, report it instead of letting pymoo silently ignore the setting.
    if unknown_ga_arguments:
        raise ValueError(f"Unsupported pymoo_arguments: {sorted(unknown_ga_arguments)}. "
                         "Use optimize_turbine's own seed, verbose and stopping inputs for run control.")
    # If custom mating and individual operators are both supplied, reject the ambiguous configuration.
    if ga_arguments.get("mating") is not None and ga_arguments.keys() & \
            ({"selection", "crossover", "mutation"} | operator_defaults.keys()):
        raise ValueError("Supply either a custom mating object or individual operator settings, not both.")
    # Extract scalar operator settings separately from native pymoo objects such as custom mating.
    operator_settings = {name: ga_arguments.pop(name, default) for name, default in operator_defaults.items()}
    # Take population and operator values as supplied; pymoo handles their numerical requirements.
    population_size = ga_arguments.pop("pop_size", population_size)
    model_class = {"Turbine1D": Turbine1D, "SupersonicTurbine": SupersonicTurbine}[mode]
    # Translate the two efficiency choices into the corresponding meanline result-dictionary key.
    efficiency_names = {("total-to-total", "isentropic"): "eta_tt",
                        ("total-to-static", "isentropic"): "eta_ts",
                        ("total-to-total", "polytropic"): "eta_poly_tt",
                        ("total-to-static", "polytropic"): "eta_poly_ts"}
    efficiency_name = efficiency_names[efficiency, efficiency_type]

    # --- Fixed model inputs and the finite set of bounded design variables ---

    # Derive supported names and defaults from the models; gas and blading dictionaries have separate inputs.
    shared_inputs = {"self", "gas", "loss_model"}
    blading_inputs = {"Turbine1D_arguments", "SupersonicStatorNozzle_arguments", "SupersonicRotorBlades_arguments"}
    meanline_parameters = {name: parameter for name, parameter in
                           inspect.signature(Turbine1D.size_turbine).parameters.items() if name not in shared_inputs}
    construction_parameters = {name: parameter for name, parameter in
                               inspect.signature(SupersonicTurbine).parameters.items()
                               if name not in shared_inputs | blading_inputs}
    # Read names and bounds here; fixed values are copied when constant_inputs is assembled below.
    fixed_meanline_inputs = dict(Turbine1D_arguments)
    fixed_construction_inputs = dict(SupersonicTurbine_arguments or {})
    optimization_inputs = dict(optimization_arguments)
    allowed_variables = {"RPM", "work_coefficient_tt", "flow_coefficient", "no_blades_stator", "no_blades_rotor",
                         "admission_fraction", "separation_margin", "reaction_isentropic_tt"}
    # If an optimization name is outside the supported set, reject it before scouting or construction begins.
    unknown_variables = optimization_inputs.keys() - allowed_variables
    if unknown_variables:
        raise ValueError(f"Unsupported optimization variables: {sorted(unknown_variables)}")
    # Check fixed inputs in their own model namespaces, leaving numeric requirements to the downstream models.
    fixed_dictionaries = [("Turbine1D_arguments", fixed_meanline_inputs, meanline_parameters)]
    # If coupled mode is selected, include fixed constructor settings and validate their names too.
    if mode == "SupersonicTurbine":
        fixed_dictionaries.append(("SupersonicTurbine_arguments", fixed_construction_inputs, construction_parameters))
    for label, inputs, model_parameters in fixed_dictionaries:
        unknown_inputs = inputs.keys() - model_parameters.keys()
        # If a fixed setting belongs to another model or is misspelled, identify its input dictionary.
        if unknown_inputs:
            raise ValueError(f"Unknown names in {label}: {sorted(unknown_inputs)}")
        overlapping_inputs = inputs.keys() & optimization_inputs.keys()
        # If an input is also optimized, require the caller to remove its fixed value.
        if overlapping_inputs:
            raise ValueError(f"Inputs are both fixed and optimized: {sorted(overlapping_inputs)}")
    # Unpack bounds explicitly; scalar values and incomplete pairs are no longer inferred as fixed inputs.
    optimization_bounds = {}
    for name, bounds in optimization_inputs.items():
        lower, upper = bounds
        # If the interval has no search width, require a fixed model input instead.
        if not lower < upper:
            raise ValueError(f"Optimization bounds for {name} must be strictly increasing; "
                             "put fixed values in model inputs.")
        optimization_bounds[name] = (lower, upper)
    # Assemble only settings for the selected mode, preserving model-parameter order for seeded proposal generation.
    parameters = meanline_parameters | (construction_parameters if mode == "SupersonicTurbine" else {})
    supplied_inputs = fixed_meanline_inputs | (fixed_construction_inputs if mode == "SupersonicTurbine" else {})
    variable_bounds = {name: optimization_bounds[name] for name in parameters if name in optimization_bounds}
    # If the stator always has one nozzle, admission is an inner unknown and only needs an initial estimate.
    if mode == "SupersonicTurbine" and fixed_meanline_inputs.get("no_blades_stator") == 1 \
            and "admission_fraction" in variable_bounds:
        admission_bounds = variable_bounds.pop("admission_fraction")
        supplied_inputs["admission_fraction"] = sum(admission_bounds) / 2
    # If no search dimension remains after mode-specific filtering, reject the call before constructing any turbines.
    if not variable_bounds:
        raise ValueError(f"optimization_arguments must contain at least one active variable with bounds for {mode}.")
    constant_inputs = {}
    integer_names = {"no_blades_stator", "no_blades_rotor"}
    for name, parameter in parameters.items():
        # If this input is optimized, its candidate value will be supplied by complete_design below.
        if name in variable_bounds:
            continue
        # Otherwise retain its fixed value or model default, including vector-valued settings unchanged.
        value = (supplied_inputs[name] if parameter.default is inspect.Parameter.empty
                 else supplied_inputs.get(name, parameter.default))
        constant_inputs[name] = deepcopy(value)
    # Read blading settings here; each construction attempt receives its own deep copy.
    stator_arguments = dict(SupersonicStatorNozzle_arguments or {})
    rotor_arguments = dict(SupersonicRotorBlades_arguments or {})

    # --- Constraint names and ranges ---

    # A misspelled constraint must fail before scouting, instead of invalidating every candidate.
    # Inspect an empty meanline object to obtain its geometry attributes and result-dictionary names.
    template = Turbine1D()
    # Start with geometry attributes, then add the scalar names stored inside the result dictionaries.
    geometry_names = {name for name, value in vars(template).items() if not isinstance(value, dict)}
    output_names = geometry_names.copy()
    for value in vars(template).values():
        # If this meanline attribute is a result dictionary, include its keys among the supported constraint names.
        if isinstance(value, dict):
            output_names.update(value)
    output_names.update(("chord_over_pitch_stator", "chord_over_pitch_rotor", "rotor_width_to_chord",
                         "s_ax_over_pitch_rotor"))
    # Packing margins exist only after SupersonicTurbine construction, unlike common meanline outputs.
    coefficient_names = {"height_optimality", "width_optimality", "t_TE_stator_min_margin",
                         "t_LE_rotor_min_margin", "t_TE_rotor_min_margin"}
    # If the full turbine model is used, also allow its packing coefficients and thickness margins.
    if mode == "SupersonicTurbine":
        output_names.update(coefficient_names)
    constraint_ranges = dict(constraints or {})
    for name in constraint_ranges:
        parts = name.split("/") if isinstance(name, str) else []
        # If a constraint names an unavailable output or an unsupported expression, report it before scouting.
        if not 1 <= len(parts) <= 2 or not set(parts) <= output_names:
            raise ValueError(f"Unknown constraint '{name}' for {mode}.")

    # nextafter makes the two mandatory upper bounds strict even at exactly 0.2.
    for ratio_name in ("t_TE_stator/p_stator", "t_TE_rotor/p_rotor"):
        interval = constraint_ranges.get(ratio_name, [None, None])
        constraint_ranges[ratio_name] = [interval[0], min(interval[1] if interval[1] is not None else .2,
                                                        np.nextafter(0.2, 0.))]
    # Count each finite range end as one inequality, plus one residual for overall efficiency/model validity.
    constraint_count = 1 + sum(1 if isinstance(interval, bool) else sum(bound is not None for bound in interval)
                             for interval in constraint_ranges.values())

    # --- Worker configuration and search state ---

    available_threads = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    # If the operating system exposes CPU affinity, use only the logical CPUs available to this process.
    if hasattr(os, "sched_getaffinity"):
        available_threads = min(available_threads, len(os.sched_getaffinity(0)))
    # Cap the requested worker count and create a dedicated random generator for this optimization run.
    worker_count = available_threads if n_threads is None else min(available_threads, n_threads)
    random_generator = np.random.default_rng(seed)
    variable_names = list(variable_bounds)
    thread_objects = local()
    # Keep successful coupling seeds separate from the cache of complete candidate evaluations.
    evaluations, history, coupling_archive, cache = [], [], [], {}
    # Construction counts include failed initialization retries; feasible designs are counted separately.
    statistics = dict(threads=worker_count, evaluations=0, cache_hits=0, construction_attempts=0,
                      successful_constructions=0, failed_constructions=0, feasible_designs=0,
                      failure_reasons=Counter(), evaluation_errors=Counter())
    best_record, best_turbine = None, None
    started = time.perf_counter()
    # --- Plot setup: create the axes once; update their artists during the search ---

    figure = None
    interactive_plot = False
    last_plot_state, plot_progress = None, ""
    # If the user supplied plot axes, create the figure, colorbar, maximum marker and progress text.
    if plot_axes is not None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D
        from matplotlib.ticker import PercentFormatter
        # Create the scatter artists once, then change their data instead of opening a new figure every generation.
        figure, axes = plt.subplots()
        # Share one color mapping so the same efficiency has the same color for circles and squares.
        efficiency_colormap = plt.get_cmap("RdYlGn")
        efficiency_normalization = Normalize(0., 1.)
        points = axes.scatter([], [], c=[], cmap=efficiency_colormap, norm=efficiency_normalization, marker="o", s=24,
                              label="Feasible designs", zorder=2)
        maximum_point = axes.scatter([], [], marker="o", facecolors="none", edgecolors="black",
                                     linewidths=1.6, s=95, zorder=3)
        # Color constructed, constraint-violating designs by their finite efficiency.
        infeasible_points = axes.scatter([], [], c=[], cmap=efficiency_colormap, norm=efficiency_normalization,
                                         marker="s", s=24, label="Infeasible designs", zorder=1)
        maximum_label = axes.annotate("max", (0., 0.), xytext=(6, 6), textcoords="offset points")
        # Hide the maximum label until the first feasible design supplies a valid location.
        maximum_label.set_visible(False)
        # Format efficiencies as percentages so the colorbar does not need a scientific-notation offset.
        figure.colorbar(points, ax=axes, format=PercentFormatter(xmax=1., decimals=2),
                        label=f"{efficiency_type} {efficiency} efficiency")
        axes.set(xlabel=plot_axes[0], ylabel=plot_axes[1], title=f"{mode}: constructed designs")
        # Use fixed shape samples so the legend remains visible even before either class has any points.
        legend_handles = [
            Line2D([], [], marker="o", linestyle="none", color=points.cmap(1.), label="Feasible designs"),
            Line2D([], [], marker="s", linestyle="none", color=points.cmap(1.), label="Infeasible designs")]
        axes.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
        # Enable both grid levels to make narrow feasible regions easier to inspect.
        axes.minorticks_on()
        axes.grid(which="major", alpha=.45)
        axes.grid(which="minor", alpha=.2, linestyle=":")
        axes.set_axisbelow(True)
        # Show the search domain immediately, even before the first feasible design exists.
        for axis_name, set_limits in zip(plot_axes, (axes.set_xlim, axes.set_ylim)):
            lower, upper = variable_bounds.get(axis_name, (constant_inputs.get(axis_name),) * 2)
            padding = .03 * (upper - lower) if upper != lower else max(abs(lower) * .05, .01)
            set_limits(lower - padding, upper + padding)
        # Add visible progress text even when no feasible points are available to plot.
        plot_status = axes.text(.02, .98, "Waiting for turbine evaluations...", transform=axes.transAxes,
                                va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        # Draw the axes and status before model workers start, so the initial figure is not blank.
        figure.tight_layout()
        figure.canvas.draw()
        # Remember whether this backend supports windows, so the final display can use the same check.
        interactive_plot = plt.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "pgf", "template")
        # If the backend supports interactive windows, show the figure without blocking the optimization.
        if interactive_plot:
            plt.show(block=False)

    # --- Translate optimizer inputs and turbine outputs ---

    def complete_design(candidate):
        """Combine the fixed settings with one candidate's real and integer variables."""
        return constant_inputs | {name: int(candidate[name]) if name in integer_names else float(candidate[name])
                                  for name in variable_names}

    def euler_diameter(design):
        """Calculate Euler diameter in metres without constructing a turbine."""
        specific_work = design["shaft_power"] / design["mdot"]
        # Convert blade speed and shaft speed to diameter using the same Euler relation as meanline sizing.
        return 60 * np.sqrt(specific_work / design["work_coefficient_tt"]) / (np.pi * design["RPM"])

    def output_values(turbine):
        """Collect scalar results under the names accepted by the constraints dictionary."""
        meanline = turbine if mode == "Turbine1D" else turbine.turbine_meanline
        # Begin with geometry, then merge the named thermodynamic, blade-row and loss results.
        values = {name: getattr(meanline, name) for name in geometry_names}
        for result in vars(meanline).values():
            # If the attribute contains analysis, blade-row or loss results, merge those named outputs into the values.
            if isinstance(result, dict):
                values.update(result)
        rotor_chord = meanline.c_rotor
        # If blading was constructed, add only its packing/thickness margins and measure rotor contour chord/pitch.
        if mode == "SupersonicTurbine":
            values.update({name: turbine.coefficients[name] for name in coefficient_names})
            # Measure the constructed contour, rather than its nearby meanline estimate.
            rotor_contour = turbine.rotor_shape.dimensionalize().uncorrected
            pressure_surface, suction_surface = rotor_contour.pressure_surface, rotor_contour.suction_surface
            rotor_chord = np.hypot(
                (pressure_surface.x[-1] + suction_surface.x[-1] - pressure_surface.x[0] - suction_surface.x[0]) / 2,
                (pressure_surface.y[-1] + suction_surface.y[-1] - pressure_surface.y[0] - suction_surface.y[0]) / 2)
        # Calculate each final ratio once, using the contour chord when blading was constructed.
        values.update(chord_over_pitch_stator=meanline.c_stator / meanline.p_stator,
                      chord_over_pitch_rotor=rotor_chord / meanline.p_rotor,
                      rotor_width_to_chord=meanline.width_rotor / meanline.c_rotor,
                      s_ax_over_pitch_rotor=meanline.s_ax / meanline.p_rotor)
        # Return only scalar diagnostics; gas, loss-model and other complex objects stay in the turbine object.
        return {name: value for name, value in values.items()
                if value is None or isinstance(value, (RealNumber, bool, np.bool_))}

    def constraint_residuals(values):
        """Convert ranges and booleans to pymoo inequalities: each residual must be <= 0."""
        residuals = []
        for name, interval in constraint_ranges.items():
            # Resolve either a direct output name or a simple ratio such as l_rotor/D_mean.
            numerator, *denominator = name.split("/")
            value = values[numerator] / values[denominator[0]] if denominator else values[numerator]
            # If a constrained output is unavailable or nonfinite, treat this output evaluation as invalid.
            if value is None or not np.isfinite(value):
                raise ValueError(f"Constraint output {name} is not finite.")
            # Store each resolved constraint value in the record so the user can inspect the actual constrained
            # quantity.
            values[name] = value
            # If the constraint expects a boolean, return zero only when both the type and requested value match.
            if isinstance(interval, bool):
                residuals.append(0. if isinstance(value, (bool, np.bool_)) and value == interval else 1.)
            else:
                # For a numeric constraint, express its lower and upper limits as separate signed residuals.
                # Normalize different units so a metre-valued packing error can guide selection too.
                scale = max([abs(bound) for bound in interval if bound is not None])
                scale = scale or (1e-3 if name in coefficient_names else 1.)
                # Apply the length tolerance only to the packing and thickness coefficients.
                tolerance = geometry_tolerance if name in coefficient_names else 0.
                # If a lower bound is present, positive residual means the result is too small.
                if interval[0] is not None:
                    residuals.append((interval[0] - value - tolerance) / scale)
                # If an upper bound is present, positive residual means the result is too large.
                if interval[1] is not None:
                    residuals.append((value - interval[1] - tolerance) / scale)
        return residuals

    # --- Coupling initialization: nearest compatible solution, then one fallback ---

    def initialization_choices(design):
        # The archive is frozen during each batch; thread scheduling cannot change its seeds.
        neighbors = []
        for previous in coupling_archive:
            previous_design = previous["design"]
            # One-nozzle coupling solves admission; multi-nozzle coupling solves stator TE thickness.
            # If the archived design uses the other nozzle branch, skip its incompatible fourth coupling unknown.
            if (previous_design["no_blades_stator"] == 1) != (design["no_blades_stator"] == 1):
                continue
            # If the nozzle counts differ too much, skip this seed to avoid transferring between unlike packing layouts.
            if max(previous_design["no_blades_stator"], design["no_blades_stator"]) / \
                    min(previous_design["no_blades_stator"], design["no_blades_stator"]) > 1.5:
                continue
            # Measure proximity in fractions of the input spans so RPM does not dominate dimensionless coefficients.
            distance = np.linalg.norm([(previous_design[name] - design[name]) / (upper - lower)
                                       for name, (lower, upper) in variable_bounds.items()
                                       if name != "admission_fraction" or design["no_blades_stator"] != 1])
            # If the normalized design distance is small enough, retain the archived solution as a possible warm start.
            if distance <= .6:
                neighbors.append((distance, previous["solution"]))

        # Transfer all five inner unknowns, including rotor width/chord.
        # Begin with the supplied initial estimates; replace them only if a compatible successful neighbor exists.
        primary = dict(design)
        # If compatible neighbors exist, use the closest solution to initialize all five coupling unknowns.
        if neighbors:
            solution = min(neighbors, key=lambda item: item[0])[1]
            fourth_name = "admission_fraction" if design["no_blades_stator"] == 1 else "t_TE_stator"
            primary.update(zip(("chord_over_pitch_stator", "chord_over_pitch_rotor", "t_TE_rotor",
                                fourth_name, "rotor_width_to_chord"), solution))
        # Prepare one lower-blockage fallback estimate for use only when the first construction attempt fails.
        fallback = dict(design, chord_over_pitch_stator=.15, chord_over_pitch_rotor=2.8,
                        t_TE_stator=.00025, t_TE_rotor=.00025, rotor_width_to_chord=1.)
        choices = [primary, fallback]
        for choice in choices:
            # Keep starting thicknesses safely below the mandatory 0.2 pitch limit.
            diameter = euler_diameter(choice)
            choice["t_TE_rotor"] = np.clip(choice["t_TE_rotor"], 0., .18 * np.pi * diameter / choice["no_blades_rotor"])
            choice["t_TE_stator"] = np.clip(choice["t_TE_stator"], 0., .18 * np.pi * diameter
                                             * choice["admission_fraction"] / choice["no_blades_stator"])
        return choices

    # --- Evaluate one turbine; construction failure is an optimization result ---

    def evaluate_design(candidate):
        # Combine this candidate with fixed settings, then record each attempted initialization independently.
        design = complete_design(candidate)
        record = dict(design=design, constructed=False, feasible=False, efficiency=None, values={}, attempts=[])
        # Loss models may contain mutable interpolation state: never share them between workers.
        # If this worker has not evaluated a design yet, create its private gas and loss-model copies once.
        if not hasattr(thread_objects, "gas"):
            thread_objects.gas, thread_objects.loss_model = deepcopy(gas), deepcopy(loss_model)
        # Meanline mode needs one attempt; coupled mode may use the nearest successful root and then a fallback.
        choices = initialization_choices(design) if mode == "SupersonicTurbine" else [design]
        turbine = None

        # Retry a failed coupled construction once. Do not retry a successful but infeasible design.
        for initial_design in choices:
            try:
                # Pass only meanline arguments into sizing; construction settings are supplied separately below.
                meanline_inputs = deepcopy({name: initial_design[name] for name in meanline_parameters})
                # If meanline mode was selected, size Turbine1D directly without constructing nozzle or rotor contours.
                if mode == "Turbine1D":
                    turbine = model_class()
                    turbine.size_turbine(
                        gas=thread_objects.gas, loss_model=thread_objects.loss_model, **meanline_inputs)
                else:
                    # For the coupled model, construct the blading wrapper and solve its inner meanline/blading
                    # coupling.
                    construction_inputs = {name: design[name] for name in construction_parameters}
                    turbine = model_class(
                        gas=thread_objects.gas, loss_model=thread_objects.loss_model,
                        Turbine1D_arguments=meanline_inputs, **construction_inputs,
                        SupersonicStatorNozzle_arguments=deepcopy(stator_arguments),
                        SupersonicRotorBlades_arguments=deepcopy(rotor_arguments))
                    turbine.couple_meanline_and_blading()
                record["attempts"].append(None)
                record["constructed"] = True
                # A successful construction ends the retry loop, even if its outer geometry constraints later fail.
                break  # Use the first converged root, regardless of its efficiency or feasibility.
            except (ValueError, RuntimeError, ArithmeticError, IndexError, KeyError) as error:
                # Save the exact construction error and continue to the remaining initialization, if one is available.
                record["attempts"].append(f"{type(error).__name__}: {error}")

        # A completed construction still has to satisfy efficiency and geometry constraints.
        # If construction completed, calculate efficiency and constraint residuals for this returned design.
        if record["constructed"]:
            try:
                # Read the selected efficiency from the final meanline results and check its physical range.
                values = output_values(turbine)
                selected_efficiency = float(values[efficiency_name])
                valid = np.isfinite(selected_efficiency) and 0 < selected_efficiency <= 1
                # If the design includes blading, save its coupling solution for later initializations.
                # The saved solution may initialize later designs even when this design violates an outer constraint.
                if mode == "SupersonicTurbine":
                    record["coupling_solution"] = np.asarray(turbine.x_solution).copy()
                # Add validity and geometry residuals to one vector that pymoo can use for feasibility-first selection.
                residuals = [0. if valid else 1000.] + constraint_residuals(values)
                record.update(values=values, efficiency=selected_efficiency, residuals=residuals)
            except (ValueError, TypeError, ArithmeticError, KeyError) as error:
                # Keep output-reading failures distinct from construction failures in the returned diagnostics.
                record["evaluation_error"] = f"{type(error).__name__}: {error}"
                record["residuals"] = [1000.] * constraint_count
        else:
            # Analytical pitch penalties retain useful direction even when construction fails.
            # Use the analytical rotor thickness/pitch ratio to distinguish failed designs without requiring a full
            # geometry.
            diameter = euler_diameter(design)
            thickness_ratio = design["t_TE_rotor"] * design["no_blades_rotor"] / (np.pi * diameter)
            penalty = 1000. + max(0., thickness_ratio - .2)
            record["residuals"] = [penalty] * constraint_count

        # Feasibility is separate from successful construction; CV sums only positive residuals.
        record["violation"] = float(np.maximum(record["residuals"], 0.).sum())
        record["feasible"] = record["constructed"] and record["violation"] == 0.
        return record, turbine

    def rank(record):
        """Prefer smaller violation, then greater efficiency when violations are equal."""
        return (record["violation"], -record["efficiency"] if record["efficiency"] is not None
                and np.isfinite(record["efficiency"]) else 1.)

    # --- Parallel batches, result caching and construction statistics ---

    def evaluate_batch(candidates, executor):
        nonlocal best_record, best_turbine
        # Use outer design coordinates as cache keys; fixed settings are shared throughout this run.
        pending = {}
        keys = [tuple(candidate[name] for name in variable_names) for candidate in candidates]

        # A repeated design reuses its record instead of constructing the same turbine again.
        for key, candidate in zip(keys, candidates):
            # If this design is neither cached nor already submitted, dispatch exactly one construction task.
            if key not in cache and key not in pending:
                pending[key] = executor.submit(evaluate_design, candidate)
            else:
                # For repeated proposals, count a cache hit instead of spending another construction evaluation.
                statistics["cache_hits"] += 1
        # Pump GUI events while workers run; preview completed designs without changing coupling seeds.
        future_keys = {future: key for key, future in pending.items()}
        remaining, completed_results = set(future_keys), {}
        while remaining:
            # Poll futures on the calling thread, allowing the GUI to process events while model threads are busy.
            finished, remaining = wait(remaining, timeout=.1, return_when=FIRST_COMPLETED)
            for future in finished:
                completed_results[future_keys[future]] = future.result()
            # Preview completed records immediately, but keep them out of the coupling archive until the batch ends.
            update_plot([result[0] for result in completed_results.values()], running=len(remaining))
        # Commit in submission order only after the whole batch, preserving nearest-root choices.
        for key in pending:
            record, turbine = completed_results[key]
            # Commit the completed record and update unique-design counts, including all initialization attempts.
            cache[key] = record
            evaluations.append(record)
            statistics["evaluations"] += 1
            statistics["construction_attempts"] += len(record["attempts"])
            statistics["successful_constructions"] += record["constructed"]
            statistics["failed_constructions"] += sum(error is not None for error in record["attempts"])
            statistics["failure_reasons"].update(error for error in record["attempts"] if error is not None)
            # If construction succeeded but reading or checking outputs failed, count that failure separately.
            if "evaluation_error" in record:
                statistics["evaluation_errors"][record["evaluation_error"]] += 1
            statistics["feasible_designs"] += record["feasible"]

            # Keep only the best feasible full object; the remaining records store scalar diagnostics.
            # If the design is feasible and improves the incumbent, retain its full turbine object as the new winner.
            if record["feasible"] and (best_record is None or record["efficiency"] > best_record["efficiency"]):
                best_record, best_turbine = record, turbine
            # If the record contains a successful inner solution, make it available to subsequent evaluation batches.
            if "coupling_solution" in record:
                coupling_archive.append(dict(design=record["design"], solution=record["coupling_solution"]))
        # Refresh committed totals after the batch, then return records in the same order as the input candidates.
        update_plot()
        return [cache[key] for key in keys]

    # --- Scouting proposals shared with GA initialization ---

    # Bind this run's bounds, model settings and generator once, retaining the original proposal-call interface.
    sample_designs = partial(
        scouting_functions.sample_designs, variable_bounds=variable_bounds, integer_names=integer_names,
        constraint_ranges=constraint_ranges, mode=mode, random_generator=random_generator,
        complete_design=complete_design, euler_diameter=euler_diameter)
    local_designs = partial(
        scouting_functions.local_designs, variable_bounds=variable_bounds, integer_names=integer_names,
        random_generator=random_generator, relative_scales=tuple(perturbation_scales),
        standard_deviations=dict(perturbation_standard_deviations or {}))

    # --- Plot and terminal progress; both run on the calling thread ---

    # Refresh the progress text and both design classes while model evaluations are running.
    # Completed worker previews can appear before batch commit, without changing the optimization archive.
    def update_plot(preview_records=(), running=0, *, progress=None, force=False):
        nonlocal last_plot_state, plot_progress
        # If plotting is disabled or the user closed the window, return without accessing its artists.
        if figure is None or not plt.fignum_exists(figure.number):
            return
        # If the caller supplied a new phase or generation label, retain it for subsequent within-batch updates.
        if progress is not None:
            plot_progress = progress
        # Combine committed evaluations with finished worker previews to show all evaluated locations immediately.
        display_records = evaluations + list(preview_records)
        # Remember counts and phase so unchanged polling updates need only process GUI events.
        plot_state = (len(display_records), running, plot_progress)
        # If a generation requests a redraw or the displayed state changed, rebuild the plot contents and status.
        if force or plot_state != last_plot_state:
            # Plot only completed constructions with finite efficiency; retain all records for status and statistics.
            plotted_records = [record for record in display_records if record["constructed"]
                               and record["efficiency"] is not None and np.isfinite(record["efficiency"])]
            # Separate the plotted designs by feasibility, using the same efficiency filter for both marker shapes.
            feasible = [record for record in plotted_records if record["feasible"]]
            infeasible = [record for record in plotted_records if not record["feasible"]]
            # Draw the infeasible designs first so feasible circles remain visible at overlapping locations.
            infeasible_coordinates = np.array(
                [[record["design"][name] for name in plot_axes] for record in infeasible]).reshape(-1, 2)
            infeasible_points.set_offsets(infeasible_coordinates)
            # Give each plotted square its finite efficiency on the shared color scale.
            infeasible_efficiencies = np.array([record["efficiency"] for record in infeasible])
            infeasible_points.set_array(infeasible_efficiencies)
            # Include both classes when scaling the colorbar, even before any feasible design has been found.
            plotted_efficiencies = [record["efficiency"] for record in plotted_records]
            # If finite efficiencies exist, apply their full range to the shared normalization.
            if plotted_efficiencies:
                minimum_efficiency, maximum_efficiency = min(plotted_efficiencies), max(plotted_efficiencies)
                # If all values match, extend the range down by one percentage point to avoid a zero-width scale.
                if minimum_efficiency == maximum_efficiency:
                    minimum_efficiency -= .01
                points.set_clim(minimum_efficiency, maximum_efficiency)
            constructed_count = sum(record["constructed"] for record in display_records)
            # Build the status from completed designs, showing construction success separately from full feasibility.
            status = f"{len(display_records)} evaluated | {constructed_count} constructed | {len(feasible)} feasible"
            # If evaluations are still running, include their count so a slow batch does not look stalled.
            if running:
                status += f" | {running} running"
            # If no design is feasible yet, explain why the evaluated locations appear only as squares.
            if not feasible:
                status = "No feasible designs yet\n" + status
                # If any evaluations have finished, also show the smallest violation as an indication of feasibility
                # progress.
                if display_records:
                    smallest_violation = min(record["violation"] for record in display_records)
                    status += f"\nSmallest constraint violation: {smallest_violation:.4g}"
            # Apply the current scout/generation label and evaluation status to the existing text artist.
            plot_status.set_text(f"{plot_progress}\n{status}" if plot_progress else status)
            # If feasible designs exist, update their positions, efficiency colors and the maximum-efficiency marker.
            if feasible:
                # Collect x/y coordinates and efficiencies in the same order so each point receives the correct color.
                coordinates = np.array([[record["design"][name] for name in plot_axes] for record in feasible])
                efficiencies = np.array([record["efficiency"] for record in feasible])
                # Replace the scatter coordinates and color values with all feasible designs found so far.
                points.set_offsets(coordinates)
                points.set_array(efficiencies)
                # Highlight the largest feasible efficiency using the shared colors, regardless of infeasible values.
                maximum_feasible_efficiency = float(efficiencies.max())
                best_coordinates = coordinates[int(np.argmax(efficiencies))]
                maximum_point.set_offsets(best_coordinates.reshape(1, 2))
                maximum_point.set_facecolors(points.cmap(points.norm(maximum_feasible_efficiency)))
                # Move the max annotation with the winner and place it away from the nearest horizontal plot edge.
                maximum_label.xy = best_coordinates
                label_on_left = best_coordinates[0] > sum(axes.get_xlim()) / 2
                maximum_label.set_position((-6 if label_on_left else 6, 6))
                maximum_label.set_ha("right" if label_on_left else "left")
                # Make the maximum annotation visible now that a feasible winner supplies its position.
                maximum_label.set_visible(True)
            # Save this displayed state and request a GUI redraw; a generation boundary can force the next redraw.
            last_plot_state = plot_state
            figure.canvas.draw_idle()
        # GUI event processing belongs on the calling thread, never in model workers.
        figure.canvas.flush_events()

    # Record one completed scout iteration or GA generation, and optionally print the same data to the terminal.
    def report_progress(stage, generation):
        # Choose the least-violating evaluation; when feasible records exist, its violation is zero.
        closest = min(evaluations, key=rank)
        # Store cumulative counts and the best feasible efficiency, together with elapsed time for this progress row.
        row = dict(stage=stage, generation=generation, seconds=time.perf_counter() - started,
                   evaluations=statistics["evaluations"], constructions=statistics["successful_constructions"],
                   failures=statistics["failed_constructions"], feasible=statistics["feasible_designs"],
                   best_efficiency=best_record["efficiency"] if best_record else None,
                   minimum_violation=closest["violation"])
        # Append the completed progress row to the history returned to the caller.
        history.append(row)
        # Use explicit phase names so scout iterations are not mistaken for GA generations.
        progress_label = f"GA generation {generation}" if stage == "GA" else f"Prescout iteration {generation}"
        # If terminal progress is enabled, print this completed batch immediately with its cumulative statistics.
        if verbose:
            best_text = f"{100 * row['best_efficiency']:.4f}%" if best_record else "--"
            print(f"{progress_label} completed | {row['seconds']:7.1f}s | eval {row['evaluations']:5d} | "
                  f"built {row['constructions']:5d} | failed {row['failures']:5d} | feasible {row['feasible']:5d} | "
                  f"best {best_text:>10s} | CV {row['minimum_violation']:.5g}", flush=True)
        # Redraw even a cache-only generation, so the displayed generation always advances.
        update_plot(progress=progress_label, force=True)
        return closest

    # --- Pymoo adapter: maximize efficiency by minimizing its negative ---

    # Expose the turbine evaluator through pymoo: one scalar objective and a fixed-length inequality vector.
    class TurbineProblem(Problem):
        def __init__(self, executor):
            # Keep the executor available to pymoo batches and declare which design variables are real or integer.
            self.executor = executor
            super().__init__(vars={name: (Integer if name in integer_names else Real)(bounds=bounds)
                                   for name, bounds in variable_bounds.items()}, n_obj=1, n_ieq_constr=constraint_count)

        def _evaluate(self, candidates, out, *args, **kwargs):
            # Evaluate the candidate batch through the same cache, failure handling and nearest-coupling archive as
            # scouting.
            records = evaluate_batch(candidates, self.executor)
            # Failed constructions are rejected through G; every objective value stays finite.
            out["F"] = np.array([-record["efficiency"] if record["efficiency"] is not None
                                 and np.isfinite(record["efficiency"]) else 0. for record in records]).reshape(-1, 1)
            # Return the constraint residual vector separately so pymoo can prioritize feasibility.
            out["G"] = np.array([record["residuals"] for record in records])

    # --- Run prescouting and GA with the same bounds, constraints and coupling archive ---

    # If terminal progress is enabled, print the chosen model, objective and worker count before scouting.
    if verbose:
        print(f"\n{mode}: maximize {efficiency_name}; {worker_count} threads; "
              f"{len(variable_names)} search variables. CV=0 means feasible.", flush=True)
    # Use one thread pool for scouting and GA so worker-local gas and loss-model copies can be reused.
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        # Share evaluation state with scouting and use its returned leader to initialize GA.
        anchor = scouting_functions.prescout_design_domain(
            evaluate_batch=partial(evaluate_batch, executor=executor), sample_designs=sample_designs,
            local_designs=local_designs, report_progress=report_progress, update_plot=update_plot,
            evaluations=evaluations, rank=rank, scout_size=scout_size,
            worker_count=worker_count, stagnation_generations=stagnation_generations, verbose=verbose)
        # Default to the exhausted-offspring reason if pymoo has no further generations to advance.
        termination = "GA produced no further unique offspring"
        # Seed half the GA locally and half globally; reuse the scout leader without reevaluation.
        local_count = population_size // 2
        initial = [{name: anchor["design"][name] for name in variable_names}]
        initial += local_designs(anchor, max(0, local_count - 1))
        initial += sample_designs(population_size - len(initial))

        # Mixed operators round blade counts after crossover and mutation.
        duplicate_elimination = ga_arguments.pop("eliminate_duplicates", MixedVariableDuplicateElimination())
        # If duplicate elimination was supplied as a boolean, convert it to the matching pymoo strategy object.
        if isinstance(duplicate_elimination, bool):
            duplicate_elimination = (MixedVariableDuplicateElimination() if duplicate_elimination
                                     else NoDuplicateElimination())
        mating_arguments = dict(selection=ga_arguments.pop("selection", RandomSelection()))
        # Build real and integer crossover/mutation operators from the chosen probabilities and distribution
        # indices.
        for operator_name, operator_class in (("crossover", SBX), ("mutation", PM)):
            settings = dict(prob=operator_settings[operator_name + "_probability"],
                            eta=operator_settings[operator_name + "_eta"])
            mating_arguments[operator_name] = ga_arguments.pop(operator_name, {
                Real: operator_class(**settings),
                Integer: operator_class(**settings, vtype=float, repair=RoundingRepair())})
        # Use custom mating unchanged when supplied; otherwise assemble the tuned mixed-variable mating operator.
        mating = ga_arguments.pop("mating", None)
        # If the user did not supply custom mating, build it from the selected real/integer operators.
        if mating is None:
            mating = MixedVariableMating(**mating_arguments, eliminate_duplicates=duplicate_elimination)
        # Initialize pymoo with the scout-derived population and keep stopping decisions in the loop below.
        algorithm = MixedVariableGA(pop_size=population_size, mating=mating, sampling=Population.new(X=initial),
                                    eliminate_duplicates=duplicate_elimination, **ga_arguments)
        # Disable the native display because report_progress supplies the turbine-specific generation summaries.
        algorithm.setup(TurbineProblem(executor), seed=seed, termination=NoTermination(), verbose=False)

        # Stop on the requested generation cap, exhausted offspring, or incumbent stagnation.
        improvement_history = []
        generation = 0
        while algorithm.has_next():
            # If terminal progress is enabled, announce the generation before its potentially expensive evaluations.
            if verbose:
                print(f"GA generation {generation + 1} started.", flush=True)
            update_plot(progress=f"GA generation {generation + 1}")
            # Advance exactly one GA generation, then publish its completed statistics and redraw the plot.
            algorithm.next()
            generation += 1
            closest = report_progress("GA", generation)
            # Track feasible efficiency once available; before feasibility, track decreasing total constraint
            # violation.
            metric = (True, best_record["efficiency"]) if best_record else (False, -closest["violation"])
            improvement_history.append(metric)
            # If the requested generation limit has been reached, return the best feasible incumbent found so far.
            if max_generations is not None and generation >= max_generations:
                termination = "generation limit"
                break
            # If pymoo cannot continue generating unique offspring, finish rather than repeating cached designs.
            if not algorithm.has_next():
                break
            # If enough generations have completed, test improvement over the stagnation window.
            if len(improvement_history) > stagnation_generations:
                # Compare metrics only across the same feasibility state, using the appropriate improvement
                # tolerance.
                earlier = improvement_history[-stagnation_generations - 1]
                tolerance = efficiency_tolerance if metric[0] else 1e-5
                # If the feasibility state stayed the same and improvement is too small, stop for incumbent
                # stagnation.
                # A newly feasible design starts a new comparison period instead of being compared with an
                # infeasible score.
                if earlier[0] == metric[0] and metric[1] - earlier[1] <= tolerance:
                    termination = "incumbent stagnation"
                    break

    # --- Return the feasible winner separately from construction and search diagnostics ---

    # Convert failure counters to ordinary dictionaries for the result object.
    statistics["failure_reasons"] = dict(statistics["failure_reasons"])
    statistics["evaluation_errors"] = dict(statistics["evaluation_errors"])
    # Keep the least-violating record available when the search returns no feasible turbine.
    closest = min(evaluations, key=rank)
    # If terminal progress is enabled, report the stopping reason and number of completed GA generations.
    if verbose:
        completed_generations = sum(row["stage"] == "GA" for row in history)
        print(f"Optimization finished: {termination}; {completed_generations} GA generations completed.", flush=True)
    # Capture results and elapsed optimization time before any wait for the user to close the final plot.
    optimization_result = dict(
        success=best_record is not None, best_turbine=best_turbine,
        best_design=deepcopy(best_record["design"]) if best_record else None,
        best_results=deepcopy(best_record["values"]) if best_record else None,
        best_efficiency=best_record["efficiency"] if best_record else None,
        best_infeasible=None if best_record else closest, termination=termination,
        elapsed_seconds=time.perf_counter() - started, statistics=statistics,
        evaluations=evaluations, history=history, figure=figure)
    # If the caller wants to inspect an open interactive plot, keep it open after the calculations finish.
    if figure is not None and interactive_plot and block_plot and plt.fignum_exists(figure.number):
        plt.show(block=True)
    return optimization_result
