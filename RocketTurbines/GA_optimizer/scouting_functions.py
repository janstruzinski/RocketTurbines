import numpy as np
from scipy.stats import qmc


# --- Global proposals inside the original search bounds ---

def sample_designs(count, *, variable_bounds, integer_names, constraint_ranges, mode, random_generator,
                   complete_design, euler_diameter):
    """Propose geometry-informed Sobol designs without evaluating either turbine model.

    Use the original variable bounds throughout. Analytical diameter, velocity-triangle and metal-thickness
    estimates bias the initial proposals toward useful geometries; they do not replace the actual constraints.
    The same proposal function also supplies the global part of GA's initial population.

    :param integer count: Number of proposals. A power-of-two Sobol sequence is generated and then truncated.
    :param dict variable_bounds: At least one search-variable name mapped to (lower, upper) bounds, in optimizer order.
    :param set integer_names: Names of integer-valued inputs, including stator and rotor blade counts.
    :param dict constraint_ranges: Output constraints used for inexpensive tip-diameter and blade-height estimates.
    :param string mode: "Turbine1D" or "SupersonicTurbine"; selects the rotor thickness estimate.
    :param numpy.random.Generator random_generator: This run's shared generator, advanced for each Sobol seed.
    :param callable complete_design: Combine a candidate dictionary with fixed inputs and convert count types.
    :param callable euler_diameter: Calculate analytical Euler diameter (m) from a complete input dictionary.
    :return: Candidate dictionaries containing only the search variables, clipped to their original bounds.
    :rtype: list of dict
    """
    # Keep the bound-dictionary order used by the optimizer when assigning Sobol coordinates to variables.
    variable_names = list(variable_bounds)
    # Generate enough Sobol points for a power-of-two sequence, then keep the requested number of proposals.
    sobol_seed = int(random_generator.integers(2**31))
    sample_power = int(np.ceil(np.log2(count)))
    sampler = qmc.Sobol(len(variable_names), scramble=True, seed=sobol_seed)
    samples = sampler.random_base2(sample_power)[:count]
    candidates = []

    # Read the fixed annulus constraints once for this batch of proposals.
    tip_interval = constraint_ranges.get("D_tip", (None, None))
    tip_limit = None if isinstance(tip_interval, bool) else tip_interval[1]
    height_intervals = [constraint_ranges.get(name, (None, None)) for name in ("l_rotor", "l_stator")]
    blade_height = max(0. if isinstance(interval, bool) else interval[0] or 0. for interval in height_intervals)
    diameter_limit = None
    # If tip diameter and minimum blade height define a usable annulus, estimate its Euler-diameter limit.
    # This estimate biases proposals only; actual constructed geometry is still checked by the constraints.
    if tip_limit is not None and tip_limit > blade_height > 0:
        diameter_limit = np.hypot(tip_limit - blade_height, blade_height)

    for sample in samples:
        # Start with uniform real variables and uniformly sampled inclusive integer bounds.
        fractions = dict(zip(variable_names, sample))
        candidate = {}
        for name, (lower, upper) in variable_bounds.items():
            # If this variable is an integer count, sample all allowed integers with equal initial probability.
            if name in integer_names:
                offset = min(int(fractions[name] * (upper - lower + 1)), upper - lower)
                candidate[name] = int(lower + offset)
            # If the variable is continuous, map the Sobol fraction directly into the supplied numeric interval.
            else:
                candidate[name] = lower + fractions[name] * (upper - lower)

        # Estimate the largest Euler diameter compatible with the requested tip and blade height.
        design = complete_design(candidate)
        # If an annulus limit is available, bias RPM and loading toward compatible Euler diameters.
        if diameter_limit is not None:
            specific_work = design["shaft_power"] / design["mdot"]
            work_lower, work_upper = variable_bounds.get(
                "work_coefficient_tt", (design["work_coefficient_tt"],) * 2)

            # Sample RPM and loading together through diameter, without changing their bounds.
            # If RPM is a search variable, favor speeds that can satisfy the diameter limit at the largest loading.
            if "RPM" in candidate:
                lower, upper = variable_bounds["RPM"]
                rpm_for_diameter = 60 * np.sqrt(specific_work / work_upper) / (np.pi * diameter_limit)
                preferred_lower = max(lower, rpm_for_diameter)
                # If the estimated speed range is nonempty, map the Sobol fraction into that range.
                if preferred_lower < upper:
                    candidate["RPM"] = preferred_lower + fractions["RPM"] * (upper - preferred_lower)
            # Translate the loading bounds at this RPM into the corresponding allowable Euler-diameter interval.
            rpm = candidate.get("RPM", design["RPM"])
            diameter_lower = max(.60 * diameter_limit, 60 * np.sqrt(specific_work / work_upper) / (np.pi * rpm))
            diameter_upper = min(diameter_limit, 60 * np.sqrt(specific_work / work_lower) / (np.pi * rpm))
            # If loading is variable and a diameter interval is available, propose loading through that diameter.
            if "work_coefficient_tt" in candidate and diameter_lower < diameter_upper:
                diameter_span = diameter_upper - diameter_lower
                diameter = diameter_lower + fractions["work_coefficient_tt"] * diameter_span
                candidate["work_coefficient_tt"] = specific_work * (60 / (np.pi * rpm * diameter))**2

        # Cover the single-nozzle branch as well as modest and larger nozzle counts.
        design = complete_design(candidate)
        # If stator nozzle count is variable, distribute proposals among single, modest and larger nozzle counts.
        if "no_blades_stator" in candidate:
            fraction = fractions["no_blades_stator"]
            lower, upper = variable_bounds["no_blades_stator"]
            # If one nozzle is allowed and the sample lies in the single-nozzle stratum, propose one nozzle.
            if lower == 1 and fraction < .35:
                candidate["no_blades_stator"] = 1
            # If the sample lies in the modest-count stratum, sample its available nozzle counts.
            elif fraction < .85 and max(2, lower) <= min(30, upper):
                local_fraction = (fraction - .35) / .5 if lower == 1 else fraction / .85
                # Map this stratum into the available modest-count interval without changing the declared count
                # bounds.
                nozzle_span = min(30, upper) - max(2, lower) + 1
                candidate["no_blades_stator"] = int(max(2, lower) + local_fraction * nozzle_span)

        # Relate flow to loading to favor plausible velocity triangles in the initial proposals.
        # If flow coefficient is variable, scale its proposal with the square root of loading and clip to its
        # bounds.
        if "flow_coefficient" in candidate:
            proposed_flow = (.5 + .9 * fractions["flow_coefficient"]) * np.sqrt(design["work_coefficient_tt"])
            candidate["flow_coefficient"] = np.clip(proposed_flow, *variable_bounds["flow_coefficient"])

        # Favor rotor counts that leave room for the circumferential metal thickness.
        design = complete_design(candidate)
        # If rotor blade count is variable, favor counts that leave room for the projected minimum metal thickness.
        if "no_blades_rotor" in candidate:
            lower, upper = variable_bounds["no_blades_rotor"]
            diameter = euler_diameter(design)
            thickness = design["t_TE_rotor"]
            # If blading will be constructed, project its minimum metal thickness using the estimated inlet angle.
            if mode == "SupersonicTurbine":
                cosine_beta = design["flow_coefficient"] / np.hypot(
                    design["flow_coefficient"], design["work_coefficient_tt"] / 2)
                thickness = design.get("t_min_rotor", 0.) / max(cosine_beta, 1e-6)
            # Reserve a thickness/pitch ratio of 0.18 in these proposals, leaving room below the enforced 0.2 limit.
            count_for_thickness = int(.18 * np.pi * diameter / max(thickness, 1e-12))
            preferred_upper = max(lower, min(upper, count_for_thickness))
            candidate["no_blades_rotor"] = int(
                lower + fractions["no_blades_rotor"] * (preferred_upper - lower + 1))

        # Single-nozzle admission is only an initial estimate for the inner coupling.
        # If admission is variable, propose a branch-dependent estimate and keep it within the supplied bounds.
        if "admission_fraction" in candidate:
            proposed_admission = (.08 if design["no_blades_stator"] == 1 else
                                  (.06 + .16 * fractions["admission_fraction"])
                                  * np.sqrt(design["no_blades_stator"]))
            candidate["admission_fraction"] = np.clip(proposed_admission, *variable_bounds["admission_fraction"])
        # Retain each completed proposal; its efficiency and full geometry will be evaluated later by the workers.
        candidates.append(candidate)
    return candidates


# --- Local proposals around completed evaluations ---

def local_designs(anchor, count, *, variable_bounds, integer_names, random_generator,
                  relative_scales=(.035, .012), standard_deviations=None):
    """Generate bounded local proposals around a feasible or least-violating evaluation.

    By default, alternate continuous perturbations of 3.5% and 1.2% of each bound span. Stator count uses a
    standard deviation of one nozzle; other integer counts use three. Absolute per-variable overrides take
    precedence over these settings. Integer proposals are rounded after clipping. These proposals are used
    during both prescouting and GA initialization.

    :param dict anchor: Evaluation record whose "design" dictionary supplies the center of the perturbations.
    :param integer count: Number of local proposals; zero returns an empty list.
    :param dict variable_bounds: Search-variable names mapped to their unchanged (lower, upper) bounds.
    :param set integer_names: Names of variables that must be rounded to integers.
    :param numpy.random.Generator random_generator: The optimizer's shared random generator.
    :param tuple or list relative_scales: Nonempty sequence of nonnegative continuous standard deviations as
     fractions of each variable's bound span. Default (0.035, 0.012). Cycle through the sequence within this call;
      a call requesting one proposal uses only the first scale.
    :param dict or None standard_deviations: Absolute nonnegative Gaussian standard deviations by active variable
     name, in that input's units, overriding both continuous and integer defaults. Zero retains the anchor value.
      None uses defaults. This dictionary is read without modification; fixed inputs are never included in proposals.
    :return: Candidate dictionaries containing only the search variables.
    :rtype: list of dict
    """
    candidates = []
    # Read optional per-variable widths without changing the caller's settings.
    standard_deviations = standard_deviations or {}
    # Cycle through the requested relative widths when generating a group of nearby seeds.
    for index in range(count):
        radius = relative_scales[index % len(relative_scales)]
        candidate = {}
        for name, (lower, upper) in variable_bounds.items():
            # If an absolute width was supplied for this variable, use it instead of the default perturbation.
            if name in standard_deviations:
                deviation = standard_deviations[name]
            # If this is an integer count without an override, retain the original blade-count widths.
            elif name in integer_names:
                deviation = 1. if name == "no_blades_stator" else 3.
            else:
                # Otherwise, express the continuous width as a fraction of this variable's original bound span.
                deviation = radius * (upper - lower)
            # Draw a Gaussian perturbation, clip it to the original bounds and round counts back to integers.
            value = np.clip(anchor["design"][name] + random_generator.normal(0., deviation), lower, upper)
            candidate[name] = int(round(value)) if name in integer_names else float(value)
        candidates.append(candidate)
    return candidates


# --- Feasibility discovery before efficiency optimization ---

def prescout_design_domain(evaluate_batch, sample_designs, local_designs, report_progress, update_plot,
                          evaluations, rank, *, scout_size=64, worker_count=1,
                          stagnation_generations=8, verbose=True):
    """Search for a feasible design before starting GA, using the optimizer's existing evaluation state.

    Evaluate global Sobol proposals first. If no feasible design exists, perturb the eight least-violating
    records and add two global proposals per iteration. Stop at feasibility or violation stagnation. The shared
    evaluator retains the cache, construction diagnostics and successful coupling archive.

    :param callable evaluate_batch: Evaluate candidate dictionaries using the optimizer's existing thread pool;
     append unique records to evaluations and return their records. The executor is already bound by the caller.
    :param callable sample_designs: Generate the requested number of geometry-informed global proposals.
    :param callable local_designs: Generate local proposals from an evaluation record and a requested count.
    :param callable report_progress: Record and print a completed (stage, iteration); return the closest record.
    :param callable update_plot: Refresh the existing plot, accepting a progress label as a keyword argument.
    :param list evaluations: Shared list of completed evaluation records, populated by evaluate_batch.
    :param callable rank: Rank records by constraint violation, then by descending efficiency.
    :param integer scout_size: Number of initial global proposals. Default 64.
    :param integer worker_count: Available evaluation threads; local batches contain max(6, worker_count - 2) designs.
    :param integer stagnation_generations: Violation-improvement window in completed scout iterations. Default 8.
     Stop when improvement over that window is at most 1e-5.
    :param bool verbose: Print flushed scout-iteration start messages. Default True. Completion messages are
     controlled by the supplied report_progress callback.
    :return: Best-ranked evaluation when scouting ends, whether feasible or still constraint-violating.
    :rtype: dict
    """
    # Scouting seeks feasibility first, retaining global proposals on every iteration.
    # If terminal progress is enabled, announce the initial scout batch before waiting for its evaluations.
    if verbose:
        print(f"Prescout iteration 0 started ({scout_size} proposals); GA begins after scouting.", flush=True)
    update_plot(progress="Prescout iteration 0")
    # Evaluate the initial Sobol proposals first, then keep scouting only if feasibility has not been reached.
    evaluate_batch(sample_designs(scout_size))
    scout_scores = []
    iteration = 0
    while True:
        # After each completed batch, store the smallest violation for the scouting stagnation comparison.
        closest = report_progress("scout", iteration)
        scout_scores.append(closest["violation"])
        # If a feasible seed has been found, hand the scout results to GA.
        if closest["feasible"]:
            return closest
        # If enough scout iterations exist, compare violation improvement across the requested stagnation window.
        if len(scout_scores) > stagnation_generations:
            violation_improvement = scout_scores[-stagnation_generations - 1] - scout_scores[-1]
            # If violation has stopped improving sufficiently, hand the least-violating design to GA.
            if violation_improvement <= 1e-5:
                return closest

        # Keep exploring globally while perturbing the least-violating scout leaders.
        # Keep the best eight scout records as leaders for local perturbations.
        leaders = sorted(evaluations, key=rank)[:8]
        local_count = max(6, worker_count - 2)
        proposals = [local_designs(leaders[index % len(leaders)], 1)[0] for index in range(local_count)]
        # If terminal progress is enabled, announce the next local/global scout batch before evaluating it.
        if verbose:
            print(f"Prescout iteration {iteration + 1} started ({len(proposals) + 2} proposals).", flush=True)
        update_plot(progress=f"Prescout iteration {iteration + 1}")
        # Evaluate the local proposals together with two new global proposals, keeping the same domain bounds.
        evaluate_batch(proposals + sample_designs(2))
        iteration += 1
