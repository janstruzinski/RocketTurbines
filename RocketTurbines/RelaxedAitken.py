import numpy as np

def solve_with_relaxed_aitken(func, x0, initial_relaxation=1., min_relaxation=0.05, max_relaxation=1.5,
                             xtol=1e-6, rtol=0., maxiter=300, bounds=(-np.inf, np.inf), x_scale=1.):
    """Find a scalar or vector root of a function with bounded, adaptive Aitken relaxation.

    Relaxation factor is calculated from successive residual vectors and applied to all components.
    Convergence requires the residual, func(x) - x, to satisfy the supplied tolerances.

    :param callable func: Function to solve, returning the same shape as its input.
    :param float or array_like x0: Initial solution estimate.
    :param float initial_relaxation: Relaxation factor for the first update, within the supplied caps.
    :param float min_relaxation: Lower cap on the relaxation factor.
    :param float max_relaxation: Upper cap on the relaxation factor.
    :param float xtol: Absolute tolerance on each unscaled residual component.
    :param float rtol: Relative tolerance, added as rtol * abs(x) to xtol.
    :param int maxiter: Maximum number of relaxed updates.
    :param tuple bounds: Lower and upper solution bounds, broadcastable to x0. Updates are clipped to these bounds.
    :param float or array_like x_scale: Positive normalization scales used to calculate the relaxation factor.
    :return: Converged root, as a float for scalar x0 or a NumPy array otherwise.
    :rtype: float or numpy.ndarray
    :raises ValueError: If the settings, initial estimate or map output shape are invalid.
    :raises RuntimeError: If the iteration becomes nonfinite, stagnates or exceeds maxiter.
    """

    # Convert the initial estimate to an array and expand bounds and normalization scales to match its shape.
    x = np.array(x0, dtype=float, copy=True)
    lower, upper, scale = [np.broadcast_to(value, x.shape) for value in (*bounds, x_scale)]
    # Check relaxation caps, convergence tolerances and the allowed number of updates.
    if (not np.all(np.isfinite([initial_relaxation, min_relaxation, max_relaxation, xtol, rtol]))
            or not 0 < min_relaxation <= initial_relaxation <= max_relaxation or min(xtol, rtol) < 0
            or xtol + rtol == 0 or not isinstance(maxiter, (int, np.integer)) or maxiter < 1):
        raise ValueError("Aitken requires positive, ordered relaxation factors, valid tolerances and maxiter >= 1.")
    # Ensure the initial estimate is within bounds and the component scales are finite and positive.
    if (not np.all(np.isfinite(x)) or np.any(np.isnan(lower)) or np.any(np.isnan(upper))
            or np.any(lower > upper) or np.any(x < lower) or np.any(x > upper)
            or not np.all(np.isfinite(scale)) or np.any(scale <= 0)):
        raise ValueError("Aitken requires a finite initial estimate within bounds and finite, positive scales.")

    # Store the normalized residual and relaxation factor for the Aitken update.
    previous_residual = None
    previous_relaxation = initial_relaxation
    # Start iterations
    for iteration in range(maxiter + 1):
        # Evaluate the solution, passing a float for scalar problems and an array otherwise.
        mapped = np.asarray(func(float(x) if x.ndim == 0 else x), dtype=float)
        if mapped.shape != x.shape:
            raise ValueError("The Aitken fixed-point map must preserve the input shape.")
        # Check the actual residual
        residual = mapped - x
        # Raise an error if residual is infinite
        if not np.all(np.isfinite(residual)):
            raise RuntimeError("Aitken encountered a nonfinite fixed-point residual.")
        # If residual is below the acceptable error, return the solution
        if np.all(np.abs(residual) < xtol + rtol * np.abs(x)):
            return float(x) if x.ndim == 0 else x
        # If maximum number of iterations reached, break the loop.
        if iteration == maxiter:
            break

        # Normalizing the residual affects the shared relaxation factor.
        normalized_residual = (residual / scale).ravel()
        # Use the initial relaxation factor when there is no usable estimate from previous residuals.
        relaxation = initial_relaxation
        # Otherwise calculate the relaxation factor
        if previous_residual is not None:
            # Estimate one relaxation factor shared by all components from the residual change.
            difference = normalized_residual - previous_residual
            denominator = np.dot(difference, difference)
            # Avoid dividing by a nearly zero residual difference.
            if denominator > 1e-28:
                candidate = -previous_relaxation * np.dot(previous_residual, difference) / denominator
                # If candidate relaxation factor is finite, then accept it
                if np.isfinite(candidate):
                    # Apply the supplied relaxation caps.
                    relaxation = np.clip(candidate, min_relaxation, max_relaxation)
        # Retain this normalized residual and relaxation factor for the next iteration.
        previous_residual = normalized_residual
        previous_relaxation = relaxation

        # Calculate next trail solution. Clip it to be in the inside the supplied domain.
        next_x = np.clip(x + relaxation * residual, lower, upper)
        # A step clipped or rounded to zero is a failure if the residual has not converged.
        if not np.all(np.isfinite(next_x)) or np.array_equal(next_x, x):
            raise RuntimeError("Aitken stagnated or produced a nonfinite trial state.")
        x = next_x

    # Report nonconvergence when all allowed updates have been used.
    raise RuntimeError("Aitken fixed-point iteration did not converge within maxiter.")
