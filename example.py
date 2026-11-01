import numpy as np
from RocketTurbines.Turbine1D import Turbine1D
from RocketTurbines.IdealGas import IdealGas
from RocketTurbines.TraupelLossModel import TraupelLossModel
from RocketTurbines.SupersonicTurbine import SupersonicTurbine
from RocketTurbines.GA_optimizer.optimization_functions import optimize_turbine
from RocketTurbines.GA_optimizer.Smith_chart import plot_Smith_chart

# Create a gas object
gas = IdealGas(T_max=900, T_min=450, species=["H2O","O2"], mass_fractions=[0.54, 0.46])
# Create a loss model object
loss_model = TraupelLossModel(extrapolation_method="closest", rotor="impulse_high_M")
# Create meanline analysis object
turbine = Turbine1D()
turbine.size_turbine(gas=gas, work_coefficient_tt=4.321, flow_coefficient=1.570, reaction_isentropic_tt=0, RPM=30e3,
                    shaft_power=30e3, mdot=0.15, T_t0=900, p_t0=50e5, radial_clearance=0.1e-3, no_blades_stator=1,
                    no_blades_rotor=47, chord_over_pitch_stator=3.300, chord_over_pitch_rotor=2.518,
                    t_TE_stator=0.5e-3, t_TE_rotor=0.5e-3, Ra_roughness=3e-6, loss_model=loss_model,
                    admission_fraction=0.05440, delta_s_estimate=[350, 130, 200])
# Print its results
turbine.print_analysis_results()
turbine.plot_velocity_triangles()

# It is also possible to create SupersonicTurbine class, which will couple meanline analysis in Turbine1D and blading
# design from SupersonicTurbineBlading library
# First create arguments for Turbine1D
Turbine1D_arguments = dict(
    work_coefficient_tt=4.321, flow_coefficient=1.570, reaction_isentropic_tt=0, RPM=30e3,
    shaft_power=30e3, mdot=0.15, T_t0=900, p_t0=50e5, radial_clearance=0.1e-3,
    no_blades_stator=1, no_blades_rotor=47, chord_over_pitch_stator=3.300,
    chord_over_pitch_rotor=2.518, t_TE_stator=0.5e-3, t_TE_rotor=0.5e-3,
    Ra_roughness=3e-6, admission_fraction=0.05440, s_ax_over_pitch_rotor=0.35,
    delta_s_estimate=[350, 130, 200])

# Then create arguments for SupersonicStatorNozzle
SupersonicStatorNozzle_arguments = dict(
    contour_method="conical", number_of_nodes=51, half_cone_metal_angle=7.0,
    calculate_boundary_layer=False)

# And for SupersonicStatorBlade
SupersonicRotorBlades_arguments = dict(
    number_of_nodes=51, calculate_starting=False, calculate_boundary_layer=True,
    use_leading_edge_entry_correction=False)

# Create SupersonicTurbine object
supersonic_turbine = SupersonicTurbine(
    gas=gas, loss_model=loss_model, Turbine1D_arguments=Turbine1D_arguments,
    SupersonicStatorNozzle_arguments=SupersonicStatorNozzle_arguments,
    SupersonicRotorBlades_arguments=SupersonicRotorBlades_arguments,
    separation_margin=0.175, t_min_stator=0.5e-3, t_min_rotor=0.5e-3,
    stator_nozzle_chord_scale=1.0, print_user_warnings=False)

# Call automatic coupling
supersonic_turbine.couple_meanline_and_blading()
# Print all margins and optimality coefficients that describe nozzle packing
supersonic_turbine.print_design_coupling_results()
# Print geometry of converged & coupled turbine
supersonic_turbine.turbine_meanline.print_geometry()
# Plot 3D geometry
supersonic_turbine.plot_3D_geometry()
# Plot 2D geometry of the blades
supersonic_turbine.plot_2D_geometry(number_of_rotor_blades=10)

# The code also has functionality to generate Smith diagrams by exploring efficiency over flow and work coefficients
# while keeping the other meanline inputs fixed.
Smith_chart_arguments = dict(
    reaction_isentropic_tt=0, RPM=30e3, shaft_power=30e3, mdot=0.15, T_t0=900, p_t0=50e5,
    radial_clearance=0.1e-3, no_blades_stator=1, no_blades_rotor=47,
    chord_over_pitch_stator=3.300, chord_over_pitch_rotor=2.518,
    t_TE_stator=0.5e-3, t_TE_rotor=0.5e-3, Ra_roughness=3e-6, admission_fraction=0.05440,
    s_ax_over_pitch_rotor=0.35, delta_s_estimate=[350, 130, 200])
# The number of samples sets map resolution; choose any of the four efficiency names from Turbine1D.
Smith_chart_results = plot_Smith_chart(
    flow_coefficient_array=np.linspace(0.25, 3., 50),
    work_coefficient_array=np.linspace(1., 6., 50), efficiency_to_plot="eta_tt",
    Turbine1D_arguments=Smith_chart_arguments, gas=gas, loss_model=loss_model)

# Now optimize the same turbine with geometry-informed scouting of the design space, followed by a mixed-integer GA.
# Define the fixed inputs to Turbine1D
Turbine1D_arguments = dict(
    reaction_isentropic_tt=0, shaft_power=30e3, mdot=0.15, T_t0=900, p_t0=50e5,
    radial_clearance=0.1e-3, chord_over_pitch_stator=3.300, chord_over_pitch_rotor=2.518,
    t_TE_stator=0.5e-3, t_TE_rotor=0.5e-3, Ra_roughness=3e-6,
    s_ax_over_pitch_rotor=0.35, delta_s_estimate=[350, 130, 200])

# Now define design variables being optimized. Only bounded variables belong in optimization_arguments; reaction may
# also be added here if it is optimized.
optimization_arguments = dict(
    RPM=[30000., 50000.], work_coefficient_tt=[4, 5.5], flow_coefficient=[1, 3],
    no_blades_stator=[1, 5], no_blades_rotor=[20, 100], admission_fraction=[0.025, 0.25],
    separation_margin=[0.05, 0.50])

# Define settings for SupersonicTurbine, which determine coupling between meanline and blading designs.
SupersonicTurbine_arguments = dict(
    t_min_stator=0.5e-3, t_min_rotor=0.5e-3, stator_nozzle_chord_scale=1.,
    print_user_warnings=False)

# Dictionary below constrains outputs; TE/pitch < 0.2 limits for stator and rotor are always applied internally.
optimization_constraints = {
    "D_tip": [None, 0.15], "l_rotor": [0.010, None],
    "l_rotor/D_mean": [None, 0.25], "alpha_1_deg": [None, 75.],
    "M_ax_1": [None, 1.], "M_ax_2": [None, 1.], "chord_over_pitch_rotor": [2.5, 3.3],
    "height_optimality": [-0.003, 0.], "width_optimality": [None, 0.],
    "t_TE_stator_min_margin": [0., None],
}

# Omit this dictionary to use tuned default settings for pymoo GA, or use it to override only selected entries.
pymoo_arguments = dict(pop_size=16, n_offsprings=16, crossover_probability=0.9,
                       crossover_eta=15., mutation_probability=1., mutation_eta=20.)

# These two inputs below determine how to perturb results from prescouting when passing them to GA. Perturbation is
# calculated based on the range of perturbation_scales (from which a random number is drawn) and design variables
# bounds range. Calculated perturbation can be also overrode by values given directly in
# perturbation_standard_deviations
perturbation_scales = (0.035, 0.012)
perturbation_standard_deviations = dict(no_blades_stator=1., no_blades_rotor=3.)

# Call the optimizer. n_threads=None uses the available logical CPUs.
optimization_results = optimize_turbine(
    gas=gas, loss_model=loss_model, Turbine1D_arguments=Turbine1D_arguments,
    optimization_arguments=optimization_arguments, SupersonicTurbine_arguments=SupersonicTurbine_arguments,
    constraints=optimization_constraints, mode="SupersonicTurbine",
    efficiency="total-to-total", efficiency_type="isentropic",
    SupersonicStatorNozzle_arguments=SupersonicStatorNozzle_arguments,
    SupersonicRotorBlades_arguments=SupersonicRotorBlades_arguments,
    n_threads=None, plot_axes=["flow_coefficient", "work_coefficient_tt"],
    pymoo_arguments=pymoo_arguments, perturbation_scales=perturbation_scales,
    perturbation_standard_deviations=perturbation_standard_deviations, seed=1, verbose=True)

# A failed search returns no feasible turbine, while keeping the closest design and failure counts.
print(optimization_results["termination"], optimization_results["statistics"])
if optimization_results["success"]:
    print("Best efficiency:", optimization_results["best_efficiency"])
    print("Best outer inputs:", optimization_results["best_design"])
    optimized_turbine = optimization_results["best_turbine"]
    optimized_turbine.turbine_meanline.print_geometry()
