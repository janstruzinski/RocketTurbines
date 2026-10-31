from Turbine1D import Turbine1D
from IdealGas import IdealGas
from TraupelLossModel import TraupelLossModel
from SupersonicTurbine import SupersonicTurbine

# Create a gas object
gas = IdealGas(T_max=900, T_min=450, species=["H2O","O2"], mass_fractions=[0.54, 0.46])
# Create a loss model object
loss_model = TraupelLossModel(extrapolation_method="closest", rotor="impulse_high_M")
# Create meanline analysis object
turbine = Turbine1D()
turbine.size_turbine(gas=gas, work_coefficient_tt=4.321, flow_coefficient=1.570, reaction_isentropic_tt=0, RPM=30e3,
                    shaft_power=25e3, mdot=0.15, T_0=800, p_0=50e5, radial_clearance=0.1e-3, no_blades_stator=1,
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
    shaft_power=25e3, mdot=0.15, T_0=800, p_0=50e5, radial_clearance=0.1e-3,
    no_blades_stator=1, no_blades_rotor=47, chord_over_pitch_stator=3.300,
    chord_over_pitch_rotor=2.518, t_TE_stator=0.5e-3, t_TE_rotor=0.5e-3,
    Ra_roughness=3e-6, admission_fraction=0.05440, delta_s_estimate=[350, 130, 200])

# Then create arguments for SupersonicStatorNozzle
SupersonicStatorNozzle_arguments = dict(
    contour_method="conical", number_of_nodes=51, half_cone_metal_angle=7.0,
    calculate_boundary_layer=False)

# And for SupersonicStatorBlade
SupersonicRotorBlades_arguments = dict(
    number_of_nodes=51, calculate_starting=False, calculate_boundary_layer=False,
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