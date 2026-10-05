from Turbine1D import Turbine1D
from IdealGas import IdealGas
from TraupelLossModel import TraupelLossModel

gas = IdealGas(T_max=900, T_min=450, species=["H2O","O2"], mass_fractions=[0.54, 0.46])
loss_model = TraupelLossModel(rotor="impulse_high_M")
turbine = Turbine1D()
turbine.size_turbine(gas=gas, work_coefficient_tt=5.82, flow_coefficient=1.63, reaction_isentropic_tt=0, RPM=30e3,
                    shaft_power=21e3, mdot=0.1, T_0=900, p_0=40e5, radial_clearance=0.1e-3, no_blades_stator=1,
                    no_blades_rotor=43, chord_over_pitch_stator=1.37, chord_over_pitch_rotor=3.3,
                    t_TE_stator=0.5e-3, t_TE_rotor=0.5e-3, Ra_roughness=3e-6, loss_model=loss_model,
                    admission_fraction=0.084, delta_s_estimate=[350, 130, 200])
