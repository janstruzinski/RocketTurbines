"""Redraw the coefficient data used by TraupelLossModel for the README.

Run from a checkout with matplotlib installed:
    python docs/plot_traupel_coefficients.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'RocketTurbines'))
from TraupelLossModel import TraupelLossModel

OUTPUT = ROOT / 'docs' / 'figures'
OUTPUT.mkdir(parents=True, exist_ok=True)
model = TraupelLossModel('closest', warn_on_extrapolation=False)
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.22,
                     'lines.linewidth': 2, 'legend.fontsize': 9, 'figure.dpi': 150})


def values(function, coordinates, *args):
    return [function(float(x), *args) for x in coordinates]


def save(figure, name):
    figure.savefig(OUTPUT / name, dpi=170, facecolor='white')
    plt.close(figure)


# Profile loss: all Mach curves start at zero and retain their separate upper limits.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
curves = [(1, 1.29, 'regular', 'tab:blue'), (2, 1.4, 'near_sonic_outlet', 'tab:orange'),
          (3, .75, 'impulse_low_M', 'tab:red'), (4, 1.3, 'impulse_high_M', 'tab:green')]
for curve, upper, label, color in curves:
    mach = np.linspace(0, upper, 180)
    axes[0].plot(mach, values(getattr(model, f'interpolator_chi_M_curve_{curve}'), mach), label=label, color=color)
axes[0].set(xlabel='Outlet Mach number', ylabel=r'$\chi_M$', title='Mach correction')
axes[0].legend()
angles = np.linspace(20, 120, 201)
for outlet in [15, 20, 25, 30, 35, 45]:
    axes[1].plot(angles, values(model.calculate_zeta_p0, angles, outlet), label=f'{outlet}°')
axes[1].set(xlabel='Traupel inlet angle (degrees)', ylabel=r'$\zeta_{p0}$', title='Basic profile loss')
axes[1].legend(title='Outlet angle')
ratio = np.linspace(1, 10, 181)
for blockage in [.04, .08, .12, .16, .20]:
    axes[2].plot(ratio, values(model.calculate_zeta_h, ratio, blockage, 1.5e5), label=f'{blockage:.2f}')
axes[2].set(xlabel=r'$\delta_a/(\chi_M\chi_R\zeta_{p0})$', ylabel=r'$\zeta_{h,\infty}$',
            title='Trailing-edge underpressure')
axes[2].legend(title=r'$\delta_a$')
save(fig, 'traupel_profile_coefficients.png')

# Reynolds-number corrections and disk friction use the logarithmic graph axes.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
reynolds = np.geomspace(1e4, 1e7, 220)
for roughness in [1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 2e-3]:
    axes[0].semilogx(reynolds, values(model.calculate_chi_R, reynolds, roughness), label=f'{roughness:g}')
axes[0].set(xlabel='Chord Reynolds number', ylabel=r'$\chi_R$', title='Profile Reynolds / roughness correction')
axes[0].legend(title=r'$k_s/c$')
for roughness in [0, 2e-4, 5e-4, 1e-3, 2e-3]:
    axes[1].semilogx(reynolds, values(model.calculate_c_f, reynolds, roughness), label=f'{roughness:g}')
axes[1].set(xlabel='Hydraulic-diameter Reynolds number', ylabel=r'$c_f$', title='Endwall friction factor')
axes[1].legend(title=r'$k_s/d_h$')
disk_reynolds = np.geomspace(7e4, 1e7, 180)
axes[2].loglog(disk_reynolds, values(model.calculate_C_M, disk_reynolds))
axes[2].set(xlabel='Disk Reynolds number', ylabel=r'$C_M$', title='Disk-friction coefficient')
save(fig, 'traupel_reynolds_coefficients.png')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
length = np.linspace(0, .2, 180)
for speed in [.5, .7, .9]:
    axes[0].plot(length, values(model.calculate_zeta_f, length, speed), label=f'{speed:.1f}')
axes[0].set(xlabel=r'$l/D_m$', ylabel=r'$\zeta_f$', title='Fanning loss')
axes[0].legend(title=r'$\nu$')
turning = np.linspace(10, 140, 180)
for velocity_ratio in [.2, .4, .6, .8, 1.0]:
    axes[1].plot(turning, values(model.calculate_F, turning, velocity_ratio), label=f'{velocity_ratio:.1f}')
axes[1].set(xlabel='Turning angle (degrees)', ylabel=r'$F$', title='Endwall / secondary-flow factor')
axes[1].legend(title='Inlet / outlet speed')
for level, upper in [('high', 43.7), ('medium', 43.7), ('low', 60)]:
    incidence_model = TraupelLossModel('closest', False, incidence_loss=level)
    angle = np.linspace(-50, upper, 220)
    axes[2].plot(angle, values(incidence_model.calculate_z, angle), label=level)
axes[2].set(xlabel='Corrected incidence (degrees)', ylabel=r'$z$', title='Incidence factor')
axes[2].legend()
save(fig, 'traupel_secondary_incidence_coefficients.png')

# Solid lines retain each panel's measured interval; dashed parts are prepared grid extensions.
fig, axes = plt.subplots(2, 2, figsize=(10.6, 8), layout='constrained')
panels = [(.30, 1.75, 4), (.35, 1.5, 3.5), (.40, 1.25, 3), (.50, 1, 2.25)]
for axis, (sine, lower, upper) in zip(axes.flat, panels):
    velocity = np.linspace(1, 4, 181)
    for clearance in [.01, .02, .03, .045, .06]:
        coefficient = np.array([
            model.calculate_K_sigma(np.arcsin(sine), float(v), clearance) for v in velocity])
        line, = axis.plot(velocity, np.where((velocity >= lower) & (velocity <= upper), coefficient, np.nan),
                          label=f'{clearance:g}')
        axis.plot(velocity, np.where((velocity <= lower) | (velocity >= upper), coefficient, np.nan),
                  '--', color=line.get_color())
    axis.set(xlabel=r'$|\Delta w_\theta|/v_{ax}$', ylabel=r'$K_\sigma$',
             title=rf'$\sin(\beta_{{2,T}})={sine:.2f}$')
    axis.legend(title=r'$x=s_r/c-0.002$', ncol=2)
save(fig, 'traupel_clearance_coefficients.png')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
pressure_ratio = np.linspace(.3, 1, 220)
for teeth in [4, 5, 6, 8, 10, 12, 15]:
    axes[0].plot(pressure_ratio, values(model.calculate_phi, pressure_ratio, 7, teeth), label=str(teeth))
axes[0].set(xlabel=r'$p_2/p_1$', ylabel=r'$\Phi$', title=r'Half-labyrinth flow at $a/s_r=7$')
axes[0].legend(title='Teeth', ncol=2)
teeth = np.linspace(4, 15, 180)
for spacing in [5, 7, 10, 13]:
    axes[1].plot(teeth, values(model.interpolator_critical_pressure_ratio, teeth, spacing), label=str(spacing))
axes[1].set(xlabel='Number of teeth', ylabel=r'$(p_2/p_1)_{crit}$', title='Critical pressure ratio')
axes[1].legend(title=r'$a/s_r$')
save(fig, 'traupel_labyrinth_coefficients.png')
print(f'Wrote five coefficient figures to {OUTPUT}')
