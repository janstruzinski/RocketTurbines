"""Draw the README's illustrative supersonic nozzle geometry (not solved designs).

Run from a checkout with NumPy and Matplotlib installed:
    python docs/plot_supersonic_geometry.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


OUTPUT = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE, INK, GREY = "#2679b5", "#d56b17", "#243746", "#85929e"
FILL = "#dceef7"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans", "figure.dpi": 160})


def point(x, y, angle=0):
    """Rotate local tangent/radial coordinates about the machine axis."""
    return np.array([x * np.cos(angle) + y * np.sin(angle),
                     -x * np.sin(angle) + y * np.cos(angle)])


def arc(ax, radius, lo, hi, **style):
    angles = np.linspace(lo, hi, 180)
    ax.plot(radius * np.sin(angles), radius * np.cos(angles), **style)


def dimension(ax, start, end, label, offset=(0, 0), color=INK):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.2, shrinkA=0, shrinkB=0))
    midpoint = (np.array(start) + end) / 2 + np.array(offset)
    ax.text(*midpoint, label, ha="center", va="center", color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2))


def leader(ax, label, target, label_at, color=INK, align="left"):
    ax.annotate(label, xy=target, xytext=label_at, ha=align, va="center", color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.9),
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))


def annulus(ax, hub, tip, euler, title):
    for radius in (hub, tip):
        arc(ax, radius, -.7, .7, color=GREY, lw=1.3)
    arc(ax, euler, -.7, .7, color=BLUE, ls="--", lw=1.4)
    ax.set(aspect="equal", xlim=(-.58, .63), ylim=(hub - .19, tip + .17))
    ax.axis("off")
    panel = ax.get_position(original=True)
    ax.figure.text(panel.x0, panel.y1 + .025, title, fontsize=13, weight="bold", color=INK)
    for radius, label in ((hub, "$R_h$"), (euler, "$R_E$"), (tip, "$R_t$")):
        theta = -.48
        target = (radius * np.sin(theta), radius * np.cos(theta))
        ax.text(target[0] - .025, target[1], label, ha="right", va="center",
                color=BLUE if radius == euler else INK, fontsize=10)


def rectangle(ax, width, height, inner, angle=0, ghost=False):
    corners = np.array([point(x, y, angle) for x, y in
                        [(-width / 2, inner), (width / 2, inner),
                         (width / 2, inner + height), (-width / 2, inner + height)]])
    ax.add_patch(Polygon(corners, closed=True, facecolor="none" if ghost else FILL,
                         edgecolor=GREY if ghost else INK, lw=1.3, ls="--" if ghost else "-"))


def ellipse(ax, width, height, radius, angle=0):
    theta = np.linspace(0, 2 * np.pi, 220)
    coordinates = point(width / 2 * np.cos(theta), radius + height / 2 * np.sin(theta), angle)
    ax.fill(*coordinates, facecolor=FILL, edgecolor=INK, lw=1.3)


def ellipse_arc(width, height, mean, euler):
    # Bisection of the same circle/ellipse intersection described in the README.
    lo, hi = 0., np.arcsin(width / (2 * euler))
    for _ in range(60):
        mid = (lo + hi) / 2
        value = (2 * euler * np.sin(mid) / width)**2 + (2 * (euler * np.cos(mid) - mean) / height)**2 - 1
        if value > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def packing_figure():
    hub, tip = .9, 1.1
    euler, mean = np.sqrt((hub**2 + tip**2) / 2), (hub + tip) / 2
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
    fig.subplots_adjust(left=.045, right=.97, top=.84, bottom=.08, hspace=.38, wspace=.18)
    fig.suptitle("Stator nozzle exits in the annulus", fontsize=19, weight="bold", color=INK, y=.98)
    fig.text(.5, .925, "Axial view  |  Shaded: open exit   Dashed blue: Euler circle   Orange: open Euler arc",
             ha="center", color=INK, fontsize=11)

    ax = axes[0, 0]
    annulus(ax, hub, tip, euler, "(a) Multiple MOC nozzles")
    delta, gap = .4, .025
    width = (2 * hub * np.sin(delta / 2) - gap) / np.cos(delta / 2)
    # Show the intermediate regime: tip contact while the actual inner side still clears the hub.
    maximum = np.sqrt(tip**2 - width**2 / 4) - hub
    height = .94 * maximum
    inner = max(hub, min(euler - height / 2, hub + maximum - height))
    for angle in (-delta / 2, delta / 2):
        rectangle(ax, width, maximum, hub, angle, ghost=True)
        extension = np.array([point(x, y, angle) for x, y in
                              [(-width / 2, inner), (-width / 2, hub),
                               (width / 2, hub), (width / 2, inner)]])
        ax.plot(extension[:, 0], extension[:, 1], color=GREY, ls=":", lw=1.)
        rectangle(ax, width, height, inner, angle)
        ax.plot(*point(0, inner + height / 2, angle), "+", color=INK, ms=8)
    angle = delta / 2
    dimension(ax, point(-width / 2, hub - .035, angle), point(width / 2, hub - .035, angle),
              "$w$", offset=(0, -.023))
    dimension(ax, point(width / 2 + .025, inner, angle), point(width / 2 + .025, inner + height, angle),
              "$H$", offset=(.028, 0))
    theta = np.arcsin(width / (2 * euler))
    arc(ax, euler, angle - theta, angle + theta, color=ORANGE, lw=4)
    leader(ax, "$L_e$", point(0, euler, angle), (.26, 1.20), ORANGE)
    leader(ax, "$H_{max}$ (dashed limit)", point(-width / 2, hub, -angle), (-.33, .80), align="center")
    left_inner = point(width / 2, inner, -angle)
    right_inner = point(-width / 2, inner, angle)
    dimension(ax, left_inner, right_inner, "$g$", offset=(0, -.036))
    arc(ax, tip + .065, -angle, angle, color=GREY, lw=1)
    ax.text(0, tip + .083, "$\\delta=p_s/R_E$", ha="center", color=INK)
    ax.text(.02, hub - .155, "Start at $R_E$; shift inward to fit tip", ha="center", color=INK, fontsize=10)

    ax = axes[0, 1]
    # Choose width at Euler radius, then show a taller exit shifted inward to retain tip contact.
    shub, stip = .65, 1.25
    seuler = np.sqrt((shub**2 + stip**2) / 2)
    annulus(ax, shub, stip, seuler, "(b) Single MOC nozzle")
    half = 2 * (stip**2 - seuler**2) / (np.sqrt(seuler**2 + 8 * stip**2) + 3 * seuler)
    width = 2 * np.sqrt(stip**2 - (seuler + half)**2)
    maximum = np.sqrt(stip**2 - width**2 / 4) - shub
    height = .86 * maximum
    inner = max(shub, min(seuler - height / 2, shub + maximum - height))
    rectangle(ax, width, maximum, shub, ghost=True)
    rectangle(ax, width, height, inner)
    ax.plot(0, inner + height / 2, "+", color=INK, ms=10)
    theta = np.arcsin(width / (2 * seuler))
    arc(ax, seuler, -theta, theta, color=ORANGE, lw=4)
    dimension(ax, (-width / 2, shub - .055), (width / 2, shub - .055), "$w$", offset=(0, -.03))
    dimension(ax, (width / 2 + .025, inner),
              (width / 2 + .025, inner + height), "$H$", offset=(.025, 0))
    leader(ax, "$H_{max}$ reference", (width / 2, shub), (.62, .60), align="center")
    leader(ax, "Centre shifted from $R_E$", (0, inner + height / 2), (-.32, 1.36), align="center")
    ax.text(0, seuler + .095, "$L_e$", color=ORANGE, ha="center")
    ax.set(xlim=(-.8, .8), ylim=(.51, 1.45))

    ax = axes[1, 0]
    annulus(ax, hub, tip, euler, "(c) Multiple de Laval nozzles")
    width, height, delta = .48, .18, .53
    angle = delta / 2
    for phi in (-angle, angle):
        ellipse(ax, width, height, mean, phi)
        ax.plot(*point(0, mean, phi), "+", color=INK)
    theta = ellipse_arc(width, height, mean, euler)
    arc(ax, euler, angle - theta, angle + theta, color=ORANGE, lw=4)
    arc(ax, euler, -angle + theta, angle - theta, color="#b53d45", lw=3)
    leader(ax, "$p_s-L_e$ (arc)", (0, euler), (-.03, 1.20), "#b53d45", align="center")
    support = np.hypot(width / 2 * np.cos(angle), height / 2 * np.sin(angle))
    centre_x, centre_y = point(0, mean, angle)
    # Support coordinates along the horizontal line through the two centres.
    ax.plot([-centre_x, centre_x], [centre_y, centre_y], color=GREY, ls=":", lw=1)
    dimension(ax, (-centre_x + support, centre_y), (centre_x - support, centre_y),
              "$g$ (straight)", offset=(0, -.065))
    dimension(ax, point(-width / 2, mean - .14, angle), point(width / 2, mean - .14, angle),
              "$w$", offset=(0, -.025))
    dimension(ax, point(width / 2 + .035, mean - height / 2, angle),
              point(width / 2 + .035, mean + height / 2, angle), "$H$", offset=(.025, 0))
    leader(ax, "Centres at $R_m$", (-centre_x, centre_y), (-.36, .76), align="center")

    ax = axes[1, 1]
    annulus(ax, hub, tip, euler, "(d) Single de Laval nozzle")
    ellipse(ax, width, height, mean)
    ax.plot(0, mean, "+", color=INK, ms=10)
    arc(ax, euler, -theta, theta, color=ORANGE, lw=4)
    dimension(ax, (-width / 2, mean - height / 2 - .045),
              (width / 2, mean - height / 2 - .045), "$w$", offset=(0, -.025))
    dimension(ax, (width / 2 + .035, mean - height / 2),
              (width / 2 + .035, mean + height / 2), "$H$", offset=(.025, 0))
    dimension(ax, (.43, hub), (.43, tip), "$H_a=H_{max}$", offset=(.09, 0))
    leader(ax, "Centre at $R_m$", (0, mean), (-.16, .77), align="center")
    ax.text(0, 1.15, "$\\varepsilon=L_e/(2\\pi R_E)$", ha="center", color=ORANGE)
    fig.text(.5, .025, "Illustrative geometry; each panel uses its own scale. Radial labels identify concentric circles.",
             ha="center", fontsize=10, color=GREY)
    save(fig, "supersonic_nozzle_packing.png")


def dimensions_figure():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.subplots_adjust(left=.035, right=.98, top=.80, bottom=.16, wspace=.16)
    fig.suptitle("Nozzle scale and measured stator chord", fontsize=18, weight="bold", color=INK, y=.98)
    fig.text(.5, .90, "Local nozzle-axis view  |  Schematics before boundary-layer correction",
             ha="center", color=INK)
    for ax, moc in zip(axes, (True, False)):
        # Equal wall opening widths at a common nozzle-axis coordinate; the upper
        # wall then extends to meet the tilted machine-axial exit plane.
        throat, end, lower_exit, upper_exit = .16, .50, 1.7, 2.55
        lower_x = np.linspace(0, lower_exit, 150)
        growth = np.sin(np.pi / 2 * lower_x / lower_exit)**1.3 if moc else lower_x / lower_exit
        lower_y = -(throat + (end - throat) * growth)
        upper_x = np.r_[lower_x, upper_exit]
        upper_y = np.r_[-lower_y, end]
        polygon = np.vstack([np.column_stack((lower_x, lower_y)),
                             np.column_stack((upper_x, upper_y))[::-1]])
        ax.add_patch(Polygon(polygon, facecolor=FILL, edgecolor="none"))
        ax.plot(lower_x, lower_y, color=INK, lw=1.7)
        ax.plot(upper_x, upper_y, color=INK, lw=1.7)
        ax.plot([lower_exit, upper_exit], [-end, end], "--", color=BLUE, lw=1.4)
        ax.plot([-.1, 2.75], [0, 0], ":", color=GREY, lw=1)
        exit_centre = np.array([(lower_exit + upper_exit) / 2, 0.])
        ax.plot([0, exit_centre[0]], [0, 0], "o", color=ORANGE, ms=4)
        dimension(ax, (0, -.90), (exit_centre[0], -.90), "$c_s$ before chord multiplier", offset=(0, -.09))
        ax.plot([0, 0], [-.92, .05], color=GREY, ls=":", lw=.8)
        ax.plot([exit_centre[0], exit_centre[0]], [-.92, .05], color=GREY, ls=":", lw=.8)
        dimension(ax, (-.12, -throat), (-.12, throat), "$b^*$" if moc else "$d^*$", offset=(-.10, 0))
        dimension(ax, (lower_exit, -end), (lower_exit, end), "$b_e$" if moc else "$d_e=H$", offset=(-.12, 0))
        leader(ax, "Exit centre", exit_centre, (1.95, .22), align="center")
        leader(ax, "Machine axial\nexit plane", (2.28, .18), (2.48, .76), align="center")
        leader(ax, "Throat centre", (0, 0), (.05, .61))
        machine_angle = -np.arctan((upper_exit - lower_exit) / (2 * end))
        machine_end = exit_centre + .82 * np.array([np.cos(machine_angle), np.sin(machine_angle)])
        ax.annotate("", xy=machine_end, xytext=exit_centre,
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
        ax.text(2.75, -.68, "Machine axis", ha="center", color=GREY, fontsize=10)
        angles = np.linspace(machine_angle, 0, 60)
        ax.plot(exit_centre[0] + .35 * np.cos(angles), .35 * np.sin(angles), color=GREY, lw=1)
        ax.text(2.60, -.13, "$\\alpha_m$", ha="center", color=INK)
        ax.set_title("MOC: $A^*_{total}/n=b^*H$, reference length $w$" if moc else
                     "De Laval: $A^*_{total}/n=\\pi(d^*)^2/4$, reference length $d^*$",
                     fontsize=12, loc="left", color=INK, pad=15)
        ax.set(aspect="equal", xlim=(-.5, 3.05), ylim=(-1.16, .96))
        ax.axis("off")
    fig.text(.5, .065,
             "$w=b_e/\\cos\\alpha_m$ (MOC);  $w=d_e/\\cos\\alpha_m$ (de Laval).  "
             "$\\alpha_m$: outlet metal/nozzle-axis angle from the machine axis.",
             ha="center", color=INK, fontsize=11)
    save(fig, "supersonic_nozzle_dimensions.png")


def save(figure, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / name, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    packing_figure()
    dimensions_figure()
