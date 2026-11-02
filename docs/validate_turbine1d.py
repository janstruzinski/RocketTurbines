"""Reproduce the README's original-report Turbine1D comparisons.

Run from any directory with the package dependencies and Matplotlib installed:
    python docs/validate_turbine1d.py

The external reconstruction matches the original report's pressure ratio and
annulus, retaining a design flow coefficient and reaction. It solves for work
and entropy closure, then replays every accepted state with size_turbine.
Measured efficiency is used only in the final comparison, never in a residual.
The README documents the published inputs and the reconstruction assumptions.
"""

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from RocketTurbines.Turbine1D import Turbine1D
from RocketTurbines.TraupelLossModel import TraupelLossModel
from RocketTurbines.IdealGas import IdealGas

INCH = 0.0254
PSI = 6894.757293168
REPORTS = {
    "moffitt1958": {
        "report": "NACA RM E58F20a",
        "ntrs_id": "19630010654",
        "locators": "pp. 4-8, Tables I-II, Figures 1 and 6",
    },
    "whitney1958": {
        "report": "NACA RM E57J29",
        "ntrs_id": "19930089982",
        "locators": "pp. 2-4, Tables I-II, Figures 2 and 4(a)",
    },
    "whitney1954": {
        "report": "NACA RM E54F23",
        "ntrs_id": "19930090527",
        "locators": "pp. 4-5: original design assumptions for the shared velocity diagram",
    },
    "rohlik1961": {
        "report": "NASA TM X-475",
        "ntrs_id": "19660030323",
        "locators": "pp. 6-12, Tables I-III, Figures 5-6: first stage operating alone",
    },
    "klassen1968": {
        "report": "NASA TN D-4700",
        "ntrs_id": "19680020769",
        "locators": "pp. 2-8, Table I, Figures 1 and 5",
    },
    "moffitt1959": {
        "report": "NASA TM X-80",
        "ntrs_id": "19670022828",
        "locators": "pp. 4-6, Figures 1 and 6; geometry from NACA RM E58F20a",
    },
}
for report in REPORTS.values():
    report["url"] = (
        f"https://ntrs.nasa.gov/api/citations/{report['ntrs_id']}/downloads/{report['ntrs_id']}.pdf"
    )


class NoClearance(TraupelLossModel):
    """A diagnostic override; physical clearance and every other loss stay active."""

    def calculate_unshrouded_rotor_clearance_loss(self, *args, **kwargs):
        return 0.0


def cases():
    """Return physical inputs reduced from the reports listed in REPORTS.

    All dimensions below are converted to SI. Missing gaps, gas composition,
    common areas, width projection, and retained design coefficients are
    assumptions described individually in the README before each comparison.
    """
    # Moffitt: use the outlet hub/tip ratio for the common annulus. The stator
    # straight chord joins the first tabulated face midpoint (.200, -.013)
    # to the trailing-edge centre (-.487, 1.125), all coordinates in inches.
    common = dict(
        D=10.3 * INCH,
        h=10.3 * INCH * 0.13 / 1.87,
        cs=np.hypot(0.687, 1.138) * INCH,
        cr=1.595 * INCH,
        width_ratio=np.cos(np.deg2rad(9.6)),
        ns=32,
        nr=48,
        te_s=0.010 * INCH,
        te_r=0.010 * INCH,
        clearance=0.010 * INCH,
        alpha=72.6,
        beta=-68.3,
        phi=1 / (1 / np.tan(np.deg2rad(21.7)) - 1 / np.tan(np.deg2rad(27.6))),
        rotor="impulse_high_M",
        epsilon=1.0,
        pressure_type="ts",
        PR=30.0,
        species=["N2", "O2"],
        fractions=[0.767, 0.233],
    )
    moffitt = dict(
        common,
        name="Moffitt",
        T0=(340 - 32) * 5 / 9 + 273.15,
        p0=75 * PSI,
        u=424.8 * 0.3048,
    )
    whitney = dict(
        name="Whitney",
        D=0.85 * 6.970 * INCH,
        h=0.15 * 6.970 * INCH,
        cs=1.034 * INCH,
        cr=1.443 * INCH,
        width_ratio=np.cos(np.deg2rad(7.9)),
        ns=32,
        nr=37,
        te_s=0.016 * INCH,
        te_r=0.020 * INCH,
        clearance=0.015 * INCH,
        alpha=60.8,
        beta=-43.0,
        phi=0.724 / 0.550,
        rotor="regular",
        epsilon=1.0,
        pressure_type="tt",
        PR=2.05,
        species=["N2", "O2"],
        fractions=[0.767, 0.233],
        T0=600 * 5 / 9,
        p0=32 * 3386.389,
        inlet_critical_speed_ratio=0.498,
    )
    rohlik = dict(
        name="Rohlik",
        D=6.81 * INCH,
        h=0.268 * INCH,
        cs=0.333 * INCH,
        cr=0.307 * INCH,
        width_ratio=np.cos(np.deg2rad(3 + 31 / 60)),
        ns=93,
        nr=128,
        te_s=0.008 * INCH,
        te_r=0.008 * INCH,
        clearance=0.019 * INCH,
        axial=0.050 * INCH,
        alpha=75.0,
        beta=-69.3,
        phi=0.152 / 0.199,
        rotor="regular",
        epsilon=1.0,
        pressure_type="ts",
        species=["N2"],
        fractions=[1.0],
        T0=(52 - 32) * 5 / 9 + 273.15,
        p0=100 * PSI,
        rpm=6830.0,
        nu=0.284,
    )
    klassen = dict(
        name="Klassen",
        D=3.75 * INCH,
        h=0.217 * INCH,
        cs=0.453 * INCH,
        cr=0.312 * INCH,
        width_ratio=np.cos(np.deg2rad(7.1)),
        ns=51,
        nr=95,
        te_s=0.0048 * INCH,
        te_r=0.005 * INCH,
        clearance=0.1e-3,
        alpha=76.61,
        beta=-57.12,
        phi=0.372 / 0.531,
        rotor="impulse_low_M",
        epsilon=1.0,
        pressure_type="ts",
        species=["N2", "O2"],
        fractions=[0.767, 0.233],
        T0=300.0,
        p0=25 * PSI,
        PR=3.0,
        nu=0.377,
    )
    mk = dict(
        common,
        name="Moffitt-Klag",
        T0=(200 - 32) * 5 / 9 + 273.15,
        p0=75 * PSI,
        PR=1 / 0.033,
    )
    mk["u"] = 342 * 0.3048 * np.sqrt(mk["T0"] / 288.15)
    return [moffitt, whitney, rohlik, klassen, mk]


def prepare(c):
    """Reduce original speed/pressure data and design reaction conventions."""
    c = dict(c)
    gas = IdealGas(
        T_min=150.0,
        T_max=max(c["T0"], 301.0),
        species=c["species"],
        mass_fractions=c["fractions"],
    )
    if "inlet_critical_speed_ratio" in c:
        c["u"] = c["inlet_critical_speed_ratio"] * np.sqrt(
            2 * gas.gamma * gas.R * c["T0"] / (gas.gamma + 1)
        )
    if "rpm" in c:
        c["u"] = c["rpm"] * 2 * np.pi / 60 * c["D"] / 2
    if "PR" not in c:
        c["PR"] = (1 - (c["u"] / c["nu"]) ** 2 / (2 * gas.Cp * c["T0"])) ** (
            -gas.Cp / gas.R
        )
    if "u" not in c:
        c["u"] = c["nu"] * np.sqrt(
            2 * gas.Cp * c["T0"] * (1 - c["PR"] ** (-gas.R / gas.Cp))
        )
    c["DE"] = np.hypot(c["D"], c["h"])
    c["uE"] = c["u"] * c["DE"] / c["D"]
    c["rpm"] = c["u"] * 2 / c["D"] * 60 / (2 * np.pi)
    # Reconstruct isentropic reaction from original DESIGN pressure/velocity
    # data using the reports' gamma=1.4 convention. These reductions are frozen
    # before solving the current loss model. Only the Whitney reduction uses
    # a published design efficiency (0.88); its measured 0.85 is never an input.
    exponent = (1.4 - 1) / 1.4
    k = (1.4 - 1) / (1.4 + 1)
    if c["name"] in ["Moffitt", "Moffitt-Klag"]:
        p1_ratio = 0.86 / (1 + 0.2 * 2.43**2) ** 3.5
        p2_ratio = 0.0333
        pt2_ratio = p2_ratio * (1 + 0.2 * 1.69**2) ** 3.5
        c["Ri"] = (p1_ratio**exponent - p2_ratio**exponent) / (1 - pt2_ratio**exponent)
    elif c["name"] == "Whitney":
        t1_ideal = 0.97**exponent * (1 - k * 1.122**2)
        tt2 = (0.498 / 0.550) ** 2
        tt2_ideal = 1 - (1 - tt2) / 0.88
        t2_ideal = tt2_ideal * (1 - k * 0.735**2)
        c["Ri"] = (t1_ideal - t2_ideal) / (1 - tt2_ideal)
    elif c["name"] == "Rohlik":
        t1_ideal = 0.96**exponent * (1 - k * 0.541**2)
        t2_ideal = (1 / 1.326) ** exponent
        tt2_ideal = t2_ideal / (1 - k * 0.255**2)
        c["Ri"] = (t1_ideal - t2_ideal) / (1 - tt2_ideal)
    return c, gas


def configure(c, gas, loss, y):
    """Evaluate current Turbine1D closure at a trial work/entropy state.

    The first call finds the thermodynamic state. Its density then sets mass
    flow to the fixed source annulus; the second call evaluates all losses at
    that annulus and the consistent trailing-edge intercepts.
    """
    ss, sr, sa, psi, phi, ri = map(float, y)
    u = c["uE"]
    omega = c["rpm"] * 2 * np.pi / 60
    work = psi * u * u
    h0 = gas.Cp * c["T0"]
    if h0 - work - (phi * u) ** 2 / 2 <= 0:
        raise ValueError("Infeasible trial energy")
    t = Turbine1D()
    t.D_Euler = c["DE"]
    t.R_Euler = c["DE"] / 2
    t.p_stator = c["epsilon"] * np.pi * c["DE"] / c["ns"]
    t.p_rotor = np.pi * c["DE"] / c["nr"]
    t.c_stator = c["cs"]
    t.c_rotor = c["cr"]
    t.width_rotor = c["cr"] * c["width_ratio"]
    t.s_ax = c.get("axial", 0.35 * t.p_rotor)
    t.s_r = c["clearance"]
    t.t_TE_stator = c["te_s"] / np.cos(np.deg2rad(c["alpha"]))
    t.t_TE_rotor = c["te_r"] / np.cos(np.deg2rad(c["beta"]))
    t.t_LE_rotor = t.t_TE_rotor
    t.admission_fraction = c["epsilon"]
    t.partial_admission_rotor = "free"
    t.shrouded_rotor = False
    t.sand_grain_roughness = 0.0
    t.no_blades_stator = c["ns"]
    t.no_blades_rotor = c["nr"]
    area = np.pi * c["D"] * c["h"]
    mdot = c["epsilon"] * area * c["p0"] / c["PR"] / (gas.R * c["T0"] * 0.7) * phi * u
    t.analysis_results_at_design_point.update(
        dict(
            gas=gas,
            psi_tt=psi,
            theta_2=phi,
            R_h_tt_ideal=ri,
            RPM=c["rpm"],
            omega=omega,
            u=u,
            P_shaft=mdot * work,
            mdot_total=mdot,
            delta_h_t=work,
            T_t0=c["T0"],
            p_t0=c["p0"],
            h_t0=h0,
            h_t2=h0 - work,
            T_t2=(h0 - work) / gas.Cp,
            loss_model=loss,
        )
    )
    residual, a, b, l = t.calculate_entropy_rise(ss, sr, sa, loss)
    mdot = c["epsilon"] * area * a["rho_2"] * phi * u
    t.analysis_results_at_design_point.update(P_shaft=mdot * work, mdot_total=mdot)
    t.t_TE_stator = c["te_s"] / np.cos(a["alpha_1"])
    t.t_TE_rotor = c["te_r"] / abs(np.cos(b["beta_2_blade"]))
    residual, a, b, l = t.calculate_entropy_rise(ss, sr, sa, loss)
    return t, residual, a, b, l


def solve(c, policy, no_clearance=False, seed=None):
    """Match the report conditions, replay size_turbine, and check the result."""
    c, gas = prepare(c)
    loss = (NoClearance if no_clearance else TraupelLossModel)(
        extrapolation_method=policy,
        stator="regular",
        rotor=c["rotor"],
        incidence_loss="medium",
        warn_on_extrapolation=False,
    )

    def fun(y):
        t, res, a, b, l = configure(c, gas, loss, y)
        p2 = a["p_t2"] if c["pressure_type"] == "tt" else a["p_2"]
        return [
            *np.array(res) / 100,
            np.log(c["p0"] / p2 / c["PR"]),
            a["R_h_tt"] if c["name"] == "Klassen" else y[5] - c["Ri"],
            y[4] - c["phi"] * c["D"] / c["DE"],
        ]

    starts = [seed] if seed is not None else []
    expected = {
        "Moffitt": (6, 2, 0.2),
        "Whitney": (1.5, 1.3, 0.5),
        "Rohlik": (3.5, 0.9, 0.2),
        "Klassen": (2, 0.7, 0.2),
        "Moffitt-Klag": (6, 2, 0.2),
    }[c["name"]]
    for ent in [(50, 30, 20), (150, 100, 50), (20, 20, 10), (300, 200, 100)]:
        starts.append([*ent, *expected])
    errors = []
    for start in starts:
        try:
            opt = least_squares(
                fun,
                start,
                bounds=([0, 0, 0, 0.01, 0.01, 0], [2000, 2000, 2000, 25.0, 8.0, 1.0]),
                x_scale=[100, 100, 100, 3, 1, 0.2],
                diff_step=1e-4,
                xtol=1e-11,
                ftol=1e-11,
                gtol=1e-11,
                max_nfev=200,
            )
            err = max(abs(opt.fun))
            errors.append(("residual", err, opt.x.tolist()))
            if err < 1e-7:
                break
        except (ValueError, RuntimeError, ArithmeticError) as e:
            errors.append((type(e).__name__, str(e)))
    else:
        return dict(name=c["name"], policy=policy, epsilon=c["epsilon"], errors=errors)
    t, res, a, b, l = configure(c, gas, loss, opt.x)
    kwargs = dict(
        gas=gas,
        loss_model=loss,
        work_coefficient_tt=opt.x[3],
        flow_coefficient=opt.x[4],
        reaction_isentropic_tt=opt.x[5],
        RPM=c["rpm"],
        shaft_power=a["P_shaft"],
        mdot=a["mdot_total"],
        T_t0=c["T0"],
        p_t0=c["p0"],
        radial_clearance=c["clearance"],
        no_blades_stator=c["ns"],
        no_blades_rotor=c["nr"],
        chord_over_pitch_stator=t.c_stator / t.p_stator,
        chord_over_pitch_rotor=t.c_rotor / t.p_rotor,
        rotor_width_to_chord=c["width_ratio"],
        t_TE_stator=t.t_TE_stator,
        t_TE_rotor=t.t_TE_rotor,
        Ra_roughness=0.0,
        s_ax_over_pitch_rotor=t.s_ax / t.p_rotor,
        admission_fraction=c["epsilon"],
        delta_s_estimate=opt.x[:3],
    )
    pub = Turbine1D()
    pub.size_turbine(**kwargs)
    ap = pub.analysis_results_at_design_point
    # Independently verify pressure-based efficiencies, geometry, closure, and
    # consistency between the external solve and the public sizing method.
    pressure = ap["p_t2"] if c["pressure_type"] == "tt" else ap["p_2"]
    checks = {
        "pressure_ratio_relative_error": abs(c["p0"] / pressure / c["PR"] - 1),
        "span_relative_error": abs(pub.l_rotor / c["h"] - 1),
        "mean_diameter_relative_error": abs(pub.D_mean / c["D"] - 1),
        "inverse_public_efficiency_difference": max(
            abs(ap[k] - a[k]) for k in ["eta_ts", "eta_tt"]
        ),
    }
    for suffix, pressure_key in [("ts", "p_2"), ("tt", "p_t2")]:
        eta = ap["delta_h_t"] / (
            gas.Cp * c["T0"] * (1 - (ap[pressure_key] / c["p0"]) ** (gas.R / gas.Cp))
        )
        checks[f"independent_eta_{suffix}_difference"] = abs(eta - ap[f"eta_{suffix}"])
    entropy = [
        ap["delta_s_stator"],
        ap["delta_s_rotor"],
        ap["delta_s_rotor_additional"],
    ]
    checks["public_entropy_residual_J_per_kgK"] = max(
        abs(np.array(pub.calculate_entropy_rise(*entropy, loss)[0]))
    )
    assert checks["public_entropy_residual_J_per_kgK"] < 1e-3, checks
    assert all(
        value < 1e-5
        for key, value in checks.items()
        if key != "public_entropy_residual_J_per_kgK"
    ), checks
    # Record default-initialization behavior separately. A failure to initialize
    # does not invalidate the independently checked, seeded public replay.
    default = Turbine1D()
    try:
        default.size_turbine(**dict(kwargs, delta_s_estimate=[0.0, 0.0, 0.0]))
        difference = max(
            abs(default.analysis_results_at_design_point[k] - ap[k])
            for k in ["eta_ts", "eta_tt"]
        )
        default_replay = dict(
            success=True,
            efficiency_difference=difference,
            agreement_within_0_01_percentage_point=difference < 1e-4,
        )
    except (ValueError, RuntimeError, ArithmeticError) as error:
        default_replay = dict(
            success=False, error=type(error).__name__, message=str(error)
        )
    if c["name"] == "Rohlik" and not no_clearance:
        without = NoClearance(
            extrapolation_method=policy,
            stator="regular",
            rotor=c["rotor"],
            warn_on_extrapolation=False,
        )
        frozen = configure(c, gas, without, opt.x)[4]
        for key in [
            "zeta_aerodynamic_stator",
            "zeta_aerodynamic_rotor",
            "zeta_incidence_rotor",
            "zeta_disk_friction",
            "zeta_admission",
        ]:
            assert abs(frozen[key] - l[key]) < 1e-10, (key, frozen[key], l[key])
        assert frozen["zeta_clearance_rotor"] == 0.0
        checks["clearance_fraction_of_additional_work_loss"] = (
            l["zeta_clearance_rotor"] / l["zeta_rotor_additional"]
        )
    return dict(
        name=c["name"],
        policy=policy,
        no_clearance=no_clearance,
        epsilon=c["epsilon"],
        eta_ts=ap["eta_ts"],
        eta_tt=ap["eta_tt"],
        inverse_eta_ts=a["eta_ts"],
        inverse_eta_tt=a["eta_tt"],
        x=opt.x.tolist(),
        max_residual=err,
        PR=c["PR"],
        RPM=c["rpm"],
        checks=checks,
        default_replay=default_replay,
        reconstruction_inputs=c,
        gas=dict(
            R=gas.R,
            Cp=gas.Cp,
            gamma=gas.gamma,
            viscosity_temperature_minimum=gas._viscosity_temperature_min,
        ),
        losses={
            k: float(v)
            for k, v in l.items()
            if np.isscalar(v) and isinstance(v, (float, int, np.number))
        },
        analysis={
            k: float(v)
            for k, v in ap.items()
            if np.isscalar(v) and isinstance(v, (float, int, np.number))
        },
        height=pub.l_rotor,
        expected_height=c["h"],
        inputs={
            k: (
                v.tolist()
                if isinstance(v, np.ndarray)
                else float(v) if isinstance(v, np.floating) else v
            )
            for k, v in kwargs.items()
            if k not in ("gas", "loss_model")
        },
    )


def comparison_cases():
    """Targets are separate from case inputs and never passed to solve()."""
    moffitt, whitney, rohlik, klassen, mk = cases()
    yield moffitt, False, 0.414, "eta_ts", "reported measurement", 0.0
    yield whitney, False, 0.85, "eta_tt", "reported peak at PR_tt=2.0-2.1", 0.0
    yield rohlik, False, 0.67, "eta_tt", "reported first-stage-alone measurement", 0.0
    yield rohlik, True, 0.67, "eta_tt", "diagnostic against the same measured turbine", 0.0
    for ns, eta, uncertainty in [
        (51, 0.68, 0.0),
        (26, 0.64, 0.01),
        (16, 0.625, 0.01),
        (6, 0.56, 0.0),
    ]:
        c = dict(klassen, ns=ns, epsilon=ns / 51)
        yield c, False, eta, "eta_ts", (
            "report text"
            if not uncertainty
            else "Figure 5 visual reading; about +/-1 percentage point"
        ), uncertainty
    for ns in [32, 24, 16, 8, 4]:
        c = dict(mk, ns=ns, epsilon=ns / 32)
        yield c, False, 0.425, "eta_ts", "approximate reported mean trend, not separate exact measurements", 0.0


def plot_comparisons(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = REPO / "docs" / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "savefig.facecolor": "white",
        }
    )
    for name, filename, subtitle in [
        (
            "Klassen",
            "validation_klassen.png",
            "Pressure ratio 3.0; blade-jet speed ratio 0.377",
        ),
        (
            "Moffitt-Klag",
            "validation_moffitt_klag.png",
            "Outlet-static / inlet-total pressure 0.033; design speed",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(9.4, 5.6), layout="constrained")
        rows = sorted(
            [r for r in results if r["name"] == name and r["policy"] == "linear"],
            key=lambda r: r["epsilon"],
        )
        admission = np.array([r["epsilon"] * 100 for r in rows])
        measured = np.array([r["reference_efficiency"] * 100 for r in rows])
        if name == "Klassen":
            ax.errorbar(
                admission,
                measured,
                yerr=[r["reading_uncertainty"] * 100 for r in rows],
                fmt="o",
                color="#202020",
                capsize=4,
                markersize=6,
                label="Report (middle points read from Fig. 5)",
                zorder=4,
            )
        else:
            ax.plot(
                admission,
                measured,
                "--",
                color="#202020",
                linewidth=2,
                label="Reported approximately constant trend: 42.5%",
            )
        for policy, color, marker in [
            ("linear", "#0072B2", "s"),
            ("closest", "#D55E00", "D"),
        ]:
            data = sorted(
                [r for r in results if r["name"] == name and r["policy"] == policy],
                key=lambda r: r["epsilon"],
            )
            ax.plot(
                [r["epsilon"] * 100 for r in data],
                [r["eta_ts"] * 100 for r in data],
                color=color,
                marker=marker,
                linewidth=2,
                markersize=6,
                label=f"Turbine1D: {policy}",
            )
        ax.set(
            xlabel="Admission (%)",
            ylabel="Total-to-static efficiency (%)",
            xlim=(7, 104),
        )
        ax.set_ylim((35, 75) if name == "Klassen" else (20, 48))
        ax.set_xticks(
            admission, [f"{value:.2f}".rstrip("0").rstrip(".") for value in admission]
        )
        ax.set_title(
            f"{name}: partial-admission comparison\n", loc="left", fontsize=15, pad=12
        )
        ax.text(
            0, 1.025, subtitle, transform=ax.transAxes, color="#555555", fontsize=10
        )
        ax.grid(alpha=0.22)
        ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10)
        fig.savefig(figures / filename, dpi=200)
        plt.close(fig)


def main():
    model_files = [
        REPO / "RocketTurbines" / name
        for name in [
            "Turbine1D.py",
            "TraupelLossModel.py",
            "IdealGas.py",
            "RelaxedAitken.py",
        ]
    ]
    hashes = {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in model_files
    }
    results = []
    seeds = {}
    for (
        case,
        disabled,
        target,
        eta_key,
        reference_kind,
        uncertainty,
    ) in comparison_cases():
        for policy in ["linear", "closest"]:
            key = (case["name"], disabled, policy)
            result = solve(case, policy, disabled, seeds.get(key))
            if "errors" in result:
                raise RuntimeError(result)
            seeds[key] = result["x"]
            result.update(
                reference_efficiency=target,
                efficiency_type=eta_key,
                reference_kind=reference_kind,
                reading_uncertainty=uncertainty,
                error_percentage_points=100 * (result[eta_key] - target),
            )
            results.append(result)
            print(
                f"{case['name']:13s} admission={case['epsilon']:.5f} {policy:7s} "
                f"clearance={'off' if disabled else 'on ':3s} {eta_key}={100*result[eta_key]:.4f}%",
                flush=True,
            )
    assert len(results) == 26
    assert hashes == {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in model_files
    }
    metadata = dict(
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        source_sha256=hashes,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        reports=REPORTS,
        solver="simultaneous entropy/work reconstruction; public size_turbine replay with solved and default seeds",
        acceptance="inverse normalized residual <1e-7; public entropy residual <1e-3 J/(kg K); geometry/pressure/efficiency checks <1e-5",
    )
    destination = REPO / "docs" / "validation"
    destination.mkdir(exist_ok=True)
    with (destination / "turbine1d_literature.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            dict(metadata=metadata, results=results),
            stream,
            indent=2,
            default=lambda value: value.item(),
        )
        stream.write("\n")
    with (destination / "turbine1d_literature.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "turbine",
                "admission",
                "extrapolation",
                "clearance_losses",
                "efficiency_type",
                "literature_percent",
                "model_percent",
                "error_percentage_points",
                "reference_kind",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                dict(
                    turbine=r["name"],
                    admission=r["epsilon"],
                    extrapolation=r["policy"],
                    clearance_losses=not r["no_clearance"],
                    efficiency_type=r["efficiency_type"],
                    literature_percent=100 * r["reference_efficiency"],
                    model_percent=100 * r[r["efficiency_type"]],
                    error_percentage_points=r["error_percentage_points"],
                    reference_kind=r["reference_kind"],
                )
            )
    plot_comparisons(results)
    defaults = sum(r["default_replay"]["success"] for r in results)
    print(
        f"Validated all 26 reconstructions and seeded public replays; {defaults}/26 default initializations succeeded. Wrote results and two comparison figures."
    )


if __name__ == "__main__":
    main()
