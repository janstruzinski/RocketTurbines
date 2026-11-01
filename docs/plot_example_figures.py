"""Regenerate the README figures from the current example.py inputs.

Run with the package dependencies, Kaleido and Chrome installed:
    python docs/plot_example_figures.py

Use --only geometry or --only smith to regenerate one group. Geometry is coupled
with boundary-layer calculations disabled; example.py itself is not changed.
"""

import argparse
import ast
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from RocketTurbines.IdealGas import IdealGas
from RocketTurbines.TraupelLossModel import TraupelLossModel
from RocketTurbines.SupersonicTurbine import SupersonicTurbine
from RocketTurbines.GA_optimizer.Smith_chart import plot_Smith_chart

OUTPUT = ROOT / "docs" / "figures"
EXAMPLE = ROOT / "example.py"


def example_assignment(tree, name, namespace):
    """Evaluate the first assignment to a name without running the full example."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return eval(compile(ast.Expression(node.value), str(EXAMPLE), "eval"), namespace)
    raise ValueError(f"No assignment to {name} in {EXAMPLE}")


def geometry_figures(tree, namespace):
    for name in ("Turbine1D_arguments", "SupersonicStatorNozzle_arguments", "SupersonicRotorBlades_arguments"):
        namespace[name] = example_assignment(tree, name, namespace)
    for name in ("SupersonicStatorNozzle_arguments", "SupersonicRotorBlades_arguments"):
        namespace[name]["calculate_boundary_layer"] = False
    stage = example_assignment(tree, "supersonic_turbine", namespace)
    print("Coupling the example turbine with boundary-layer calculations disabled...", flush=True)
    stage.couple_meanline_and_blading()
    assert not stage.stator_shape.calculate_boundary_layer
    assert not stage.rotor_shape.calculate_boundary_layer

    # Capture the real plotting methods' figures without opening their interactive windows.
    figures = []
    with patch.object(go.Figure, "show", lambda figure, **kwargs: figures.append(figure)):
        stage.plot_3D_geometry()
    figures[0].write_image(OUTPUT / "example_turbine_3d.png", width=1400, height=1000, scale=1)
    print("Saved example_turbine_3d.png", flush=True)

    with patch.object(plt, "show"):
        stage.plot_2D_geometry(number_of_rotor_blades=10)
    figure = plt.gcf()
    figure.savefig(OUTPUT / "example_turbine_2d.png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print("Saved example_turbine_2d.png (10 rotor pitches)", flush=True)


def smith_figure(tree, namespace):
    namespace["Smith_chart_arguments"] = example_assignment(tree, "Smith_chart_arguments", namespace)
    print("Calculating the Smith chart using the example's coefficient grid...", flush=True)
    with patch.object(plt, "show"):
        result = example_assignment(tree, "Smith_chart_results", namespace)
    result["figure"].savefig(OUTPUT / "example_smith_chart.png", dpi=180, facecolor="white")
    plt.close(result["figure"])
    print(f"Saved example_smith_chart.png: {result['efficiencies'].count()} valid samples, "
          f"maximum efficiency {result['best_efficiency']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("geometry", "smith"))
    args = parser.parse_args()
    tree = ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    namespace = dict(np=np, IdealGas=IdealGas, TraupelLossModel=TraupelLossModel,
                     SupersonicTurbine=SupersonicTurbine, plot_Smith_chart=plot_Smith_chart)
    for name in ("gas", "loss_model"):
        namespace[name] = example_assignment(tree, name, namespace)
    namespace["loss_model"].warn_on_extrapolation = False
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.only != "smith":
        geometry_figures(tree, namespace)
    if args.only != "geometry":
        smith_figure(tree, namespace)


if __name__ == "__main__":
    main()
