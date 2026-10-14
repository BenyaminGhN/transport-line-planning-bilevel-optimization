"""The dashboard produces frames and a GIF.   >>> CHANGED: placeholders replaced <<<"""
import matplotlib
matplotlib.use("Agg")

from ptlp.evaluate import Evaluator
from ptlp.exact import enumerate_all
from ptlp.ga import GAConfig, GeneticAlgorithm
from ptlp.io import load_instance
from ptlp.viz.live import LiveDashboard, make_gif
from ptlp.viz.network_plot import node_positions


def test_dashboard_writes_one_frame_per_generation_and_a_gif(tmp_path):
    instance = load_instance("synthetic")
    dashboard = LiveDashboard(instance, "save", tmp_path / "frames", exact=enumerate_all(Evaluator(instance)))
    GeneticAlgorithm(Evaluator(instance), GAConfig(pop_size=8, n_generations=3), dashboard).run()
    dashboard.close()
    assert len(list((tmp_path / "frames").glob("frame_*.png"))) == 3
    make_gif(tmp_path / "frames", tmp_path / "ga.gif")
    assert (tmp_path / "ga.gif").stat().st_size > 0


def test_mandl_has_no_coordinates_needed_layout_fallback_works():
    instance = load_instance("mandl")
    instance.nodes = instance.nodes[["id"]]
    assert len(node_positions(instance)) == 15
