"""LiveDashboard: callback that redraws a 2x2 figure after every GA generation.

>>> CHANGED (was stubs). Same interface as in the skeleton. GIFs are made with Pillow (no imageio needed). <<<
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..exact import ExactResult
from ..ga import GenerationRecord
from ..instance import Instance
from .network_plot import node_positions, plot_plan
from .panels import plot_convergence, plot_objective_space, plot_population_matrix


class LiveDashboard:
    """Usage:
        dash = LiveDashboard(instance, mode="save", out_dir=Path("results/frames"), exact=exact_result)
        GeneticAlgorithm(evaluator, cfg, on_generation=dash).run()
        dash.close(); make_gif(Path("results/frames"), Path("results/ga.gif"))

    Layout:  [ best network + loads | convergence curves ]
             [ objective space      | population matrix  ]
    Modes:  'live' -> plt.pause (needs interactive backend)
            'save' -> frame_XXXX.png per generation
            'both' | 'off'
    """

    def __init__(self, instance: Instance, mode: str = "save", out_dir: Path | None = None,
                 every: int = 1, exact: ExactResult | None = None):
        """every: draw every n-th generation. exact: enables the optimum overlays."""
        if mode not in ("live", "save", "both", "off"):
            raise ValueError("mode must be 'live', 'save', 'both' or 'off'")
        if mode in ("save", "both") and out_dir is None:
            raise ValueError("out_dir is required to save frames")
        self.instance, self.mode, self.every, self.exact = instance, mode, max(1, every), exact
        self.out_dir = Path(out_dir) if out_dir else None
        self.history: list[GenerationRecord] = []
        self.pos = node_positions(instance)
        self.figure = self.axes = None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            for old in self.out_dir.glob("frame_*.png"):   # a new run must not mix with old frames
                old.unlink()
        if mode != "off":
            self.figure, self.axes = plt.subplots(2, 2, figsize=(12, 9))
            if mode in ("live", "both"):
                plt.ion()

    def __call__(self, record: GenerationRecord) -> None:
        """Input: GenerationRecord from the GA. Output: None (updates/saves figure)."""
        self.history.append(record)
        if self.mode == "off" or record.generation % self.every:
            return
        (ax_net, ax_conv), (ax_obj, ax_pop) = self.axes
        best = record.best_result
        plot_plan(self.instance, best.x, best.assignment, ax_net, self.pos,
                  title=f"Best plan: {''.join(map(str, best.x))}  objective {best.objective:.3f}  "
                        f"served {best.served_fraction:.0%}")
        plot_convergence(self.history, ax_conv, self.exact.best.objective if self.exact else None)
        plot_objective_space(record, ax_obj, self.exact.table if self.exact else None,
                             (self.exact.best.c_op_norm, self.exact.best.c_pass_norm) if self.exact else None)
        plot_population_matrix(record, ax_pop, [line.id for line in self.instance.lines])
        self.figure.suptitle(f"Generation {record.generation}   |   LP solves {record.n_lp_solves}"
                             f"   |   diversity {record.diversity:.2f}")
        self.figure.tight_layout()
        if self.mode in ("save", "both"):
            self.figure.savefig(self.out_dir / f"frame_{record.generation:04d}.png", dpi=80)
        if self.mode in ("live", "both"):
            plt.pause(0.05)

    def close(self) -> None:
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None


def make_gif(frames_dir: Path, out_path: Path, fps: int = 4) -> None:
    """Input: folder of frame_XXXX.png. Output: animated GIF at out_path."""
    from PIL import Image
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    images = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
