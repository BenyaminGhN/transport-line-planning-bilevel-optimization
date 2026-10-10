"""LiveDashboard: callback that redraws a 2x2 figure after every GA generation."""
from __future__ import annotations

from pathlib import Path

from ..exact import ExactResult
from ..ga import GenerationRecord
from ..instance import Instance


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
        raise NotImplementedError

    def __call__(self, record: GenerationRecord) -> None:
        """Input: GenerationRecord from the GA. Output: None (updates/saves figure)."""
        raise NotImplementedError

    def close(self) -> None: raise NotImplementedError


def make_gif(frames_dir: Path, out_path: Path, fps: int = 4) -> None:
    """Input: folder of frame_XXXX.png. Output: animated GIF at out_path."""
    raise NotImplementedError
