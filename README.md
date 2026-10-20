# Public Transport Line Planning as a Stackelberg Game

The authority chooses which candidate lines to run (budget and fleet limited). The passengers then react: they
solve a capacity-constrained LP in which unserved demand is allowed but penalised. A genetic algorithm solves the
authority's problem; exhaustive enumeration gives the exact benchmark.

    data files -> Instance -> lines -> routes -> Instance.build()
    x (0/1 per line) -> Evaluator -> passenger LP -> objective, Gini, P95, served share
    enumerate_all (exact) and GeneticAlgorithm use the same Evaluator
    GeneticAlgorithm -> on_generation(record) -> LiveDashboard (live window / PNG frames / GIF)

Install: `pip install -e .`  Quick start: see [docs/README.md](docs/README.md).

