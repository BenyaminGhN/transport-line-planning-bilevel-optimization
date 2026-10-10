# Public Transport Line Planning as a Stackelberg Game

Data flow (everything left of `Evaluator` is computed ONCE per instance):

    io -> Instance -> lines -> routes -> Instance.build()   (sparse matrices)
                                              |
    x (0/1 vector) -> Evaluator.evaluate(x) -> PassengerAssignment.solve(x) -> y*(x)
                                              |-> objective, Gini, P95
    ExactEnumeration  and  GeneticAlgorithm both call the SAME Evaluator.
    GeneticAlgorithm calls on_generation(GenerationRecord) after every generation
    -> viz.LiveDashboard draws it / saves a PNG frame.

Install: `pip install -e .`   Run: `python scripts/run_synthetic.py --viz save`
