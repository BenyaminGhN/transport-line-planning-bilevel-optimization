"""Baselines of PDF 17.   >>> NEW FILE <<<"""
import pytest

from ptlp.baselines import cost_only_baseline, uncapacitated_baseline
from ptlp.evaluate import Evaluator
from ptlp.exact import enumerate_all
from tests.helpers import two_line_instance


def test_literal_cost_only_plan_is_the_empty_plan():
    instance = two_line_instance()
    evaluator = Evaluator(instance, unmet_penalty=100)
    assert cost_only_baseline(evaluator, enumerate_all(evaluator)).x.sum() == 0


def test_cost_only_with_service_floor_picks_cheapest_plan_meeting_it():
    instance = two_line_instance()                       # A costs 6 (capacity 60), B costs 2 (capacity 100)
    evaluator = Evaluator(instance, unmet_penalty=100)
    plan = cost_only_baseline(evaluator, enumerate_all(evaluator), min_served=0.9)
    assert list(plan.x) == [0, 1]                        # B alone serves everybody for 2


def test_stackelberg_is_never_worse_than_the_uncapacitated_plan():
    instance = two_line_instance(cap_a=40, cap_b=40)
    evaluator = Evaluator(instance, unmet_penalty=100)
    best = enumerate_all(evaluator).best
    blind = uncapacitated_baseline(instance, unmet_penalty=100)
    assert best.objective <= blind.under_capacity.objective + 1e-9
    assert blind.plan.objective <= blind.under_capacity.objective + 1e-9      # blind plan looks too good
