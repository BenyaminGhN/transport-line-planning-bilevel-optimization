import pytest

from tests.helpers import two_line_instance


@pytest.fixture
def tiny_instance():
    """One OD pair (demand 100), fast line A (10 min, capacity 60), slow line B (15 min, capacity 100)."""
    return two_line_instance()
