"""Keep offline unit checks selectable without hiding model-code failures."""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "finbert" in item.fixturenames:
            item.add_marker(pytest.mark.integration)
