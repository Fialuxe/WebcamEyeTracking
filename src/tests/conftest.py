import sys, os
# Add src/ to path so "from main.xxx import ..." resolves in all test files
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: hardware/timing tests that require a physical camera (run with -m slow)",
    )
