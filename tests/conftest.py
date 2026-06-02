"""pytest config for the test suite."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: end-to-end tests that load the predictor / catalogue "
        "(skipped when those assets are absent)",
    )
