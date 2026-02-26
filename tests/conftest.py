def pytest_addoption(parser):
    parser.addoption(
        "--full-models",
        action="store_true",
        default=False,
        help="Run the full model suite (all tasks, models, and variants) instead of the smoke test.",
    )
    parser.addoption(
        "--one-per-model",
        action="store_true",
        default=False,
        help="Run one variant per model (first available) for every task, instead of the smoke test.",
    )
