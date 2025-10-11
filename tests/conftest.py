import pytest
import os
import sys


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    # Ensure tests run with repo root as CWD and make the package importable
    repo_root = os.path.dirname(os.path.dirname(__file__))
    # Add repo root to sys.path so tests can import scanner_manager
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    monkeypatch.chdir(repo_root)
