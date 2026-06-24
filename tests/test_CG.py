from pathlib import Path
import tomllib

import cosmock


def test_public_api_imports():
    assert "Gn" in cosmock.__all__
    assert "C_NG_to_C_G" in cosmock.__all__


def test_version_matches_pyproject():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    assert cosmock.__version__ == pyproject["project"]["version"]
