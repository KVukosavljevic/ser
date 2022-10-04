from pathlib import Path
from ser.params import Params, load_params, save_params
import os
import pytest

PARAMS_FILE = "model_params.json"


@pytest.fixture
def foo():
    # setup image
    # transform it
    # something else
    img = 0
    yield img
    # run after test


def test_save_params():
    test_path = Path("./tests")
    test_params = Params("test_exp", 0, 0, 0, "fergfgerg", 0.3, 2)

    save_params(test_path, test_params)

    # test whether file exists
    assert (test_path / PARAMS_FILE).is_file() is True

    # remove tmpdir
    os.remove(test_path / PARAMS_FILE)


def test_load_params():
    test_path = Path("./tests")
    test_params = Params("test_exp", 0, 0, 0, "fergfgerg", 0.3, 2)

    save_params(test_path, test_params)
    loaded_params = load_params(test_path)

    # test whether loaded params same as the one in the file
    assert loaded_params == test_params

    # remove tmpdir
    os.remove(test_path / PARAMS_FILE)
