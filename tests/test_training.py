import os

import pytest

from tests import _PATH_DATA
from week2.train import train


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_train():
    train(lr=1e-3, epochs=1, output_path="models/model.pt", save_model=False)
    assert True
