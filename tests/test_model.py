import re

import pytest
import torch

from week2.model import MyAwesomeModel


def test_model_shape():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.shape == (1, 10), "Output shape is incorrect"


def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))
    with pytest.raises(ValueError, match=re.escape("Expected each sample to have shape [1, 28, 28]")):
        model(torch.randn(1, 1, 28, 29))


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10), f"Output shape is incorrect for batch size {batch_size}"
