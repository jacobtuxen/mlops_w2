import os

import pytest
from torch.utils.data import Dataset
from week2.data import CorruptMNIST

from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_dataset_type():
    """Test the MyDataset class."""
    dataset = CorruptMNIST(_PATH_DATA)
    assert isinstance(dataset, Dataset), "The dataset is not a valid PyTorch Dataset"


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    dataset_test = CorruptMNIST(_PATH_DATA, split="test")
    dataset_train = CorruptMNIST(_PATH_DATA, split="train")
    assert len(dataset_test) == 5000, f"len(dataset_test)={len(dataset_test)}"
    assert len(dataset_train) == 30000, f"len(dataset_train)={len(dataset_train)}"
    assert all(sample.shape == (1, 28, 28) for sample, _ in dataset_test), "Images have incorrect shape"
    assert all(0 <= target <= 9 for _, target in dataset_test), "Target values are incorrect"
