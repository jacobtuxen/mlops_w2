from week2.evaluate import evaluate
from week2.data import CorruptMNIST
from week2.model import MyAwesomeModel
import torch

def test_evalaute():
    dataset = CorruptMNIST("data/raw/corruptmnist_v1", split="test")
    dataset_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    model = MyAwesomeModel()
    assert isinstance(evaluate(test_set=dataset_dataloader, model=model), float), "The evaluation score is not a float"