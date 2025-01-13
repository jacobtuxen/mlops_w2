import torch
import tqdm
from torch.utils.data import DataLoader

from week2.model import MyAwesomeModel


def evaluate(test_set: DataLoader, model: MyAwesomeModel) -> None:
    """Evaluate a trained model."""
    correct = 0
    total = 0
    for images, target in tqdm.tqdm(test_set):
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target.squeeze(1)).sum().item()
    return correct / total
