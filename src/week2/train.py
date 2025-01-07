import torch
import tqdm
import typer

from data import CorruptMNIST
from model import MyAwesomeModel
from evaluate import evaluate

def train(lr: float = 1e-3, epochs: int = 10, output_path: str = 'models/model.pt') -> None:
    """Train a model on MNIST."""

    model = MyAwesomeModel()
    train_dataset = CorruptMNIST("data/raw/corruptmnist_v1", split='train')
    test_dataset = CorruptMNIST("data/raw/corruptmnist_v1", split='test')

    train_set = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_set = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    # breakpoint()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for images, target in tqdm.tqdm(train_set):
            output = model(images)
            loss = torch.nn.functional.cross_entropy(output, target.squeeze(1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Loss: {loss.item()} at epoch {epoch + 1}")
        current_acc = evaluate(test_set=test_set, model = model)

        if epoch == 0:
            best_acc = current_acc
            torch.save(model.state_dict(), output_path)
        else:
            if current_acc > best_acc:
                best_acc = current_acc
                torch.save(model.state_dict(), output_path)
        print(best_acc)

if __name__ == "__main__":
    typer.run(train)