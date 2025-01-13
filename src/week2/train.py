import torch
import tqdm
import typer

from week2.data import CorruptMNIST
from week2.evaluate import evaluate
from week2.model import MyAwesomeModel


def train(lr: float = 1e-3, epochs: int = 10, output_path: str = "models/model.pt", save_model: bool = False) -> None:
    """Train a model on MNIST."""
    model = MyAwesomeModel()
    train_dataset = CorruptMNIST("data/raw/corruptmnist_v1", split="train")
    test_dataset = CorruptMNIST("data/raw/corruptmnist_v1", split="test")

    train_set = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_set = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for images, target in tqdm.tqdm(train_set):
            output = model(images)
            loss = torch.nn.functional.cross_entropy(output, target.squeeze(1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Loss: {loss.item()} at epoch {epoch + 1}")

        if save_model:
            current_acc = evaluate(test_set=test_set, model=model)
            if epoch == 0:
                best_acc = current_acc
                torch.save(model.state_dict(), output_path)
            else:
                if current_acc > best_acc:
                    best_acc = current_acc
                    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    typer.run(train)
