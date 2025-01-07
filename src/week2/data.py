from pathlib import Path

import typer
import torch
import glob
from torch.utils.data import Dataset


class CorruptMNIST(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path, split: str = 'test') -> None:
        self.data_path = raw_data_path
        self.split = split

        image_paths = sorted(glob.glob(f"{self.data_path}/{split}_images*.pt"))
        target_paths = sorted(glob.glob(f"{self.data_path}/{split}_target*.pt"))

        self.images_files = torch.cat([torch.load(file) for file in image_paths])
        self.target_files = torch.cat([torch.load(file) for file in target_paths])

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images_files)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        
        # Normalize the images
        self.images_files[index] = (self.images_files[index] - self.images_files[index].mean()) / self.images_files[index].std()

        return self.images_files[index].unsqueeze(0), self.target_files[index].unsqueeze(0)

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        print("Preprocessing data...")
        # Preprocess the data and save it to the output folder
        torch.save(self.images_files, output_folder / f"{self.split}_images.pt")
        torch.save(self.target_files, output_folder / f"{self.split}_target.pt")

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = CorruptMNIST(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)