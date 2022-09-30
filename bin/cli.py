from pathlib import Path
import torch
from torch import optim


import ser.model
import ser.transforms 
import ser.data
import ser.train
import ser.validate

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        ..., "-epochs", help="Number of epochs to run."
    ),
    batch_size: int = typer.Option(
        ..., "-batch_size", help="Data batch size."
    ),
    learning_rate: float = typer.Option(
        ..., "-learning_rate", help="Training learning rate."
    ),
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save the parameters!

    # load model
    model = ser.model.Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = ser.transforms.transforms()
    
    # get dataloaders
    training_dataloader, validation_dataloader = ser.data.get_dataloaders(batch_size, DATA_DIR, ts)

    # train
    for epoch in range(epochs):

        loss = ser.train.train(model, training_dataloader, optimizer, device)
        print(
            f"Train Epoch: {epoch}"
            f"| Loss: {loss.item():.4f}"
        )

        val_loss, val_acc = ser.validate.validate(model, validation_dataloader, device)
        print(
            f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
        )


@main.command()
def infer():
    print("This is where the inference code will go")
