import torch
from torch import optim
from dataclasses import dataclass
import git
import sys

import ser.model
import ser.transforms 
import ser.data
import ser.train
import ser.validate
import ser.manage_exp_dir

@dataclass
class HyperParams:
    name: str
    epochs: int
    batch_size: str
    learning_rate: float
    git_commit_hash : str

import typer

main = typer.Typer()

def check_commits(repo):

    changed = [ item.a_path for item in repo.index.diff(None) ]
    print(repo, changed, repo.untracked_files) 

    if (len(repo.untracked_files) + len(changed) ) > 0:
        print('There are uncommitted changes with this run.')
        print('Changed: ', changed)
        print('Untracked: ', repo.untracked_files)
        sys.exit("Error message")
    
    print('Everything good')


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        ..., "--epochs", help="Number of epochs to run."
    ),
    batch_size: int = typer.Option(
        ..., "--batch_size", help="Data batch size."
    ),
    learning_rate: float = typer.Option(
        ..., "--learning_rate", help="Training learning rate."
    ),
    track_commits: int = typer.Option(
        ..., "--track_commits", help="Output warning if there are uncommitted changes."
    ),
):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    if track_commits == 1: check_commits(repo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyperparams = HyperParams(name, epochs, batch_size, learning_rate, sha)

    # Create experiment dir
    exp_dir = ser.manage_exp_dir.create_exp_dir(name)

    # Save model parameters!
    ser.manage_exp_dir.save_model_params(exp_dir, hyperparams)

    # load model
    model = ser.model.Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = ser.transforms.transforms()
    
    # get dataloaders
    training_dataloader, validation_dataloader = ser.data.get_dataloaders(batch_size, ts)

    # train
    best_val_acc = 0
    best_epoch = 0
    best_model = model

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model = model

    # Update model params
    ser.manage_exp_dir.save_model_params(exp_dir, hyperparams, best_val_acc=best_val_acc, best_val_acc_epoch=best_epoch)
    ser.manage_exp_dir.save_model(exp_dir, best_model, name)
    

@main.command()
def infer():
    print("This is where the inference code will go")
