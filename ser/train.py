from pytest import param
from torch import optim
import torch
import torch.nn.functional as F
import numpy as np

from ser.model import Net
from ser.transforms import flip
from utils import utils
from ser.params import save_params


def train(run_path, params, train_dataloader, val_dataloader, device, random_flip):

    global plotter
    plotter = utils.VisdomLinePlotter(env_name="Training plots")

    # setup model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # train
    for epoch in range(params.epochs):
        _train_batch(
            model,
            train_dataloader,
            optimizer,
            epoch,
            device,
            random_flip,
            plotter,
        )
        val_acc = _val_batch(model, val_dataloader, device, epoch, random_flip, plotter)

        if val_acc > params.best_val_acc:
            params.best_val_acc = val_acc
            params.best_val_acc_epoch = epoch
    # TODO have a second pass with and without flipped to have the two validation accuracies

    # save model and save model params
    torch.save(model, run_path / "model.pt")

    # Update parameters for the run
    save_params(run_path, params)


def _train_batch(model, dataloader, optimizer, epoch, device, random_flip, plotter):
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        if np.random.rand() < random_flip:
            images = flip()(images)
        model.train()
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        print(
            f"Train Epoch: {epoch} | Batch: {i}/{len(dataloader)} "
            f"| Loss: {loss.item():.4f}"
        )

        plotter.plot("loss", "train", "Training Loss", epoch, loss.item())


@torch.no_grad()
def _val_batch(model, dataloader, device, epoch, random_flip, plotter):
    val_loss = 0
    correct = 0
    random_flip = 1.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        if np.random.rand() < random_flip:
            images = flip()(images)
        model.eval()
        output = model(images)
        val_loss += F.nll_loss(output, labels, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    val_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {accuracy}")

    if epoch == 1 or epoch == 9:
        plotter.plot("val loss", "val", "Validation Loss", epoch, val_loss)
        plotter.plot("acc", "val", "Validation Accuracy", epoch, accuracy)

    return accuracy
