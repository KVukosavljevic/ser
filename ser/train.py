from torch import optim
import torch
import torch.nn.functional as F
import numpy as np

from ser.model import Net
from ser.transforms import flip


def train(run_path, params, train_dataloader, val_dataloader, device, random_flip):
    # setup model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # train
    for epoch in range(params.epochs):
        _train_batch(model, train_dataloader, optimizer, epoch, device, random_flip)
        _val_batch(model, val_dataloader, device, epoch, random_flip)

    # TODO have a second pass with and without flipped to have the two validation accuracies

    # save model and save model params
    torch.save(model, run_path / "model.pt")


def _train_batch(model, dataloader, optimizer, epoch, device, random_flip):
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


@torch.no_grad()
def _val_batch(model, dataloader, device, epoch):
    val_loss = 0
    correct = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        model.eval()
        output = model(images)
        val_loss += F.nll_loss(output, labels, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    val_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {accuracy}")
