import torch.nn.functional as F

def train(model, training_dataloader, optimizer, device):

    for i, (images, labels) in enumerate(training_dataloader):
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        
    return loss