import torch
from sklearn.metrics import accuracy_score


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    total_acc = 0
    model.eval()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)

            scores = model(inputs)
            _, predictions = scores.max(1)
            total_acc += accuracy_score(targets.cpu(), predictions.cpu())
            num_samples += 1

        return f"Got {num_correct} / {num_samples} with accuracy {total_acc/float(num_samples)}"


def check_metric(loader, model, metric, average, score, device):
    total_metric = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            scores = model(inputs)
            _, predictions = scores.max(1)
            num_samples += 1
            total_metric += metric(targets.cpu(), predictions.cpu(), average=average)

        return f"Got {score} Score {total_metric/float(num_samples)}"
