import torch


def confusion(preds, labels, num_classes):
    return torch.bincount(labels * num_classes + preds, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics(cm):
    cm = cm.double()
    tp = cm.diag()
    support = cm.sum(1)
    predicted = cm.sum(0)
    recall = tp / support.clamp(min=1)
    precision = tp / predicted.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    n = cm.sum()
    correct = tp.sum()
    mcc = (correct * n - (predicted * support).sum()) / (
        (n**2 - (predicted**2).sum()).sqrt() * (n**2 - (support**2).sum()).sqrt()).clamp(min=1e-12)
    return {
        "accuracy": (correct / n).item(),
        "balanced_accuracy": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "mcc": mcc.item(),
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "f1": f1.tolist(),
        "support": support.long().tolist(),
    }
