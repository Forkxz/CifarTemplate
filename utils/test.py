import torch
import torch.nn as nn
from utils.util import AverageMeter

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True).cpu()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def test(model,testloader,device):
    model.eval()
    losses = AverageMeter('Loss',':4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    criterion = nn.CrossEntropyLoss()
    for data, target in testloader:
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            output = model(data)
            loss = criterion(output, target)

            losses.update(loss.item(), data.size(0))
            acc1 = accuracy(output, target)
            top1.update(acc1[0].numpy()[0], data.size(0))

    return losses.avg, top1.avg
