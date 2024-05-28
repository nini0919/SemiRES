import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets,weight=None):
        loss = F.cross_entropy(inputs, targets, reduction='none')
        if weight is not None:
            loss = loss * weight
        loss = torch.mean(loss)
        return loss