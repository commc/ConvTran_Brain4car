from torch.nn import Module
import torch
from torch.nn import CrossEntropyLoss, LogSoftmax, Softmax, NLLLoss

class Myloss(Module):
    def __init__(self):
        super(Myloss, self).__init__()
        self.loss_function = CrossEntropyLoss()
        self.exponential_loss_function = CrossEntropyLoss(reduction='none')

    def forward(self, y_pre, y_true):
        batch, timestep = y_true.shape
        y_true = y_true.long()
        loss = self.loss_function(y_pre, y_true)
        
        # exponentially growing loss
        exponential_loss = self.exponential_loss_function(y_pre, y_true)
        weights = torch.exp(torch.arange(-timestep + 1, 1).float()).unsqueeze(0).expand_as(exponential_loss).to(exponential_loss.device)
        weighted_loss = exponential_loss * weights
        mean_exponential_loss = weighted_loss.sum(dim=1).mean()
        
        return loss