from torch.nn import Module
import torch
from torch.nn import CrossEntropyLoss, LogSoftmax, Softmax, NLLLoss

class Myloss(Module):
    def __init__(self):
        super(Myloss, self).__init__()
        self.CELoss = CrossEntropyLoss(reduction='none')
        self.logsoftmax = LogSoftmax(dim=1)
        self.softmax = Softmax(dim=1)
    
    @staticmethod
    def ContextRule(y_true, predictions, lane_info):
        assert lane_info.shape[-1] == 3
        left_lane = lane_info[:, 0].unsqueeze(dim=1).expand_as(predictions)
        right_lane = lane_info[:, 1].unsqueeze(dim=1).expand_as(predictions)
        intersection = lane_info[:, 2].unsqueeze(dim=1).expand_as(predictions)
        # rules
        rule_left_change = (predictions == 2) & (left_lane == 0)
        rule_right_change = (predictions == 4) & (right_lane == 0)
        rule_left_turn1 = (predictions == 3) & (left_lane == 1) & (right_lane == 0)
        rule_left_turn2 = (predictions == 3) & (intersection == 0)
        rule_right_turn1 = (predictions == 5) & (left_lane == 0) & (right_lane == 1)
        rule_right_turn2 = (predictions == 5) & (intersection == 0)
        violations = rule_left_change | rule_right_change | rule_left_turn1 | rule_left_turn2 | rule_right_turn1 | rule_right_turn2
        
        return violations
    
    # TODO: rule-based driving context loss
    def RBLoss(self, softmax_out, y_true, lane_info):
        predictions_prob, predictions = torch.max(softmax_out, dim=1)
        rbloss = -torch.log(1.0 - predictions_prob)
        # rbloss = -torch.log(predictions_prob)
        violations = self.ContextRule(y_true, predictions, lane_info)
        rbloss = rbloss * violations
        
        return rbloss
    
    def forward(self, y_pre, y_true, lane_info, loss_type='celoss', exponential=False):
        batch, timestep = y_true.shape
        y_true = y_true.long()
        celoss = self.CELoss(y_pre, y_true)
        softmax_out = self.softmax(y_pre)
        rbloss = self.RBLoss(softmax_out, y_true, lane_info)
        # loss type
        if loss_type == 'celoss':
            loss = celoss
        elif loss_type == 'cerbloss':
            loss = celoss + rbloss
        # exponentially growing loss    
        if exponential:
            weights = torch.exp(torch.arange(-timestep + 1, 1).float()).unsqueeze(0).expand_as(loss).to(loss.device)
            loss = loss * weights
            loss = loss.sum(dim=1).mean()
        else:
            loss = loss.mean()
        
        return loss