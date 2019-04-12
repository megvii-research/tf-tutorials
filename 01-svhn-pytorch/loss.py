import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxLoss(nn.Module):
    max_type_list = ['softmax', 'abs-max', 'square-max', 'plus-one-abs-max', 'non-negative-max']

    def __init__(self, max_type):
        super(MaxLoss, self).__init__()
        assert max_type in self. max_type_list
        self.max_type = max_type
        self.NLLLoss = nn.NLLLoss()

    def _onehot(self, x, dim_num):
        one_hot_code = torch.zeros(x.size()[0], dim_num)
        one_hot_code.scatter_(1, x, 1)
        if torch.cuda.is_available():
            one_hot_code = one_hot_code.cuda()
        return one_hot_code.long()

    def forward(self, inputs, target):
        if self.max_type == 'softmax':
            pred = F.softmax(inputs, dim=1)
            pred = torch.log(pred)
        elif self.max_type == 'abs-max':
            raise NotImplementedError
        elif self.max_type == 'square-max':
            raise NotImplementedError
        elif self.max_type == 'plus-one-abs-max':
            raise NotImplementedError
        elif self.max_type == 'non-negative-max':
            raise NotImplementedError

        return self.NLLLoss(pred, target)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def _onehot(self, x, dim_num):
        one_hot_code = torch.zeros(x.size()[0], dim_num)
        one_hot_code.scatter_(1, x, 1)
        if torch.cuda.is_available():
            one_hot_code = one_hot_code.cuda()
        return one_hot_code.float()

    def forward(self, inputs, target):
        raise NotImplementedError


class LpNorm(nn.Module):
    def __init__(self, p=2, factor=1e-5):
        super(LpNorm, self).__init__()
        self.p = p
        self.factor = factor

    def forward(self, net):
        total_norm = []
        for item in net.parameters():
            total_norm.append(torch.norm(item.data, p=self.p))
        reg = torch.sum(torch.stack(total_norm, dim=0), dim=0)
        return self.factor * reg
