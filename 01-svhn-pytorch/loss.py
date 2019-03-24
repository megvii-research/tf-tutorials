import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxLoss(nn.Module):
    max_type_list = ['softmax', 'abs-max', 'square-max', 'plus-one-abs-max', 'non-negative-max']
    def __init__(self, max_type, size_average=True):
        super(MaxLoss, self).__init__()
        assert max_type in self. max_type_list
        self.max_type = max_type
        self.NLLLoss = nn.NLLLoss(size_average=size_average)

    def forward(self, inputs, target):
        if self.max_type == 'softmax':
            pred = F.softmax(inputs)
        elif self.max_type == 'abs-max':
            abs_val = torch.abs(inputs)
            pred = abs_val / torch.sum(abs_val, dim=1, keepdim=True)
        elif self.max_type == 'square-max':
            square_val = inputs * inputs
            pred = square_val / torch.sum(square_val, dim=1, keepdim=True)
        elif self.max_type == 'plus-one-abs-max':
            abs_val = torch.abs(inputs) + 1.0
            pred = abs_val / torch.sum(abs_val, dim=1, keepdim=True)
        elif self.max_type == 'non-negative-max':
            clamp_val = inputs.clamp(0)
            pred = clamp_val / torch.sum(clamp_val, dim=1, keepdim=True)

        return self.NLLLoss(pred, target)


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


def test_LpNorm():
    import model
    net = model.Model()
    reg = LpNorm()
    output = reg(net)
    print(output)


if __name__ == "__main__":
    test_LpNorm()
