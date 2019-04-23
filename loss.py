import torch
import math
from test_cov import cov

from scipy.linalg import toeplitz

class CasamentoLoss(torch.nn.Module):
    def __init__(self, sig=1.):
        super(CasamentoLoss, self).__init__()
        self.sig = sig
        self.sqrt_pi = math.sqrt(2. * math.pi)

    def forward(self, d, y):
        d = d.unsqueeze(0)
        y = y.unsqueeze(0)

        combined = -2 * self.get_component(d, y)

        same_y = self.get_component(y, y)

        same_d = self.get_component(d, d)

        return same_y + same_d + combined

    def get_component(self, y1, y2):
        d_int = (y1.t().repeat(1, y2.shape[1]) - y2.repeat(y1.shape[1], 1)) / self.sig
        gaussian = torch.exp(-1/2. * d_int * d_int) * 1/(self.sig * self.sqrt_pi)
        return torch.sum(gaussian) / (y1.shape[1] * y2.shape[1])

class MSEControl(torch.nn.Module):
    def __init__(self):
        super(MSEControl, self).__init__()

    def forward(self, d, y):
        return torch.sum((d - y) * (d - y))/d.shape[0]


class CasamentoMult(torch.nn.Module):
    def __init__(self, sig=.5):
        super(CasamentoMult, self).__init__()
        self.sig = sig
        self.sqrt_pi = math.sqrt(2. * math.pi)

    def forward(self, d, y):
        d = torch.tensor(toeplitz(d))
        y = torch.tensor(toeplitz(y))

        d = d.unsqueeze(0)
        y = y.unsqueeze(0)

        combined = -2 * self.get_component(d, y)

        same_y = self.get_component(y, y)

        same_d = self.get_component(d, d)

        return same_y + same_d + combined


    def get_component(self, y1, y2):
        self.sqrt_pi = math.sqrt(((2. * math.pi) ** y1.shape[1]) * (self.sig ** (2 * y1.shape[1])))
        d_int = (torch.transpose(y1, 0,1).repeat(1, y2.shape[1], 1) - y2.repeat(y1.shape[1], 1, 1))
        d_int_transposed = torch.transpose(d_int, 0, 1)
        mid = torch.matmul(torch.matmul(d_int_transposed, (torch.eye(d_int.shape[1]) * (self.sig ** 2)).inverse()),d_int)
        gaussian = torch.exp(-1/2. * mid) / self.sqrt_pi
        return torch.sum(gaussian) / (y1.shape[1] * y2.shape[1])

cd = CasamentoMult()

d = torch.Tensor([2.,4.,5.])
y = torch.Tensor([2.,4.,5.])
loss = cd(d, y)
print(loss)
#loss.backward(7
