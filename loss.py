import torch
import math

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

    def forward(self, d, y):

        d = d.unsqueeze(0)
        y = y.unsqueeze(0)

        d = self.toeplitz_like(d, 5).t()
        y = self.toeplitz_like(y, 5).t()

        self.sqrt_pi = math.sqrt(((2 * math.pi) ** y.shape[1]) * (self.sig ** (2 * y.shape[1])))

        d = d.unsqueeze(0)
        y = y.unsqueeze(0)

        combined = -2 * self.get_component(d, y)

        same_y = self.get_component(y, y)

        same_d = self.get_component(d, d)

        return same_y + same_d + combined


    def get_component(self, y1, y2):
        d_int = (torch.transpose(y1, 0,1).repeat(1, y2.shape[1], 1) - y2.repeat(y1.shape[1], 1, 1))

        mid = (1 /self.sig ** 2) * d_int * d_int
        gaussian = torch.exp(-1/2. * mid) / self.sqrt_pi
        return torch.sum(gaussian) / (y1.shape[1] * y2.shape[1])

    def toeplitz(self, v):
        c = v.view(-1)
        vals = torch.cat((torch.flip(c, [0]), c[1:]))
        a = torch.arange(c.shape[0]).unsqueeze(0).t()
        b = torch.arange(c.shape[0] - 1, -1, step=-1).unsqueeze(0)
        indx = a + b

        return vals[indx]

    def toeplitz_like(self, x, n):
        r = x
        stop = x.shape[0] - 1

        if n < stop:
            stop = n

        else:
            stop = 2

        r = self.toeplitz(r)

        return r.narrow(1, 0, stop).narrow(0, stop - 1, r.shape[0] - stop)

    def toeplitz_like_old(self, x, n):
        r = x
        stop = x.shape[1] - 1

        if n < stop:
            stop = n

        else:
            stop = 2

        for i in range(stop):
            r = torch.cat((r, x.roll(i+1)), 0)

        return r.narrow(1, stop, x.shape[1] - stop)
