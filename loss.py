import torch
import math

class CasamentoDistriLoss(torch.nn.Module):
    def __init__(self):
        pass

def loss_new(d, y, a=0.001, sig=2.0):
    sqrt_pi = math.sqrt(2. * math.pi)
    d = d.unsqueeze(0)
    y = y.unsqueeze(0)
    d_int = (d.t().repeat(1, d.shape[1]) - y.repeat(y.shape[1], 1)) / sig
    print(d_int)
    gaussian = torch.exp(-1/2. * d_int * d_int) * 1/(sig * sqrt_pi)
    print(gaussian)
    loss = torch.sum(gaussian) / d.shape[1]
    return loss


cd = CasamentoDistriLoss()

d = torch.Tensor([1.,2.,3.])
y = torch.Tensor([2.,3.,4.])
loss = loss_new(d, y)
print(loss)
#loss.backward()
