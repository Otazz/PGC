import torch

class CasamentoDistriLoss(torch.nn.Module):
    def __init__(self):
        pass

def loss(d, y):
    #d_int = d.t().repeat(1, d.shape[1]) - y.repeat(y.shape[1], 1)
    gaussian = torch.exp(-1/2. * d.t().repeat(1, d.shape[1]) - y.repeat(y.shape[1], 1)) * 0.5/2.

    return gaussian


cd = CasamentoDistriLoss()

d

d = torch.Tensor([[1.,2.,3.]], requires_grad=True)
y = torch.Tensor([[2.,3.,4.]], requires_grad=True)
loss = loss(d, y)
loss.backward()
