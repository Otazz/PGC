import torch
import math

class CasamentoDistriLoss(torch.nn.Module):
    def __init__(self):
        pass

def loss_new(d, y, sig=1.0):
    sqrt_pi = math.sqrt(2. * math.pi)
    d = d.unsqueeze(0)
    y = y.unsqueeze(0)

    d_int = (d.t().repeat(1, d.shape[1]) - y.repeat(y.shape[1], 1)) / sig
    gaussian = torch.exp(-1/2. * d_int * d_int) * (1/(sig * sqrt_pi))
    combined = -2 * torch.sum(gaussian) / (d.shape[1] * y.shape[1])

    print("Combined: ", combined)

    y_int = (y.t().repeat(1, y.shape[1]) - y.repeat(y.shape[1], 1)) / sig
    gaussian = torch.exp(-1/2. * y_int * y_int) * 1/(sig * sqrt_pi)
    same_y = torch.sum(gaussian) / (y.shape[1] * y.shape[1])

    print("Same y: ", same_y)

    d_int = (d.t().repeat(1, d.shape[1]) - d.repeat(d.shape[1], 1)) / sig
    gaussian = torch.exp(-1/2. * d_int * d_int) * 1/(sig * sqrt_pi)
    same_d = torch.sum(gaussian) / (d.shape[1] * d.shape[1])

    print("Same d: ", same_d)

    print("Loss: ", same_y + same_d + combined)

    return same_y + same_d + combined

cd = CasamentoDistriLoss()

d = torch.Tensor([1.,2.])
y = torch.Tensor([2.,4.])
loss = loss_new(d, y)
print(loss)
#loss.backward()
