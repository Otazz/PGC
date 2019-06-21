import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

from loss import CasamentoMult


k = 100
a = [0]
b = [0]
for n in range(1, k/2):
    x = torch.arange(k)
    cas = CasamentoMult()
    st = time.time()
    cas.toeplitz_like(x, n)
    a.append(time.time() - st)
    st = time.time()
    cas.toeplitz_like_old(x.unsqueeze(0), n).t()
    b.append(time.time() - st)

plt.plot(a, label='new')
plt.plot(b, label='old')
plt.legend()
plt.show()
