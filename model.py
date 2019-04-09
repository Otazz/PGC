import torch
import torch.nn as nn
from generate_data import *
import math

import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import argparse


import numpy as np
import random as rd
from scipy.signal import lfilter




parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epocs')
parser.add_argument('--loss', type=str, default='new', help='Loss used on the model:\n- new for Casamento\n- mse for MSE')
parser.add_argument('-s', type=float, default=0.5, help='Sigma used on the Casamento loss')
parser.add_argument('-f', type=str, default='', help='Path for the train file, if empty it uses the default one')
parser.add_argument('-t', type=str, default='', help='Path for the test file, if empty it uses the default one')
parser.add_argument('-v', type=str, default='', help='Path for the validation file, if empty it skips validation')
parser.add_argument('-n', type=int, default=1000, help='Number of examples used on the default train values')
parser.add_argument('--print', type=bool, default=False, help='Print the loss on each epoch, if False it only prints every 1/10 of the number of epochs')
parser.add_argument('--out', type=str, default='dist', help='The image output of the run:\n- hist for histogram\n-dist for distribution\n- both for both')

FLAGS, unparsed = parser.parse_known_args()

h1 = 32
output_dim = 1
num_layers = 2
learning_rate = 1e-3
num_epochs = FLAGS.n_epochs
dtype = torch.float

if FLAGS.f:
    data = np.loadtxt(FLAGS.f)
    x_data = data[:1000, : -1]
    s_data = data[:1000, -1]
    input_size = x_data.shape[1]
    num_train = x_data.shape[0]
else:
    #signals generation
    n = FLAGS.n
    #white noise generation
    s_data = np.random.uniform(-1, 1, n) #uniform white noise with 10000 samples between -0.9 and 0.9
    x_data = lfilter([1, 0.6, 0, 0, 0, 0, 0.2], 1, s_data) #addition of memory in the white noise
    num_train = n

def reshape_data(x, y, input_size):
    x = torch.from_numpy(x).type(torch.Tensor).view([input_size, -1, 1])
    y = torch.from_numpy(y).type(torch.Tensor).view(-1)

    return (x, y)

per_element = True
if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size

x_data, s_data = reshape_data(x_data, s_data, input_size)

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (torch.rand(self.num_layers, self.batch_size, self.hidden_dim),
                torch.rand(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

#model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)
model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)

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



if FLAGS.loss == 'new':
    loss_fn = CasamentoLoss(FLAGS.s)
    print("Usando Casamento")
elif FLAGS.loss == 'msec':
    loss_fn = MSEControl()
    print("Usando MSE Controle")
else:
    loss_fn = torch.nn.MSELoss(size_average=False)
    print("Usando MSE normal")

loss_mse = torch.nn.MSELoss(size_average=False)



optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


hist = np.zeros(num_epochs)
#same_y = CasamentoLoss().get_component(s_data.unsqueeze(0), s_data.unsqueeze(0))

for t in range(num_epochs):

    model.zero_grad()

    model.hidden = model.init_hidden()

    y_pred = model(x_data)

    loss = loss_fn(y_pred, s_data)
    #print("LOSS", loss)

    #if t % 400 == 0:
    if FLAGS.print:
        print("Epoch ", t, "Error: ", loss)
    else:
        if t % (num_epochs/10) == 0:
            print("Epoch ", t, "Error: ", loss)

    hist[t] = loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


print("pred: ", y_pred)
print("real: ", s_data)

print("\n\nFinal loss: ", loss_mse(y_pred, s_data))

if FLAGS.t:
    test = np.loadtxt(FLAGS.t)
    x_test = test[:, : -1]
    s_test = test[:, -1]

    x_test, s_test = reshape_data(x_test, s_test, x_test.shape[1])

    y_test = model(x_test)

    if FLAGS.out == 'dist':
        plt.plot(y_test.detach().numpy(), label="Preds")
        plt.plot(s_tests.detach().numpy(), label="Data")

    elif FLAGS.out == 'hist':
        plt.hist([s_test.detach().numpy(), y_test.detach().numpy()], bins=30, label=['d', 'y'])

    else:
        plt.plot(y_test.detach().numpy(), label="Preds")
        plt.plot(s_test.detach().numpy(), label="Data")
        plt.legend()
        plt.savefig('test_dist.png')
        plt.show()
        plt.hist([s_test.detach().numpy(), y_test.detach().numpy()], bins=30, label=['d', 'y'])

    plt.legend()
    plt.savefig('test_hist.png')
    plt.show()


if FLAGS.v:
    pass
    #TO-DO


if FLAGS.out == 'dist':
    plt.plot(y_pred.detach().numpy(), label="Preds")
    plt.plot(s_data.detach().numpy(), label="Data")

elif FLAGS.out == 'hist':
    plt.hist([s_data.detach().numpy(), y_pred.detach().numpy()], bins=30, label=['d', 'y'])

else:
    plt.plot(y_pred.detach().numpy(), label="Preds")
    plt.plot(s_data.detach().numpy(), label="Data")
    plt.legend()
    plt.savefig('dist.png')
    plt.show()
    plt.hist([s_data.detach().numpy(), y_pred.detach().numpy()], bins=30, label=['d', 'y'])

plt.legend()
plt.savefig('k.png')
plt.show()
