import torch
import torch.nn as nn
from generate_data import *
import math
from train_pipe import TrainPipeline
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.signal import lfilter
from get_data import get_data

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000,
                    help='Number of epocs')
parser.add_argument('--loss', type=str, default='new',
                    help='Loss used on the model:\
                        \n- new for Casamento\
                        \n- mse for MSE')
parser.add_argument('-s', type=float, default=0.5,
                    help='Sigma used on the Casamento loss')
parser.add_argument('--batch_size', type=int, default=100,
                    help='Batch size used for training')
parser.add_argument('-f', type=str, default='',
                    help='Path for the train file, \
                    if empty it uses the default one')
parser.add_argument('-t', type=str, default='',
                    help='Path for the test file, if empty it skips test')
parser.add_argument('-v', type=str, default='',
                    help='Path for the validation file, \
                    if empty it skips validation')
parser.add_argument('-n', type=int, default=1000,
                    help='Number of examples used on the default train values')
parser.add_argument('--print', type=bool, default=False,
                    help='Print the loss on each epoch, \
                    if False it only prints \
                    every 1/10 of the number of epochs')
parser.add_argument('--out', type=str, default='dist',
                    help='The image output of the run:\
                    \n- hist for histogram\
                    \n- dist for distribution\
                    \n- both for both')
parser.add_argument('--ep', type=bool, default=True,
                    help='Plot or not the loss comparisson by epoch')

FLAGS, unparsed = parser.parse_known_args()

h1 = 64
output_dim = 1
num_layers = 1
learning_rate = 1e-4
num_epochs = FLAGS.n_epochs
dtype = torch.float
batch_size = FLAGS.batch_size

if FLAGS.f:
    if FLAGS.f == "real":
        x_data, s_data = get_data()
    else:
        data = np.loadtxt(FLAGS.f)
        x_data = data[:, :-1]
        s_data = data[:, -1]

    input_size = x_data.shape[1]

else:
    # signals generation
    n = FLAGS.n
    # white noise generation
    s_data = np.random.uniform(-1, 1, n)
    # uniform white noise with 10000 samples between -0.9 and 0.9
    x_data = lfilter([1, 0.6, 0, 0, 0, 0, 0.2], 1, s_data)
    # addition of memory in the white noise
    batch_size = n
    input_size = 1

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
                    num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self, batch_size):
        return (torch.nn.init.xavier_uniform_(torch.rand(self.num_layers, self.batch_size, self.hidden_dim)),
                torch.nn.init.xavier_uniform_(torch.rand(self.num_layers, self.batch_size, self.hidden_dim)))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1), self.init_hidden(self.batch_size))
        intt = self.hidden[0].view(self.batch_size, self.hidden_dim)
        y_pred = self.linear(intt)
        return y_pred.view(-1)


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

class CasamentoMult(torch.nn.Module):
    def __init__(self, sig=.5):
        super(CasamentoMult, self).__init__()
        self.sig = sig

    def forward(self, d, y):
        d = self.toeplitz_like(d.unsqueeze(0), 50).t()
        y = self.toeplitz_like(y.unsqueeze(0), 50).t()

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

    def toeplitz_like(self, x, n):
        r = x
        stop = x.shape[0] - 1

        if n < stop:
            stop = n

        for i in range(stop):
            r = torch.cat((r, x.roll(i+1)), 0)

        return r

class MSEControl(torch.nn.Module):
    def __init__(self):
        super(MSEControl, self).__init__()

    def forward(self, d, y):
        return torch.sum((d - y) * (d - y))/d.shape[0]


loss_mse = torch.nn.MSELoss(size_average=False)

if FLAGS.ep:
    test_methods = ['new2', 'mse']
else:
    test_methods = [FLAGS.loss]


pipes = []

for met in test_methods:
    if met == 'new':
        method = "Casamento Univariado"
        loss_fn = CasamentoMult(FLAGS.s)
    elif met == 'new2':
        method = "Casamento Multivariado"
        loss_fn = CasamentoMult(FLAGS.s)
    else:
        method = "MSE normal"
        loss_fn = torch.nn.MSELoss()

    print("Usando", method)

    model = LSTM(lstm_input_size, h1, batch_size=batch_size, output_dim=output_dim, num_layers=num_layers)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    pipe = TrainPipeline(model, loss_fn, optimizer, num_epochs, method, batch_size)
    pipe.train(x_data, s_data, FLAGS.print)

    pipes.append(pipe)

y_pred = pipes[0].run()


if FLAGS.ep:
    fig, ax1 = plt.subplots()
    ax1.plot(pipes[0].hist, label=pipes[0].name, color='tab:red')
    plt.legend()

    ax2 = ax1.twinx()
    ax2.plot(pipes[1].hist, label=pipes[1].name, color='tab:blue')

    plt.legend()
    plt.savefig("train.png")
    plt.show()


if FLAGS.t:
    test = np.loadtxt(FLAGS.t)
    X_test = test[:, : -1]
    s_test = test[:, -1]

    X_test, s_test = reshape_data(X_test, s_test, y_test.shape[1])

    y_test = pipes[0].model(X_test)

    pipes[0].plot_results(FLAGS.out, y_test=y_test)


if FLAGS.v:
    pass
    #TO-DO


pipes[0].plot_results(FLAGS.out)
pipes[1].plot_results(FLAGS.out, file="j.png")
