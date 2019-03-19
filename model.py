import torch
import torch.nn as nn
from generate_data import *
import math
import matplotlib.pyplot as plt


# Data params
noise_var = 0
num_datapoints = 100
test_size = 0.2
num_train = int((1-test_size) * num_datapoints)


input_size = 20

per_element = True
if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size

h1 = 32
output_dim = 1
num_layers = 2
learning_rate = 1e-3
num_epochs = 1
dtype = torch.float

#####################
# Generate data
#####################
data = ARData(num_datapoints, num_prev=input_size, test_size=test_size, noise_var=noise_var, coeffs=fixed_ar_coefficients[input_size])

X_train = torch.from_numpy(data.X_train).type(torch.Tensor)
X_test = torch.from_numpy(data.X_test).type(torch.Tensor)
y_train = torch.from_numpy(data.y_train).type(torch.Tensor).view(-1)
y_test = torch.from_numpy(data.y_test).type(torch.Tensor).view(-1)

X_train = X_train.view([input_size, -1, 1])
X_test = X_test.view([input_size, -1, 1])


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
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)



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


loss_fn = loss_new
#loss_fn = torch.nn.MSELoss(size_average=False)
loss_mse = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    model.hidden = model.init_hidden()

    y_pred = model(X_train)


    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "Error: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()
print("pred", y_pred)
print("train", y_train)

print("\n\nFinal loss: ", loss_mse(y_pred, y_train))

plt.plot(y_pred.detach().numpy(), label="Preds")
plt.plot(y_train.detach().numpy(), label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()
