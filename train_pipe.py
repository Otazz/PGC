import numpy as np
import matplotlib.pyplot as plt
import torch

class TrainPipeline(object):
    def __init__(self, model, loss, opt, n_epochs, loss_name, batch_size):
        self.model = model
        self.loss_fn = loss
        self.opt = opt
        self.n_epochs = n_epochs
        self.hist = np.zeros(n_epochs)
        self.name = loss_name + ", n=" + str(n_epochs) + ", batch=" + str(batch_size)
        self.batch = batch_size

    def train(self, x_data, s_data, print_k):
        self.x_data = x_data
        self.s_data = s_data


        divider = int(len(s_data) / self.batch)

        for t in range(self.n_epochs):
            i = t % divider
            x_data = self.x_data[:, i*self.batch : (i+1)*self.batch]
            s_data = self.s_data[i*self.batch : (i+1)*self.batch]

            self.opt.zero_grad()

            self.y_pred = self.model(x_data)

            zero = torch.Tensor([0.] * len(s_data)).view(-1)

            loss = self.loss_fn(self.y_pred - s_data, zero)

            if print_k:
                print("Epoch ", t, "Error: ", loss)
            else:
                if t % (self.n_epochs/10) == 0:
                    print("Epoch ", t, "Error: ", loss)

            self.hist[t] = loss.item()


            loss.backward()
            self.opt.step()

        self.y_pred = self.run()

    def test(self, X_test):
        self.model.batch_size = self.X_test.shape[1]
        return self.model(X_test)

    def run(self):
        self.model.batch_size = self.x_data.shape[1]
        return self.model(self.x_data)

    def plot_train(self, file="train.png"):
        plt.figure()
        plt.plot(self.hist/np.linalg.norm(self.hist), label=self.name)
        plt.legend()
        plt.savefig(file)
        plt.show()

    def plot_results(self, mode="hist", file="result.png", y_test=None):
        plt.figure()

        if y_test:
            data = y_test
        else:
            data = self.y_pred


        if mode == 'dist':
            plt.plot(data.detach().numpy(), label="Preds")
            plt.plot(self.s_data.detach().numpy(), label="Data")

        elif mode == 'hist':
            plt.hist([self.s_data.detach().numpy(), data.detach().numpy()], bins=30, label=['d', 'y'])

        else:
            plt.plot(data.detach().numpy(), label="Preds")
            plt.plot(self.s_data.detach().numpy(), label="Data")
            plt.legend()
            plt.savefig('2'+file)
            plt.show()
            plt.hist([self.s_data.detach().numpy(), data.detach().numpy()], bins=30, label=['d', 'y'])

        plt.legend()
        plt.savefig(file)
        plt.show()
        plt.clf()
