import numpy as np
import matplotlib.pyplot as plt

class TrainPipeline(object):
    def __init__(self, model, loss, opt, n_epochs, loss_name):
        self.model = model
        self.loss_fn = loss
        self.opt = opt
        self.n_epochs = n_epochs
        self.hist = np.zeros(n_epochs)
        self.name = loss_name + ", n=" + str(n_epochs)

    def train(self, x_data, s_data, print_k):
        self.x_data = x_data
        self.s_data = s_data

        for t in range(self.n_epochs):

            self.model.zero_grad()

            self.model.hidden = self.model.init_hidden()

            self.y_pred = self.model(self.x_data)

            loss = self.loss_fn(self.y_pred, self.s_data)

            if print_k:
                print("Epoch ", t, "Error: ", loss)
            else:
                if t % (num_epochs/10) == 0:
                    print("Epoch ", t, "Error: ", loss)

            self.hist[t] = loss.item()

            self.opt.zero_grad()

            loss.backward()

            self.opt.step()

    def test(self, X_test):
        return self.model(X_test)

    def plot_train(self, file="train.png"):
        plt.figure()
        plt.plot(self.hist/np.linalg.norm(self.hist), label=self.name)
        plt.legend()
        plt.savefig(file)
        plt.show()

    def plot_results(self, mode="hist", file="result.png", y_test=None):
        plt.figure()

        if X_test:
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
            plt.savefig('dist.png')
            plt.show()
            plt.hist([self.s_data.detach().numpy(), data.detach().numpy()], bins=30, label=['d', 'y'])

        plt.legend()
        plt.savefig(file)
        plt.show()
        plt.clf()
