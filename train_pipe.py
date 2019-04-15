class TrainPipeline(object):
    def __init__(self, model, loss, opt, n_epochs, loss_name):
        self.model = model
        self.loss_fn = loss
        self.opt = opt
        self.n_epochs = n_epochs
        self.name = loss_name + ", n=" + str(n_epochs)

    def train(self, x_data, s_data, print):
        self.hist = np.zeros(num_epochs)

        for t in range(num_epochs):

            self.model.zero_grad()

            self.model.hidden = self.model.init_hidden()

            self.y_pred = self.model(x_data)

            loss = self.loss_fn(y_pred, s_data)

            if print:
                print("Epoch ", t, "Error: ", loss)
            else:
                if t % (num_epochs/10) == 0:
                    print("Epoch ", t, "Error: ", loss)

            self.hist[t] = loss.item()

            self.opt.zero_grad()

            loss.backward()

            self.opt.step()
