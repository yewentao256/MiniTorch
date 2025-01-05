import minitorch

# Use this function to make a random parameter in your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

class Linear(minitorch.Module):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        # Weights: shape (in_size, out_size)
        self.weights = RParam(in_size, out_size)
        # Biases: shape (out_size,)
        self.bias = RParam(out_size)

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """
        Forward pass for the Linear layer.

        Args:
            x (minitorch.Tensor): Input tensor of shape (batch_size, in_size)

        Returns:
            minitorch.Tensor: Output tensor of shape (batch_size, out_size)
        """
        batch_size = x.shape[0]
        in_size = x.shape[1]
        out_size = self.weights.value.shape[1]

        x_expanded = x.view(batch_size, in_size, 1)
        weights_expanded = self.weights.value.view(1, in_size, out_size)
        # (batch_size, in_size, 1) * (1, in_size, out_size) -> (batch_size, in_size, out_size)
        mul = x_expanded * weights_expanded

        # (batch_size, out_size)
        output = mul.sum(dim=1).view(batch_size, out_size)
        output = output + self.bias.value
        return output


class Network(minitorch.Module):
    def __init__(self, hidden_layers: int) -> None:
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """
        Forward pass for the Network.

        Args:
            x (minitorch.Tensor): Input tensor of shape (batch_size, 2)

        Returns:
            minitorch.Tensor: Output tensor of shape (batch_size, 1)
        """
        middle = self.layer1.forward(x).relu()
        end = self.layer2.forward(middle).relu()
        output = self.layer3.forward(end).sigmoid()
        return output


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                # Note: log_fn used is in `project\interface\train.py`
                # We update the code there instead of default_log_fn
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
