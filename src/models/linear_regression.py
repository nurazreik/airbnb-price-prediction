import torch

class LinearRegression(torch.nn.Module):
    """
    Linear regression model inherits the torch.nn.Module
    which is the base class for all PyTorch modules.
    """
    def __init__(self, input_dim, output_dim):
        """ Initializes internal Module state. """
        super().__init__()
        self.linear_layer = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """ Defines the computation performed at every call. """
        x = x.reshape(-1, self.linear_layer.in_features)
        return self.linear_layer(x)