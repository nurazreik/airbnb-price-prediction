import torch

class PolynomialRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        """ Initializes internal Module state. """
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.linear_layer = torch.nn.Linear(input_dim * degree, output_dim)

    def forward(self, x):
        """ Defines the computation performed at every call. """
        x = x.reshape(-1, self.input_dim)

        x_poly = [x]
        for i in range(2, self.degree+1):
            x_poly.append(x ** i)

        x_poly = torch.cat(x_poly, dim=1)

        return self.linear_layer(x_poly)