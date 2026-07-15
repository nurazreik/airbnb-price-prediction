import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, mean, std):

        self.X = torch.tensor(X.to_numpy())
        self.y = torch.tensor(y.to_numpy())

        self.mean = mean
        self.std = std


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        x = (x - self.mean) / (self.std + 1e-8)

        return x, y