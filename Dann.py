import torch.nn as nn
from GRL import GRL

class Dann(nn.Module):
    def __init__(self):
        super(Dann, self).__init__()
        self.height = 5
        # f
        self.f = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=5),
                nn.BatchNorm2d(64),
#                 nn.Dropout2d(),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(64, 50, kernel_size=5),
                nn.BatchNorm2d(50),
                nn.Dropout2d(),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(50, 50, kernel_size=3),
                nn.BatchNorm2d(50),
                nn.Dropout2d(),
                nn.MaxPool2d(2),
                nn.ReLU(),
            )
#        self.lc = nn.Sequential(
#            nn.Linear(50*self.height*self.height, 100),
#            nn.BatchNorm1d(100),
#            nn.ReLU(),
#            nn.Dropout(),
#            nn.Linear(100, 100),
#            nn.BatchNorm1d(100),
#            nn.ReLU(),
#            nn.Dropout(),
#            nn.Linear(100, 4),
#            nn.Sigmoid(),
#        )
        # C
        self.sc = nn.Sequential(
            nn.Linear(50*self.height*self.height, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 13),
            nn.Sigmoid(),   # TG: Is this necessary? -> It does not matter.
        )
        # Dfeat?
        self.dc = nn.Sequential(
            nn.Linear(50*self.height*self.height, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 2),  # TG: Why 2? It should be 1?
            nn.Sigmoid(),   # TG: Is this necessary? -> It does not matter.
        )

    def forward(self, x, alpha):
        latent = self.f(x)
        latent = latent.view(-1, 50 * self.height * self.height)
        s = self.sc(latent)
        y = GRL.apply(latent, alpha)    # TG: What's this?
        d = self.dc(y)
        s = s.view(s.shape[0], -1)
        d = d.view(d.shape[0], -1)
        return d, s, latent
