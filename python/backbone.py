import torch.nn as nn

class BackboneMLP(nn.Module):

    def __init__(
        self,
        hidden1=256,
        hidden2=128,
        num_classes=8
    ):
        super().__init__()

        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)

        self.relu = nn.ReLU()

        self.classifier = nn.Linear(
            hidden2,
            num_classes
        )

    def forward_features(self, x):

        x = x.view(x.size(0), -1)

        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))

        return h2

    def forward(self, x):

        h = self.forward_features(x)

        return self.classifier(h)

