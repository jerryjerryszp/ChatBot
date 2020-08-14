# import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    # feed forward neural net with two hidden layers
    def __init__(self, inputSize, hiddenSize, numClasses):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(inputSize, hiddenSize)
        self.layer2 = nn.Linear(hiddenSize, hiddenSize)
        self.layer3 = nn.Linear(hiddenSize, numClasses)

        # Rectified Linear Unit
        # f(x) = max(0, x)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        # no activation and no softmax
        # softmax will be applied later
        return out
