import json
from nltk_utils import tokenize, stem, bagOfWords
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

allWords = []
tags = []
xy = []
ignoreWords = ['?', "!", '.', ',']

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        word = tokenize(pattern)
        allWords.extend(word)

        xy.append((word, tag))

# stem the words in the allWords list while disregarding the ignoreWords list
allWords = [stem(word) for word in allWords if word not in ignoreWords]

# remove duplicates and sort the words
allWords = sorted(set(allWords))
tags = sorted(set(tags))

XTrain = []
YTrain = []

for (patternSentence, tag) in xy:
    bag = bagOfWords(patternSentence, allWords)

    XTrain.append(bag)

    label = tags.index(tag)
    # CrossEntropyLoss
    YTrain.append(label)

XTrain = np.array(XTrain)
YTrain = np.array(YTrain)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(XTrain)
        self.x_data = XTrain
        self.y_data = YTrain

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


batchSize = 8
learningRate = 0.001
numberOfEpochs = 2000

inputSize = len(XTrain[0])
hiddenSize = 8
outputSize = len(tags)

dataset = ChatDataset()
trainLoader = DataLoader(dataset=dataset,
                         batch_size=batchSize,
                         shuffle=True,
                         num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

for epoch in range(numberOfEpochs):
    for (words, labels) in trainLoader:
        words = words.to(device)
        labels = labels.to(device)
        lables = labels.to(dtype=torch.long)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{numberOfEpochs}, loss = {loss.item():.4f}')

print(f'Final loss = {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": inputSize,
    "output_size": outputSize,
    "hidden_size": hiddenSize,
    "all_words": allWords,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete, file saved to {FILE}')
