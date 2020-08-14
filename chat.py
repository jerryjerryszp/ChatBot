import random
import json
import torch

from model import NeuralNet
from nltk_utils import bagOfWords, tokenize

# check if the gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

inputSize = data["input_size"]
hiddenSize = data["hidden_size"]
outputSize = data["output_size"]
allWords = data["all_words"]
tags = data["tags"]
modelState = data["model_state"]


model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)
model.load_state_dict(modelState)
model.eval()

# create the chat
botName = "Yuuurrr"
print("Yo wassup! type 'quit' to exit")

while True:
    sentence = input('you: ')
    if sentence == "quit":
        break

    # first tokenize the sentence and then calculate the bag of words
    sentence = tokenize(sentence)
    X = bagOfWords(sentence, allWords)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Apply softmax so that each component will be in the interval of [0,1]
    # and they all sum up to 1 so we get the interpreted probability
    # This is to map the non-normalized output of a network
    # to a probability distribution over predicted output classes
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
    else:
        print(f'{botName}: I don\'t know what you be saying bruv')
