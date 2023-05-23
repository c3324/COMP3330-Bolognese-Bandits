import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from torch import nn
import torch.optim as opt
from Q2 import Hidden32
from Q2 import ExtraLSTMLayers
from Q2 import Hidden32WithExtraEmbedding
from Q2 import logSoftMax
from Q2 import Hidden8_16_8
from Tweet import Tweets
import matplotlib.pyplot as plt

FIGPATH = 'Q2/figs/'


def progressBar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function

    def printProgressBar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                         (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


# Check which device is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))


vocab = {}
vocab['<PAD>'] = len(vocab)
trainDataSet, validationDataSet = random_split(Tweets(
    "Q2/dataset/split_coling2022_temporal/train_2020.single.json", vocab), [0.7, 0.3])
testDataSet = Tweets(
    "Q2/dataset/split_coling2022_temporal/train_2020.single.json", vocab)


def mainModel(model,modelName, lossFn, Opts, optNames, epochs):
    bestLoss = -1
    bestacc = 0
    bestname = ''
    for opt, name in zip(Opts, optNames):
        trainAccs = []
        trainLosses = []
        validAccs = []
        validLosses = []
        for epoch in progressBar(range(epochs), prefix=modelName + name, suffix='progress', length=50):
            trainLoss, trainAcc = train(
                trainDataLoader, opt, lossFn, model)
            validLoss, validAcc = test(validationDataLoader, lossFn, model)
            trainAccs.append(trainAcc.detach().cpu().numpy())
            trainLosses.append(trainLoss.detach().cpu().numpy())
            validAccs.append(validAcc.detach().cpu().numpy())
            validLosses.append(validLoss.detach().cpu().numpy())

        pd.options.display.float_format = '.2f'.format
        df = pd.DataFrame({
            'train accuracy': trainAccs,
            'train loss': trainLosses,
            'validation accuracy': validAccs,
            'validation loss': validLosses
        }, index=range(epochs))
        plotAccuracy(df, epochs, modelName + name)
        plotLoss(df, epochs, modelName + name)
        testloss, testacc = test(testDataLoader, lossFn, model)
        if bestLoss < 0:
            bestLoss = testloss
            bestacc = testacc
            bestname = modelName + name
        elif bestLoss > testloss and bestacc < testacc:
            bestLoss = testloss
            bestacc = testacc
            bestname = modelName + name
            
                
    return bestLoss, bestacc, bestname


def plotAccuracy(df, epochs, name):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), df['train accuracy'], df['validation accuracy'])
    plt.title('model train and validation accracy over time')
    plt.xlabel('epoch')
    plt.ylabel('accruacy')
    plt.grid(True)
    plt.legend(['train', 'validation'])
    plt.savefig(FIGPATH+name+'-acc.svg', format='svg')
    plt.close()


def plotLoss(df, epochs, name):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), df['train loss'], df['validation loss'])
    plt.title('model train and validation loss over time')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend(['train', 'validation'])
    plt.savefig(FIGPATH+name+'-loss.svg', format='svg')
    plt.close()


def train(trainData, opt, lossfn, model):
    epochloss = 0
    correct = 0
    for batch in trainData:
        opt.zero_grad()

        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        scores = model(inputs)
        # print(scores)
        loss = lossfn(scores, labels)

        loss.backward()
        opt.step()

        # scores = scores.cpu().detach().numpy()
        # labels = labels.cpu().detach().numpy()
        # print(labels.argmax(dim=1))

        # print("===")
        # print(labels.shape)
        correct += ((scores.argmax(dim=1) ==
                    labels.argmax(dim=1)).float().mean())
        # correct += (np.argmax(scores, axis=1) == np.argmax(labels, axis=1))
        epochloss += loss

    return epochloss/len(trainData), correct / len(trainData)


def test(data, lossfn, model):
    epochloss = 0
    correct = 0
    with torch.no_grad():
        for batch in data:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            scores = model(inputs)

            loss = lossfn(scores, labels)

            correct += ((scores.argmax(dim=1) ==
                        labels.argmax(dim=1)).float().mean())
            epochloss += loss

    return epochloss/len(data), correct / len(data)


# trainDataLoader = DataLoader(trainDataSet, batch_size=5, shuffle=True, collate_fn=collate)
# validationDataLoader = DataLoader(validationDataSet, batch_size=5, shuffle=True, collate_fn=collate)
# testDataLoader = DataLoader(testDataSet, batch_size=5, shuffle=True, collate_fn=collate)
BATCH_SIZE = 64
trainDataLoader = DataLoader(trainDataSet, batch_size=BATCH_SIZE, shuffle=True)
validationDataLoader = DataLoader(
    validationDataSet, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testDataSet, batch_size=BATCH_SIZE)

hidden32 = Hidden32(len(vocab), 6).to(device)
hidden8_16_8 = Hidden8_16_8(len(vocab), 6).to(device)
hidden32relu = Hidden32WithExtraEmbedding(len(vocab), 6).to(device)
extra1Layers = ExtraLSTMLayers(len(vocab), 6, 2).to(device)
extra127Layers = ExtraLSTMLayers(len(vocab), 6, 64).to(device)
lsm = logSoftMax(len(vocab), 6).to(device)

mainLoss = nn.BCELoss()
lsmLoss = nn.NLLLoss()

modelLossPairs = [(hidden32, 'hidden32', mainLoss), (hidden8_16_8,'hidden 8 16 8', mainLoss), (hidden32relu,'hidden w relu',
                                                                   mainLoss), (extra1Layers,'1 extra lstm layer', mainLoss), (extra127Layers,'127 extra lstm layer', mainLoss)]

finalAccs = {}
finalLosses = {}
for model, modelName, lossfn in modelLossPairs:
    adam05 = opt.Adam(model.parameters(), lr=0.2)
    adam1 = opt.Adam(model.parameters(), lr=0.1)
    adam2 = opt.Adam(model.parameters(), lr=0.01)
    adam3 = opt.Adam(model.parameters(), lr=0.001)
    SGD05 = opt.SGD(model.parameters(), lr=0.2)
    SGD1 = opt.SGD(model.parameters(), lr=0.1)
    SGD2 = opt.SGD(model.parameters(), lr=0.01)
    SGD3 = opt.SGD(model.parameters(), lr=0.001)
    opts = [adam05,adam1, adam2, adam3, SGD05, SGD1, SGD2, SGD3]
    optNames = ['adam-0.2','adam-0.1', 'adam-0.01', 'adam-0.001', 'SGD-0.2', 'SGD-0.1', 'SGD-0.01', 'SGD-0.001']
    loss, acc, name = mainModel(model, modelName, lossfn, opts, optNames, 200)
    finalAccs[name] = acc.detach().cpu().numpy()
    finalLosses[name] = loss.detach().cpu().numpy()
    
accCourses = list(finalAccs.keys())
accValues = list(finalAccs.values())
lossCourses = list(finalLosses.keys())
lossValues = list(finalLosses.values())
fig = plt.figure(figsize=(10, 5))
plt.bar(accCourses, accValues, width=0.1)
plt.xlabel('models')
plt.ylabel('test accuracy')
plt.title('test accuracy of each tested model')
plt.savefig(FIGPATH+'finalAccuracy'+'.svg', format='svg')
plt.bar(lossCourses, lossValues, width=0.1)
plt.xlabel('models')
plt.ylabel('test loss')
plt.title('test loss of each tested model')
plt.savefig(FIGPATH+'finalLoss'+'.svg', format='svg')


# loss_function = nn.CrossEntropyLoss()
# optimizer = opt.Adam(model.parameters(), lr=0.01)
