import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from torch import nn
import torch.optim as opt
import torchtext
from Transformer import Control
from Transformer import ExtraEmbedding
from Transformer import Dropout
from Transformer import ExtraHeads
from Transformer import ExtraEncodings
from Tweet import Tweets
import matplotlib.pyplot as plt
import datasets

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
                train_dataloader, opt, lossFn, model)
            validLoss, validAcc = test(valid_dataloader, lossFn, model)
            trainAccs.append(trainAcc)
            trainLosses.append(trainLoss)
            validAccs.append(validAcc)
            validLosses.append(validLoss)

        pd.options.display.float_format = '.2f'.format
        df = pd.DataFrame({
            'train accuracy': trainAccs,
            'train loss': trainLosses,
            'validation accuracy': validAccs,
            'validation loss': validLosses
        }, index=range(epochs))
        plotAccuracy(df, epochs, modelName + name)
        plotLoss(df, epochs, modelName + name)
        testloss, testacc = test(test_dataloader, lossFn, model)
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
    losses, accuracies = [], []
    for batch in trainData:
        opt.zero_grad()

        inputs = batch['ids'].to(device)
        labels = batch['label'].to(device)

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
        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.sum(torch.argmax(scores, dim=-1) == labels) / labels.shape[0]
        accuracies.append(accuracy.detach().cpu().numpy())

    return np.mean(losses), np.mean(accuracies)


def test(data, lossfn, model):
    losses, accuracies = [], []
    with torch.no_grad():
        for batch in data:
            inputs = batch['ids'].to(device)
            labels = batch['label'].to(device)

            scores = model(inputs)

            loss = lossfn(scores, labels)

            losses.append(loss.detach().cpu().numpy())
            accuracy = torch.sum(torch.argmax(scores, dim=-1) == labels) / labels.shape[0]
            accuracies.append(accuracy.detach().cpu().numpy())

    return np.mean(losses), np.mean(accuracies)

def tokenize(example, tokenizer, max_length):
    tokens = tokenizer(example['text'])[:max_length]
    return {'tokens': tokens}

def numericalize_data(example, vocab):
    ids = [vocab[token] for token in example['tokens']]
    return {'ids': ids}

def collate(batch):
    batch_ids = [i['ids'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_label = [i['label'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {'ids': batch_ids,
             'label': batch_label}
    return batch


# Check which device is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))
BATCH_SIZE = 128


train_data, test_data = datasets.load_dataset("cardiffnlp/tweet_topic_single", split=["train_coling2022", "test_coling2022"])
# Split off 5,000 examples from train_data for validation
train_valid_data = train_data.train_test_split(test_size=0.1)
train_data = train_valid_data['train']
valid_data = train_valid_data['test']


tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
max_length = 600


train_data = train_data.map(tokenize, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})
valid_data = valid_data.map(tokenize, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})
test_data = test_data.map(tokenize, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})

vocab = torchtext.vocab.build_vocab_from_iterator(train_data['tokens'],
                                                  min_freq=5,
                                                  specials=['<unk>', '<pad>'])

vocab.set_default_index(vocab['<unk>'])
pad_index = vocab['<pad>']

train_data = train_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
valid_data = valid_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
test_data = test_data.map(numericalize_data, fn_kwargs={'vocab': vocab})

train_data = train_data.with_format(type='torch', columns=['ids', 'label'])
valid_data = valid_data.with_format(type='torch', columns=['ids', 'label'])
test_data = test_data.with_format(type='torch', columns=['ids', 'label'])

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate)



mainLoss = nn.CrossEntropyLoss().to(device=device)

control = Control(vocab).to(device)
Embeddings = ExtraEmbedding(vocab).to(device)
drop = Dropout(vocab).to(device)
heads = ExtraHeads(vocab).to(device)
encodings = ExtraEncodings(vocab).to(device)


modelLossPairs = [(drop, 'with dropout ')]
EPOCHS = 500

finalAccs = {}
finalLosses = {}
for model, modelName in modelLossPairs:
    adam1 = opt.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    adam2 = opt.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    adam3 = opt.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    adam4 = opt.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
    adam5 = opt.AdamW(model.parameters(), lr=0.00001, weight_decay=0.1)
    adam6 = opt.AdamW(model.parameters(), lr=0.000001, weight_decay=0.1)
    adam7 = opt.AdamW(model.parameters(), lr=0.1)
    adam8 = opt.AdamW(model.parameters(), lr=0.01)
    adam9 = opt.AdamW(model.parameters(), lr=0.001)
    adam10 = opt.AdamW(model.parameters(), lr=0.0001)
    adam11 = opt.AdamW(model.parameters(), lr=0.00001)
    adam12 = opt.AdamW(model.parameters(), lr=0.000001)
    opts = [adam1, adam2, adam4, adam5, adam6, adam7, adam8, adam9, adam10, adam11, adam12]
    optNames = ['e-1WD','e-2WD','e-3WD','e-4WD','e-5WD','e-6WD','e-1','e-2','e-3','e-4','e-5','e-6']
    loss, acc, name = mainModel(model, modelName, mainLoss, opts, optNames, EPOCHS)
    finalAccs[name] = acc
    finalLosses[name] = loss
    
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
