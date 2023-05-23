import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Hidden32(nn.Module):
    def __init__(self, vocabSize, tagSize):
        super(Hidden32, self).__init__()
        EMBEDDING_SIZE = 512
        HIDDEN_SIZE = 128
        self.embedding = nn.Embedding(vocabSize, embedding_dim=EMBEDDING_SIZE)
        self.ltsm = nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, num_layers=1, bidirectional=True)
        self.hidden = nn.Linear(HIDDEN_SIZE*2, tagSize)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.ltsm(x)
        x = torch.amax(x, dim=1)
        x = self.hidden(x)
        # print(x)
        return self.activation(x)
    
class ExtraLSTMLayers(nn.Module):
    def __init__(self, vocabSize, tagSize, LSTMLayers):
        super(ExtraLSTMLayers, self).__init__()
        EMBEDDING_SIZE = 512
        HIDDEN_SIZE = 128
        self.embedding = nn.Embedding(vocabSize, embedding_dim=EMBEDDING_SIZE)
        self.ltsm = nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, num_layers=LSTMLayers, bidirectional=True)
        self.activation = nn.Sigmoid()
        self.hidden = nn.Linear(HIDDEN_SIZE*2, tagSize)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.ltsm(x)
        x = torch.amax(x, dim=1)
        x = self.hidden(x)
        # print(x)
        return self.activation(x)
    
class Hidden32WithExtraEmbedding(nn.Module):
    def __init__(self, vocabSize, tagSize):
        super(Hidden32WithExtraEmbedding, self).__init__()
        EMBEDDING_SIZE = 512
        HIDDEN_SIZE = 128
        self.embedding = nn.Embedding(vocabSize, embedding_dim=EMBEDDING_SIZE)
        self.ltsm = nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, num_layers=1, bidirectional=True)
        self.hidden = nn.Linear(HIDDEN_SIZE*2, tagSize)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.ltsm(x)
        x = torch.amax(x, dim=1)
        x = self.hidden(x)
        
        # print(x)
        return self.activation(x)
    
class logSoftMax(nn.Module):
    def __init__(self, vocabSize, tagSize):
        super(logSoftMax, self).__init__()
        EMBEDDING_SIZE = 30
        HIDDEN_SIZE = 32
        self.embedding = nn.Embedding(vocabSize, embedding_dim=EMBEDDING_SIZE)
        self.ltsm = nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, num_layers=1, bidirectional=True)
        self.hidden = nn.Linear(HIDDEN_SIZE*2, tagSize)
        
    def forward(self, input):
        print(input.shape)
        x = self.embedding(input)
        print(x.shape)
        x, _ = self.ltsm(x.view(len(input),1,-1))
        print(x.shape)
        x = self.hidden(x.view(len(input),-1))
        print(x.shape)
        x = F.log_softmax(x, dim=1)
        print(x.shape)
        # x = x.long()
        # print(x)
        return x
    
class Hidden8_16_8(nn.Module):
    def __init__(self, vocabSize, tagSize):
        super(Hidden8_16_8, self).__init__()
        EMBEDDING_SIZE = 16
        HIDDEN_SIZE = 8
        self.embedding = nn.Embedding(vocabSize, embedding_dim=EMBEDDING_SIZE)
        self.ltsm = nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, num_layers=1, bidirectional=True)
        self.hidden1 = nn.Linear(HIDDEN_SIZE*2, 16)
        self.hidden2 = nn.Linear(16, 8)
        self.activation = nn.Sigmoid()
        self.hidden3 = nn.Linear(8, tagSize)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.ltsm(x)
        x = torch.amax(x, dim=1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        
        # print(x)
        return self.activation(x)
