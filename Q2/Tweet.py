import os
import torch
from torch.utils.data import Dataset
import pandas as pd
class Tweets(Dataset):
    def __init__(self, path, vocab):
        self.path = path
        self.tweets = []
        self.labels = []
        self.vocab = vocab
        # self.vocab['<PAD>'] = len(self.vocab)
        # self.padIndex = self.vocab['<PAD>']
        self.tags = {0, 1, 2, 3, 4, 5}
        
        rawData = pd.read_json(path, lines=True)
        texts = rawData.text
        self.MAX_LEN = texts.str.len().max()
        labels = rawData.label
        
        for text in texts:
            for word in text:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        
        for text in texts:
            self.tweets.append(self.prepInputs(text))
        
        for label in labels:
            self.labels.append(self.prepTargets(label))
    
    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, idx):
        return self.tweets[idx], self.labels[idx]
        
        
    def prepInputs(self, sentance):
        res = torch.tensor([self.vocab[w] for w in sentance], dtype=torch.long)
        if len(res) < self.MAX_LEN:
            remaining = self.MAX_LEN - len(res)
            res = torch.cat((res, torch.zeros(remaining,dtype=torch.long)))
        return res


    def prepTargets(self, tag):
        tensor = torch.zeros(len(self.tags), dtype=torch.float)
        tensor[tag] = 1
        return tensor
        
        