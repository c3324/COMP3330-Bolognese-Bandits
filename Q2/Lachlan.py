import numpy as np
import torch
import torch.nn as nn
import torchtext
import datasets
from datasets import load_dataset



dataset = load_dataset("cardiffnlp/tweet_topic_single", split=["train", "test"])

train_data, test_data = dataset
train_valid_data = train_data.train_test_split(test_size=0.2)
train_data = train_valid_data['train']
valid_data = train_valid_data['test']
# Preview our splits
print(train_data, valid_data, test_data)

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
max_length = 600

def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example['text'])[:max_length]
    return {'tokens': tokens}

# class LSTM(nn.Module):
#     def __init__(self, vocab_size, n_embed):
#         super().__init__()
#         self.embedding() = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed, padding_idx=pad_index)
#         self.lstm = nn.LSTM(input_size=n_embed, hidden_size=32, num_layers=1, bidirectional=True)
#         self.out = nn.Linear(64, 2)