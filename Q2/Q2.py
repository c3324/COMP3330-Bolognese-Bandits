from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
import torchtext

#Loading the data set
train_data, test_data = load_dataset("cardiffnlp/tweet_topic_single", split=["train_coling2022", "test_coling2022"])

#Allocating 20% for validation
train_valid_data = train_data.train_test_split(test_size=0.2)
train_data = train_valid_data['train_coling2022']
valid_data = train_valid_data['test_coling2022']

train_data, valid_data, test_data