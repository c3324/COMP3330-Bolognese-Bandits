import numpy as np
import torch
import torch.nn as nn
import torchtext
import datasets

#Importing our dataset
train_data, test_data = datasets.load_dataset("cardiffnlp/tweet_topic_single", split=["train_coling2022", "test_coling2022"])

#Splitting off our validation data set from our training data
train_valid_data = train_data.train_test_split(test_size=0.2)
train_data = train_valid_data['train']
valid_data = train_valid_data['test']