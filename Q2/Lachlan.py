import numpy as np
import torch
import torch.nn as nn
import torchtext
import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt

#GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


#Retrieve data
train_data, test_data = load_dataset('cardiffnlp/tweet_topic_single', split=['train_coling2022', 'test_coling2022'])
train_valid_data = train_data.train_test_split(test_size=0.2)
train_data = train_valid_data['train']
valid_data = train_valid_data['test']

# Preview our splits
print(train_data, valid_data, test_data)

#Build tokenizer to tokenize all string inputs
tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
max_length = 600

def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example['text'])[:max_length]
    return {'tokens': tokens}

train_data = train_data.map(tokenize_example, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})
valid_data = valid_data.map(tokenize_example, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})
test_data = test_data.map(tokenize_example, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})

#Build vocabulary from all tokens in the training data
vocab = torchtext.vocab.build_vocab_from_iterator(train_data['tokens'], min_freq=5, specials=['<unk>','<pad>'])
vocab.set_default_index(vocab['<unk>'])
pad_index = vocab['<pad>']


#Numericalise training, validation, and test data by indexing all tokens according to the vocabulary
def numericalise_data(example, vocab):
    ids = [vocab[token] for token in example['tokens']]
    return {'ids': ids}

train_data = train_data.map(numericalise_data, fn_kwargs={'vocab': vocab})
valid_data = valid_data.map(numericalise_data, fn_kwargs={'vocab': vocab})
test_data = test_data.map(numericalise_data, fn_kwargs={'vocab': vocab})


#Convert to torch types, keeping ids and label
train_data = train_data.with_format(type='torch', columns=['ids', 'label'])
valid_data = valid_data.with_format(type='torch', columns=['ids', 'label'])
test_data = test_data.with_format(type='torch', columns=['ids', 'label'])


#For LSTM, we must retain the order of words.
#Model needs uniform length, so helper function must be created to pad sequences with zeroes each batch

def collate(batch):
    batch_ids = [i['ids'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_label = [i['label'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {'ids': batch_ids,
             'label': batch_label}
    return batch

#Wrap in data loader with given batch size, using collate function to padd as necessary

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, collate_fn=collate, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=128, collate_fn=collate)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, collate_fn=collate)




class LSTM(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed, padding_idx=pad_index)
        self.lstm = nn.LSTM(input_size=n_embed, hidden_size=32, num_layers=1, bidirectional=True)
        self.out = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.embedding(x) #Embed each token (B, T, C)
        x, _ = self.lstm(x) #Apply LSTM(B, T, C)
        x = torch.amax(x, dim=1) #Reduce sequence dim (B, C)
        x = self.out(x) #(B, 2)
        return x
    
    
#Instantiate the model
model = LSTM(vocab_size=len(vocab), n_embed=64).to(device)

#Define the optimiser and tell it what parameters to update, as well as the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss().to(device)

print(model)


#Train model

def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses, accuracies = [], []
    for batch in dataloader:
        inputs = batch['ids'].to(device)
        labels = batch['label'].to(device)
        #Reset the gradients for all variables
        optimizer.zero_grad()
        
        #Forward pass
        preds = model(inputs)
        
        #Calculate loss
        loss = loss_fn(preds, labels)
        
        #Backward pass
        loss.backward()
        
        #Adjust weights
        optimizer.step()
        
        #Log
        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.sum(torch.argmax(preds, dim=-1) == labels) / labels.shape[0]
        accuracies.append(accuracy.detatch().cpu().numpy())
        
    return np.mean(losses), np.mean(accuracies)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    losses, accuracies = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['ids'].to(device)
            labels = batch['label'].to(device)
        
        #forward pass
        preds = model(inputs)
        
        #Calculate loss
        loss = loss_fn(preds, labels)
        
        #Log
        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.sum(torch.argmax(preds, dim=-1) == labels) / labels.shape[0]
        accuracies.append(accuracy.detach().cpu().numpy())
    return np.mean(losses), np.mean(accuracies)

train_losses, train_accuracies = [],[]
valid_losses, valid_accuracties = [],[]

for epoch in range(5):
    train_loss, train_accuracy = train(model, train_dataloader, loss_fn, optimizer, device)
    valid_loss, valid_accuracy = evaluate(model, valid_dataloader, loss_fn, device)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f}, valid_loss={valid_loss:.4f}, valid_accuracy={valid_accuracy:.4f}")
    
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8), sharex=True)
ax1.plot(train_losses, color='b', label='train')
ax1.plot(valid_losses, color='r', label='valid')
ax1.set_ylabel("Loss")
ax1.legend()
ax2.plot(train_accuracies, color='b', label='train')
ax2.plot(valid_accuracies, color='r', label='valid')
ax2.set_ylabel("Accuracy")
ax2.set_xlabel("Epoch")
ax2.legend()

test_loss, test_accuracy = evluate(model, test_dataloader, loss_fn, device)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
