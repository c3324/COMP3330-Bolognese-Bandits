import numpy as np
import torch
import torch.nn as nn
import torchtext
import datasets

# Check which device is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))

# Params
epochs = 100
learning_rate = 1e-4

# The dataset has a predefined train (25,000 examples) and test (25,000 examples) split
train_data, test_data = datasets.load_dataset("cardiffnlp/tweet_topic_single", split=["train_coling2022", "test_coling2022"])
# Split off 5,000 examples from train_data for validation
train_valid_data = train_data.train_test_split(test_size=0.2)
train_data = train_valid_data['train']
valid_data = train_valid_data['test']
# Preview our splits
train_data, valid_data, test_data

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
max_length = 600

def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example['text'])[:max_length]
    return {'tokens': tokens}

train_data = train_data.map(tokenize_example, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})
valid_data = valid_data.map(tokenize_example, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})
test_data = test_data.map(tokenize_example, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length})

vocab = torchtext.vocab.build_vocab_from_iterator(train_data['tokens'],
                                                  min_freq=5,
                                                  specials=['<unk>', '<pad>'])
vocab.set_default_index(vocab['<unk>'])

def numericalize_data(example, vocab):
    ids = [vocab[token] for token in example['tokens']]
    return {'ids': ids}

train_data = train_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
valid_data = valid_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
test_data = test_data.map(numericalize_data, fn_kwargs={'vocab': vocab})


def multi_hot_data(example, num_classes):
    encoded = np.zeros((num_classes,))
    encoded[example['ids']] = 1 
    return {'multi_hot': encoded}

train_data = train_data.map(multi_hot_data, fn_kwargs={'num_classes': len(vocab)})
valid_data = valid_data.map(multi_hot_data, fn_kwargs={'num_classes': len(vocab)})
test_data = test_data.map(multi_hot_data, fn_kwargs={'num_classes': len(vocab)})

train_data = train_data.with_format(type='torch', columns=['multi_hot', 'label'])
valid_data = valid_data.with_format(type='torch', columns=['multi_hot', 'label'])
test_data = test_data.with_format(type='torch', columns=['multi_hot', 'label'])

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=128)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128)


class BoW(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.hidden1 = nn.Linear(vocab_size, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.hidden3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 6)
    def forward(self, x):
        x = nn.ReLU()(self.hidden1(x))
        x = nn.ReLU()(self.hidden2(x))
        x = nn.ReLU()(self.hidden3(x))
        x = self.out(x)
        return x

# Instantiate the Model
model = BoW(vocab_size=len(vocab)).to(device)

# Define the optimiser and tell it what parameters to update, as well as the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss().to(device)

def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses, accuracies = [], []
    for batch in dataloader:
        inputs = batch['multi_hot'].to(device)
        labels = batch['label'].to(device)
        # Reset the gradients for all variables
        optimizer.zero_grad()
        # Forward pass
        preds = model(inputs)
        # Calculate loss
        loss = loss_fn(preds, labels)
        # Backward pass
        loss.backward()
        # Adjust weights
        optimizer.step()
        # Log
        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.sum(torch.argmax(preds, dim=-1) == labels) / labels.shape[0]
        accuracies.append(accuracy.detach().cpu().numpy())
    return np.mean(losses), np.mean(accuracies)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    losses, accuracies = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['multi_hot'].to(device)
            labels = batch['label'].to(device)
            # Forward pass
            preds = model(inputs)
            # Calculate loss
            loss = loss_fn(preds, labels)
            # Log
            losses.append(loss.detach().cpu().numpy())
        accuracy = torch.sum(torch.argmax(preds, dim=-1) == labels) / labels.shape[0]
        accuracies.append(accuracy.detach().cpu().numpy())
    return np.mean(losses), np.mean(accuracies)


train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []
for epoch in range(epochs):
    # Train
    train_loss, train_accuracy = train(model, train_dataloader, loss_fn, optimizer, device)
    # Evaluate
    valid_loss, valid_accuracy = evaluate(model, valid_dataloader, loss_fn, device)
    # Log
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)
    print("Epoch {}: train_loss={:.4f}, train_accuracy={:.4f}, valid_loss={:.4f}, valid_accuracy={:.4f}".format(
        epoch+1, train_loss, train_accuracy, valid_loss, valid_accuracy))
    
import matplotlib.pyplot as plt
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
plt.show()

test_loss, test_accuracy = evaluate(model, test_dataloader, loss_fn, device)
print("Test loss: {:.4f}".format(test_loss))
print("Test accuracy: {:.4f}".format(test_accuracy))