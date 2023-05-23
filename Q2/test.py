import numpy as np
import torch
import torch.nn as nn
import torchtext
import datasets

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# The dataset has a predefined train (25,000 examples) and test (25,000 examples) split
train_data, test_data = datasets.load_dataset("cardiffnlp/tweet_topic_single", split=["train_coling2022", "test_coling2022"])
# Split off 5,000 examples from train_data for validation
train_valid_data = train_data.train_test_split(test_size=0.1)
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
pad_index = vocab['<pad>']

def numericalize_data(example, vocab):
    ids = [vocab[token] for token in example['tokens']]
    return {'ids': ids}

train_data = train_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
valid_data = valid_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
test_data = test_data.map(numericalize_data, fn_kwargs={'vocab': vocab})

train_data = train_data.with_format(type='torch', columns=['ids', 'label'])
valid_data = valid_data.with_format(type='torch', columns=['ids', 'label'])
test_data = test_data.with_format(type='torch', columns=['ids', 'label'])

def collate(batch):
    batch_ids = [i['ids'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_label = [i['label'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch = {'ids': batch_ids,
             'label': batch_label}
    return batch


train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=512, collate_fn=collate, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=512, collate_fn=collate)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=512, collate_fn=collate)


class EncoderBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.sa = nn.MultiheadAttention(
            embed_dim=n_embed, num_heads=n_head)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embed, 2*n_embed),
            nn.ReLU(),
            nn.Linear(2*n_embed, n_embed),
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)        
    def forward(self, x):
        x = x + self.sa(x, x, x)[0]
        x = self.ln1(x)
        x = x + self.ffwd(x)
        x = self.ln2(x)
        return x
    
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embed, n_layer, n_head):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=n_embed, padding_idx=pad_index)
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=n_embed)
        self.blocks = nn.Sequential(*[EncoderBlock(n_embed=n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.out = nn.Linear(n_embed, 6)
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln(x) # (B, T, C)
        x = torch.amax(x, dim=1) # Reduce sequence dim (B, C)
        x = self.out(x)
        return x

# Instantiate a small Transformer model with 1 block and 2 heads
model = Transformer(vocab_size=len(vocab), n_embed=4, n_layer=2, n_head=2).to(device)

# Define the optimiser and tell it what parameters to update, as well as the loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.25)
loss_fn = nn.CrossEntropyLoss().to(device)

print(model)


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses, accuracies = [], []
    for batch in dataloader:
        inputs = batch['ids'].to(device)
        labels = batch['label'].to(device)
        # Reset the gradients for all variables
        optimizer.zero_grad()
        # Forward pass
        preds = model(inputs)
        # print(preds)
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
            inputs = batch['ids'].to(device)
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
for epoch in range(300):
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

test_loss, test_accuracy = evaluate(model, test_dataloader, loss_fn, device)
print("Test loss: {:.4f}".format(test_loss))
print("Test accuracy: {:.4f}".format(test_accuracy))
plt.show()