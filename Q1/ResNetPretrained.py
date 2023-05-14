from torchvision.io import read_image
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
from torchvision.models import resnet50, ResNet50_Weights
import time
from Q1 import Dataset
from PIL import Image, ImageFile
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# Check which device is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))



### Parameters ###########################
batch_size = 128
resize_resolution = 128
epochs = 100
learning_rate = 1e-3
val_size = 0.1


# Initialize model with the best available weights
weights = ResNet50_QuantizedWeights.DEFAULT
model = resnet50(weights=weights, quantize=True)
model.eval()

# Initialize the inference transforms
preprocess = weights.transforms()

# Set loss function and optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()

# Apply inference preprocessing transforms
# Transforms to prepare data and perform data augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(scale=(0.6, 1.0), size=(resize_resolution,resize_resolution)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    #transforms.Normalize( mean = (0,0,0), std = (1,1,1)), #ToTensor already normalizes?
])
eval_transforms = transforms.Compose([
    transforms.Resize(size=(resize_resolution,resize_resolution)),
    transforms.ToTensor(),
    #transforms.Normalize( mean = (0,0,0), std = (1,1,1)),
])
train_data = Dataset("dataset/seg_train", webcrawlerDataFolder='dataset/web_crawled', transform=train_transforms, data_amount=(1-val_size), copy_amount=0, preload=False, img_resolution= resize_resolution, duplicate_smaller_samples=False, shrink_larger_samples=True)
valid_data = Dataset("dataset/seg_train", webcrawlerDataFolder='dataset/web_crawled', transform=eval_transforms, use_data='last', data_amount=val_size, img_resolution= resize_resolution, preload=True, shrink_larger_samples=True)
test_data = Dataset("dataset/seg_test", transform=eval_transforms, img_resolution= resize_resolution)

# Wrap in DataLoader objects with batch size and shuffling preference
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# Evaluate before training
val_start_time = time.time()
# Track epoch loss and accuracy
accuracy, loss = 0, 0
# Switch model to evaluation (affects batch norm and dropout)
model.eval()
# Disable gradients
with torch.no_grad():
    # Iterate through batches
    for data, label in valid_loader:
        # Move data to the used device
        label = label.type(torch.LongTensor) # move to outside of training loop for efficiency..
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        label = label.to(device)
        # Forward pass
        valid_output = model(data)
        valid_loss = loss_fn(valid_output, label)
        # Compute metrics
        acc = ((valid_output.argmax(dim=1) == label).float().mean())
        accuracy += acc/len(valid_loader)
        loss += valid_loss/len(valid_loader) 

print('Initial Pretrained model Accuracy:   {:.2f}%,   Loss: {:.4f}   |   {:.2f}s'.format(accuracy*100, loss, time.time()-val_start_time))


# Train Model on new Data
print("Begining training...")

train_losses, val_losses = [], []
train_accs, val_accs = [], []
# Train for 10 epochs
for epoch in range(epochs):
    ### Training
    epoch_start_time = time.time()
    # Track epoch loss and accuracy
    epoch_loss, epoch_accuracy = 0, 0
    # Switch model to training (affects batch norm and dropout)
    model.train()
    # Iterate through batches
    for i, (data, label) in enumerate(train_loader):
        # Reset gradients
        optimizer.zero_grad()
        # Move data to the used device
        label = label.type(torch.LongTensor) # move to outside of training loop for efficiency..
        data = data.type(torch.FloatTensor)
        data = preprocess(data).unsqueeze(0)
        data = data.to(device)
        label = label.to(device)
        # Forward pass
        output = model(data)
        loss = loss_fn(output, label)
        # Backward pass
        loss.backward()
        # Adjust weights
        optimizer.step()
        # Compute metrics
        acc = ((output.argmax(dim=1) == label).float().mean())
        #print(label, output.argmax(dim=1))
        epoch_accuracy += acc/len(train_loader)
        epoch_loss += loss/len(train_loader)
    
    train_losses.append(epoch_loss.item())
    train_accs.append(epoch_accuracy.item())
    print('Epoch: {}, train accuracy: {:.2f}%, train loss: {:.4f}   |   {:.2f}s/epoch'.format(epoch+1, epoch_accuracy*100, epoch_loss,time.time()-epoch_start_time))
    
    ### Evaluation
    val_start_time = time.time()
    # Track epoch loss and accuracy
    epoch_valid_accuracy, epoch_valid_loss = 0, 0
    # Switch model to evaluation (affects batch norm and dropout)
    model.eval()
    # Disable gradients
    with torch.no_grad():
        # Iterate through batches
        for data, label in valid_loader:
            # Move data to the used device
            label = label.type(torch.LongTensor) # move to outside of training loop for efficiency..
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            label = label.to(device)
            # Forward pass
            valid_output = model(data)
            valid_loss = loss_fn(valid_output, label)
            # Compute metrics
            acc = ((valid_output.argmax(dim=1) == label).float().mean())
            epoch_valid_accuracy += acc/len(valid_loader)
            epoch_valid_loss += valid_loss/len(valid_loader) 
   
    val_losses.append(epoch_valid_loss.item())
    val_accs.append(epoch_valid_accuracy.item())
    print('Epoch: {}, val accuracy:   {:.2f}%,   val loss: {:.4f}   |   {:.2f}s'.format(epoch+1, epoch_valid_accuracy*100, epoch_valid_loss,time.time()-val_start_time))



plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='validated loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training Losses')
plt.legend()
plt.show()

plt.plot(train_accs, label='train accuracy')
plt.plot(val_accs, label='validated accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracies')
plt.legend()
plt.show()



