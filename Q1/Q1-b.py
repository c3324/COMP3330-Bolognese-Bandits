from torchvision.io import read_image
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
from torchvision.models import resnet50, ResNet50_Weights
import timeit
import time

from Q1 import Dataset
from trainer import trainer


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
warmup_epochs = 5
epochs = 15
learning_rate = 0.01
warmup_learning_rate = 1e-4
weight_decay = 0.1  #L2 regularization


# Initialize model with the best available weights
#weights = ResNet50_QuantizedWeights.DEFAULT
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 6)
model = model.to(device)
model.eval()

# Initialize the inference transforms
preprocess = weights.transforms()

# Set loss function and optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
warmup_optimizer = torch.optim.Adam(params=model.parameters(), lr=warmup_learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


print("Loading data...") #webcrawlerDataFolder='dataset/web_crawled',
train_data = Dataset("dataset/seg_train", transform_type='train', copy_amount=0, preload=False, img_resolution= resize_resolution, duplicate_smaller_samples=False, shrink_larger_samples=False)
#valid_data = Dataset("dataset/seg_train", webcrawlerDataFolder='dataset/web_crawled', transform=eval_transforms, use_data='last', data_amount=val_size, img_resolution= resize_resolution, preload=True, shrink_larger_samples=True)
test_data = Dataset("dataset/seg_test", transform_type='eval', img_resolution= resize_resolution, preload=True)

# Wrap in DataLoader objects with batch size and shuffling preference
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False) # note this is set to test set!
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
print("Begining Warmup Training...")

warmup_trainer = trainer(model, train_loader, valid_loader, warmup_epochs, warmup_optimizer, device, loss_fn)
warmup_trainer.train()

print("Begining Training...")
resnet_trainer = trainer(model, train_loader, valid_loader, epochs, optimizer, device, loss_fn)
train_losses, val_losses, train_accs, val_accs = resnet_trainer.train()


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



