from torchvision.io import read_image
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights
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
epochs = 10
learning_rate = 1e-3
val_size = 0.1
weight_decay = 0 #0.01 #L2 regularization


# Initialize model with the best available weights
#weights = ResNet50_QuantizedWeights.DEFAULT
# weights = ResNet50_Weights.IMAGENET1K_V2
# model = resnet50(weights=weights).to(device)
weights = VGG16_Weights.IMAGENET1K_V1
model = vgg16(weights=weights).to(device)
#num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(512, 6)
model = model.to(device)
model.eval()
# Initialize the inference transforms
preprocess = weights.transforms()



# Set loss function and optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()


# Set loss function and optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()


print("Loading data...") #webcrawlerDataFolder='dataset/web_crawled',
train_data = Dataset("dataset/seg_train", webcrawlerDataFolder='dataset/web_crawled', transform_type='train', data_amount=1-val_size, preload=False, img_resolution= resize_resolution, duplicate_smaller_samples=False, shrink_larger_samples=False)
valid_data = Dataset("dataset/seg_train", webcrawlerDataFolder='dataset/web_crawled', transform_type='eval', use_data='last', data_amount=val_size, img_resolution= resize_resolution, preload=True)
test_data = Dataset("dataset/seg_test", transform_type='eval', img_resolution= resize_resolution, preload=True)

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


# Test Evaluation
test_start_time = time.time()
# Track epoch loss and accuracy
accuracy, loss = 0, 0
# Switch model to evaluation (affects batch norm and dropout)
model.eval()
# Disable gradients
with torch.no_grad():
    # Iterate through batches
    preds = []
    labels = []
    for data, label in test_loader:
        # Move data to the used device
        label = label.type(torch.LongTensor) # move to outside of training loop for efficiency..
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        label = label.to(device)
        # Forward pass
        test_output = model(data)
        test_loss = loss_fn(test_output, label)
        # Compute metrics
        acc = ((test_output.argmax(dim=1) == label).float().mean())
        accuracy += acc/len(test_loader)
        loss += valid_loss/len(test_loader)
        preds += test_output.argmax(dim=1).cpu()
        labels += label.cpu()

print("Test accuracy = {:.2f}, loss = {:.5f}".format((accuracy*100), loss))

# Confusion matrix
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

#cm_analysis(labels, preds, classes)

