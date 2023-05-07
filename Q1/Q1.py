import torch
import pandas as pd
import numpy as np
import os
import glob
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Check which device is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))

### Parameters ###########################
batch_size = 128
resize_resolution = 256
epochs = 500
learning_rate = 1e-3



class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, transform=None, duplicate_smaller_samples=True, use_data='first', data_amount=1, copy_amount=0, preload=False, img_resolution=256):
        '''Dataset helper class for images containing buildings (0), forest (1), glacier (2), mountain (3), sea (4), street (5)'''
        '''params:      - data_folder:                  folder containing class folders
                        - transform:                    transforms to use on data
                        - duplicate_smaller_samples:    whether to equalize dataset sizes to reduce bias by copying samples from smaller classes
                        - use_data:                     index from first or last item - > useful for splitting data into training at test sets
                        - data_amount:                  float of amount of data to use where 1.0 = all available data
                        - copy_amount:                  int of number of times to copy dataset - > use with transform to reduce overfitting
                        - preload:                      preload images to ram to reduce disk usage.
                        - image_resolution:             resolution of images'''
        # Establish number of inputs for each class to reduce NN bias
        max_count = 0 
        self.preload = preload
        for dir in os.listdir(data_folder):
            class_folder = data_folder + "/" + dir
            file_counter = len(glob.glob1(class_folder, "*.jpg"))
            if file_counter > max_count:
                max_count = file_counter   
            
        class_num = 0
        self.labels = np.empty((0,1))
        self.image_paths = np.empty((0,1))
        for dir in os.listdir(data_folder):
            class_folder = data_folder + "/" + dir
            image_paths = glob.glob(os.path.join(class_folder, '*.jpg'))
            image_paths.sort()
            
            file_counter = len(glob.glob1(class_folder, "*.jpg"))
            
            if duplicate_smaller_samples:
                diff = max_count - file_counter
                as_pandas = pd.DataFrame(image_paths)
                dupes  = as_pandas.sample(diff, replace=True).to_numpy()
                image_paths = np.append(image_paths, dupes)
                
            if data_amount != 1:
                # shorthand train/val split..
                if use_data == 'first':
                    image_paths = image_paths[:int(len(image_paths) * data_amount)]
                elif use_data == 'last':
                    image_paths = image_paths[-int(len(image_paths) * data_amount):]
            
                
            self.image_paths = np.append(self.image_paths, image_paths) # ideally np.append/np.concat is not used as it will need to re-allocate memory each call, but this only happens 5 times?..
            
            temp_labels = np.full((len(image_paths), 1), class_num)
            self.labels = np.append(self.labels, temp_labels) # similarly..

            class_num += 1

        
        for i in range(copy_amount):
            # This will then be transformed slightly differently by the transform call.
            self.image_paths = np.append(self.image_paths, self.image_paths)
            self.labels = np.append(self.labels, self.labels)
         
        self.transform = transform
        self.n = len(self.image_paths)
        
      
        if preload:
            self.images = np.empty((self.n, 3, img_resolution, img_resolution)) # n_items, n_channels (RGB), img width, img height
            for i in range(self.n):
                img = Image.open(self.image_paths[i]) 
                self.images[i] = self.transform(img)
            #self.images = self.images.astype(torch.FloatTensor)
            
        #self.labels = self.labels.astype(torch.LongTensor)
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        """Get one example"""
        label = self.labels[idx]
        if self.preload != True:
            img_path = self.image_paths[idx]
            img = Image.open(img_path)
            img_transformed = self.transform(img)
            #img_transformed = img_transformed.astype(torch.FloatTensor)
            return img_transformed, label
        else:
            return self.images[idx], label
            
    

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
    
# Create our dataset splits
print("Preloading data...")
train_data = Dataset("dataset/seg_train", transform=train_transforms, data_amount=0.7, copy_amount=0, preload=True)
valid_data = Dataset("dataset/seg_train", transform=eval_transforms, use_data='last', data_amount=0.3, duplicate_smaller_samples=False)
test_data = Dataset("dataset/seg_test", transform=eval_transforms, duplicate_smaller_samples=False)


# Check our dataset sizes
print("Train: {} examples".format(len(train_data)))
print("Valid: {} examples".format(len(valid_data)))
print("Test: {} examples".format(len(test_data)))


# Wrap in DataLoader objects with batch size and shuffling preference
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Check number of batches
print("Train: {} batches".format(len(train_loader)))
print("Valid: {} batches".format(len(valid_loader))) 
print("Test: {} batches".format(len(test_loader))) 

print(train_data[0][0].shape)
print(valid_data[0][0].shape)

# fig = plt.figure()
# fig.set_figheight(12)
# fig.set_figwidth(12)
# for idx in range(16):
#     ax = fig.add_subplot(4,4,idx+1)
#     ax.axis('off')
#     ax.set_title(train_data[idx+6144][1])
#     plt.imshow(train_data[idx+6144][0].permute(1,2,0))
# plt.axis('off')
# plt.show()

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(16),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(32),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc_layer_1 = torch.nn.Sequential(
            torch.nn.Linear(64*256, 256),
            torch.nn.Dropout(),
            torch.nn.GELU()
        )
        self.fc_layer_2 = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(),
            torch.nn.GELU()
        )
        self.fc_layer_3 = torch.nn.Linear(128, 6)


    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_5(x)
        #x = x.view(x.size(0), -1) # Flatten
        x = x.reshape(x.shape[0], -1) # Flatten without accounting for batch size
        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)
        x = self.fc_layer_3(x)
        return x
    
def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)

# Create an object from the model
model = Model()
model.apply(init_weights)
model.to(device)


# Set loss function and optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

print("Begining training...")

train_losses, val_losses = [], []
train_accs, val_accs = [], []
# Train for 10 epochs
for epoch in range(epochs):
    ### Training
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
    print('Epoch: {}, train accuracy: {:.2f}%, train loss: {:.4f}'.format(epoch+1, epoch_accuracy*100, epoch_loss))
    train_losses.append(epoch_loss.item())
    train_accs.append(epoch_accuracy.item())
    ### Evaluation
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
    print('Epoch: {}, val accuracy: {:.2f}%, val loss: {:.4f}'.format(epoch+1, epoch_valid_accuracy*100, epoch_valid_loss))
    val_losses.append(epoch_valid_loss.item())
    val_accs.append(epoch_valid_accuracy.item())
    
    
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8), sharex=True)
ax1.plot(train_losses, color='b', label='train')
ax1.plot(val_losses, color='g', label='valid')
ax1.set_ylabel("Loss")
ax1.legend()
ax2.plot(train_accs, color='b', label='train')
ax2.plot(val_accs, color='g', label='valid')
ax2.set_ylabel("Accuracy")
ax2.set_xlabel("Epoch")
ax2.legend()


test_accuracy, test_loss = 0, 0
with torch.no_grad():
    # Iterate through batches
    for data, label in test_loader:
        # Move data to the used device
        label = label.type(torch.LongTensor) # move to outside of training loop for efficiency..
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        label = label.to(device)
        # Forward pass
        test_output_i = model(data)
        test_loss_i = loss_fn(test_output_i, label)
        # Compute metrics
        acc = ((test_output_i.argmax(dim=1) == label).float().mean())
        test_accuracy += acc/len(test_loader)
        test_loss += test_loss_i/len(test_loader)

print("Test loss: {:.4f}".format(test_loss))
print("Test accuracy: {:.2f}%".format(test_accuracy*100))