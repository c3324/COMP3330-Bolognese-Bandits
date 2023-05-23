from trainer import trainer

import torch
import pandas as pd
import numpy as np
import os
import glob
import time
from PIL import Image, ImageFile
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Check which device is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))

#ImageFile Fix
ImageFile.LOAD_TRUNCATED_IMAGES = True

### Parameters ###########################
batch_size = 128
resize_resolution = 128
epochs = 50
learning_rate = 1e-3
val_size = 0.1
weight_decay = 0 #0.01 #L2 regularization

CNN_layers = [16, 32, 32, 64, 64, 128]
Deep_layers = [256, 128]
dropout = 0


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, webcrawlerDataFolder=None, transform_type='train', duplicate_smaller_samples=False, shrink_larger_samples=False, use_data='first', data_amount=1, copy_amount=0, preload=False, img_resolution=256):
        '''Dataset helper class for images containing buildings (0), forest (1), glacier (2), mountain (3), sea (4), street (5)'''
        '''params:      - data_folder:                  folder containing class folders
                        - transform:                    transforms to use on data
                        - duplicate_smaller_samples:    whether to equalize dataset sizes to reduce bias by copying samples from smaller classes
                        - use_data:                     index from first or last item - > useful for splitting data into training at test sets
                        - data_amount:                  float of amount of data to use where 1.0 = all available data
                        - copy_amount:                  int of number of times to copy dataset - > use with transform to reduce overfitting -> useful when not transforming at train time.
                        - preload:                      preload images to ram to reduce disk usage.
                        - image_resolution:             resolution of images'''
        # Establish number of inputs for each class to reduce NN bias
        max_count = 0 
        self.preload = preload
        class_counts = {
            'buildings':0,
            'forest':0,
            'glacier':0,
            'mountain':0,
            'sea':0,
            'street':0
        }
        for dir in os.listdir(data_folder):
            class_folder = data_folder + "/" + dir
            file_counter = len(glob.glob1(class_folder, "*.jpg"))
            class_counts[dir] += file_counter
            if webcrawlerDataFolder == None:
                if file_counter > max_count:
                    max_count = file_counter   
        
        if webcrawlerDataFolder != None:
            for dir in os.listdir(webcrawlerDataFolder):
                class_folder = webcrawlerDataFolder + "/" + dir
                file_counter = len(glob.glob1(class_folder, "*.jpg"))
                class_counts[dir] += file_counter
                if class_counts[dir] > max_count:
                    max_count = class_counts[dir] 

        min_count = min(class_counts.values())
        print(max_count, class_counts)
        class_num = 0
        self.labels = np.empty((0,1))
        self.image_paths = np.empty((0,1))
        print("Found folders for", os.listdir(data_folder))
        for dir in os.listdir(data_folder):
            class_folder = data_folder + "/" + dir
            image_paths = glob.glob(os.path.join(class_folder, '*.jpg'))
            image_paths.sort()
            print("Fetching paths for", dir, "with label", class_num)
            
            file_counter = class_counts[dir]
            
            if webcrawlerDataFolder != None:
                web_class_folder = webcrawlerDataFolder + "/" + dir
                new_image_paths = glob.glob(os.path.join(web_class_folder, '*.jpg'))
                new_image_paths.sort()
                image_paths = np.append(image_paths, new_image_paths)
                
            if data_amount != 1:
                # shorthand train/val split..
                if use_data == 'first':
                    image_paths = image_paths[:int(len(image_paths) * data_amount)]
                elif use_data == 'last':
                    image_paths = image_paths[-int(len(image_paths) * data_amount):]
                
            if duplicate_smaller_samples:
                diff = int((max_count * data_amount) - len(image_paths))
                as_pandas = pd.DataFrame(image_paths)
                dupes  = as_pandas.sample(diff, replace=True).to_numpy()
                image_paths = np.append(image_paths, dupes)
                
            elif shrink_larger_samples: # if not duplicate -> shrink larger samples
                #diff = int((min_count * data_amount) - len(image_paths))
                sample_size = int(min_count * data_amount)
                as_pandas = pd.DataFrame(image_paths)
                dupes  = as_pandas.sample(sample_size).to_numpy()
                image_paths = dupes # drop excess samples
                
            print(dir, "has", len(image_paths), "samples")

                    
            self.image_paths = np.append(self.image_paths, image_paths) # ideally np.append/np.concat is not used as it will need to re-allocate memory each call, but this only happens 5 times?..
            
            temp_labels = np.full((len(image_paths), 1), class_num)
            self.labels = np.append(self.labels, temp_labels) # similarly..

            class_num += 1

        for idx, path in enumerate(self.image_paths): # thought was cause for read_iamge bug
            if "\\" in path:
                #print("Found \\ in", path)
                self.image_paths[idx] = str(path).replace('\\', '/')
                #print(self.image_paths[idx])
                
        # #exit()
        
        for i in range(copy_amount):
            # This will then be transformed slightly differently by the transform call.
            self.image_paths = np.append(self.image_paths, self.image_paths)
            self.labels = np.append(self.labels, self.labels)
         
        self.transform_type = transform_type
        self.n = len(self.image_paths)
        
        self.train_transforms = torch.nn.Sequential(
            transforms.RandomResizedCrop(scale=(0.6, 1.0), size=(resize_resolution,resize_resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.RandomGrayscale(),
            transforms.RandomPerspective(),
            transforms.RandomErasing()
            #transforms.ToTensor(), # already a tensor
            #transforms.Normalize( mean = (0,0,0), std = (1,1,1)), #ToTensor already normalizes
        )
        self.eval_transforms = eval_transforms = torch.nn.Sequential(
            transforms.Resize(size=(resize_resolution,resize_resolution)),
            #transforms.ToTensor(),
            #transforms.Normalize( mean = (0,0,0), std = (1,1,1)),
        )
        self.pil_transform = transforms.ToPILImage()
        self.tensor_transform = transforms.ToTensor()

      
        if preload:
            print("Preloading data to main memory...")
            self.images = np.empty((self.n, 3, img_resolution, img_resolution)) # n_items, n_channels (RGB), img width, img height
            for i in range(self.n):
                #img = Image.open(self.image_paths[i]).convert('RGB') #.convert RGB forces images with alpha channel into 3 RGB channels
                #print(self.image_paths[i])
                img = read_image(str(self.image_paths[i]), ImageReadMode.RGB)
                #img = self.resize_transform(img).type(torch.FloatTensor).to(device)
                #img = self.transform(img).cpu()
                self.images[i] = self.eval_transform(img).cpu()
            #self.images = self.images.astype(torch.FloatTensor)
            
        #self.labels = self.labels.astype(torch.LongTensor)
        
    def train_transform(self, img):
        # cuda transforms
        img = self.train_transforms(img)
        # below is oddly efficient normalization to [0, 1]
        img = self.pil_transform(img)
        #img.show()
        img = self.tensor_transform(img)
        return img  
    
    def eval_transform(self, img):
        img = self.eval_transforms(img)
        img = self.pil_transform(img)
        #img.show()
        img = self.tensor_transform(img)
        return img

        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        """Get one example"""
        label = self.labels[idx]
        if self.preload != True:
            img_path = self.image_paths[idx]
            #img = Image.open(img_path).convert('RGB') #.convert RGB forces images with alpha channel into 3 RGB channels
            img = read_image(str(img_path), ImageReadMode.RGB).to(device)
            if self.transform_type == 'train':
                img_transformed = self.train_transform(img)
            else:
                img_transformed = self.eval_transform(img)
            # img = self.resize_transform(img)
            # img_transformed = self.transform(img) 
            # img_transformed = self.normalize(img_transformed)
            return img_transformed, label
            
            
        else:
            return self.images[idx], label
            
    


# class Model(torch.nn.Module):
#     def __init__(self):#, dropout_amount=0.1, input_dropout=0.5):
#         super(Model, self).__init__()
#         self.conv_layer_0 = torch.nn.Sequential(
#             torch.nn.Conv2d(3, 16, kernel_size=7, padding='same'),
#             #torch.nn.Dropout(input_dropout),
#             #torch.nn.BatchNorm2d(16),
#             torch.nn.SELU(),
#             torch.nn.MaxPool2d(2)
#             #torch.nn.AvgPool2d(2, stride=2)
#         )
#         self.conv_layer_1_1 = torch.nn.Sequential(
#             torch.nn.Conv2d(16, 64, kernel_size=5, padding='same'),
#             #torch.nn.Dropout(dropout_amount),
#             #torch.nn.BatchNorm2d(64),
#             torch.nn.SELU(),
#             torch.nn.MaxPool2d(2)
#         )
#         self.conv_layer_1_2 = torch.nn.Sequential(
#             torch.nn.Conv2d(64, 64, kernel_size=5, padding='same'),
#             #torch.nn.Dropout(dropout_amount),
#             #torch.nn.BatchNorm2d(64),
#             torch.nn.SELU(),
#             torch.nn.MaxPool2d(2)
#         )
#         self.conv_layer_1_3 = torch.nn.Sequential(
#             torch.nn.Conv2d(64, 64, kernel_size=5, padding='same'),
#             #torch.nn.Dropout(dropout_amount),
#             #torch.nn.BatchNorm2d(64),
#             torch.nn.SELU(),
#             torch.nn.MaxPool2d(2)
#         )
#         self.conv_layer_1_3 = torch.nn.Sequential(
#             torch.nn.Conv2d(64, 128, kernel_size=3, padding='same'),
#             #torch.nn.Dropout(dropout_amount),
#             #torch.nn.BatchNorm2d(128),
#             torch.nn.SELU(),
#             torch.nn.MaxPool2d(2)
#         )
#         self.conv_layer_2_1 = torch.nn.Sequential(
#             torch.nn.Conv2d(128, 128, kernel_size=3, padding='same'),
#             #torch.nn.Dropout(dropout_amount),
#             #torch.nn.BatchNorm2d(128),
#             torch.nn.SELU(),
#             torch.nn.MaxPool2d(2)
#         )
#         self.conv_layer_2_2 = torch.nn.Sequential(
#             torch.nn.Conv2d(128, 128, kernel_size=3, padding='same'),
#             #torch.nn.Dropout(dropout_amount),
#             #torch.nn.BatchNorm2d(128),
#             torch.nn.SELU(),
#             torch.nn.MaxPool2d(2)
#         )
#         self.conv_layer_2_3 = torch.nn.Sequential(
#             torch.nn.Conv2d(128, 128, kernel_size=3, padding='same'),
#             #torch.nn.Dropout(dropout_amount),
#             #torch.nn.BatchNorm2d(128),
#             torch.nn.SELU(),
#             torch.nn.MaxPool2d(2)
#         )
#         self.fc_layer_1 = torch.nn.Sequential(
#             torch.nn.Linear(128, 1024),
#             #torch.nn.Dropout(dropout_amount),
#             #torch.nn.BatchNorm1d(1024),
#             torch.nn.Dropout(),
#             torch.nn.SELU()
#         )
#         self.fc_layer_2 = torch.nn.Sequential(
#             torch.nn.Linear(1024, 512),
#             #torch.nn.Dropout(dropout_amount),
#             #torch.nn.BatchNorm1d(512),
#             torch.nn.Dropout(),
#             torch.nn.SELU()
#         )
#         self.fc_layer_3 = torch.nn.Sequential(
#             torch.nn.Linear(512, 128),
#             #torch.nn.Dropout(dropout_amount),
#             #torch.nn.BatchNorm1d(128),
#             torch.nn.Dropout(),
#             torch.nn.SELU()
#         )
#         self.fc_layer_outputs = torch.nn.Linear(128, 6)


#     def forward(self, x):
#         x = self.conv_layer_0(x)
#         x = self.conv_layer_1_1(x)
#         x = self.conv_layer_1_2(x)
#         x = self.conv_layer_1_3(x)
#         x = self.conv_layer_2_1(x)
#         x = self.conv_layer_2_2(x)
#         x = self.conv_layer_2_3(x)
#         #x = x.view(x.size(0), -1) # Flatten
#         x = x.reshape(x.shape[0], -1) # Flatten without accounting for batch size
#         x = self.fc_layer_1(x)
#         x = self.fc_layer_2(x)
#         x = self.fc_layer_3(x)
#         x = self.fc_layer_outputs(x)
#         return x
    
class Model(torch.nn.Module):
    def __init__(self, CNN_layers=[], Dense_layers=[], dropout=0, activation_function=torch.nn.ReLU):#, dropout_amount=0.1, input_dropout=0.5):
        super(Model, self).__init__()
        self.cnn_layers = torch.nn.ModuleList()
        self.cnn_layers.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, kernel_size=3, padding='same'),
                torch.nn.BatchNorm2d(16),
                activation_function(),
                torch.nn.MaxPool2d(2)
            )
        )
        if dropout != 0:
            self.cnn_layers.append(torch.nn.Dropout(dropout))
        prev_layer = 16
        
        for layer in CNN_layers:
            self.cnn_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(prev_layer, layer, kernel_size=3, padding='same'),
                    torch.nn.BatchNorm2d(layer),
                    activation_function(),
                    torch.nn.MaxPool2d(2)
                )
            )
            if dropout != 0:
                self.cnn_layers.append(torch.nn.Dropout(dropout))
            prev_layer = layer
            
        prev_layer = 128
        self.dense_layers = torch.nn.ModuleList()
        for layer in Dense_layers:
            self.dense_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(prev_layer, layer),
                    activation_function()
                )
            )
            if dropout != 0:
                self.dense_layers.append(torch.nn.Dropout(dropout))
            prev_layer = layer
        self.fc_layer_outputs = torch.nn.Linear(prev_layer, 6)


    def forward(self, x):
        for layer in self.cnn_layers:
            x = layer(x)
        x = x.view(x.size(0), -1) # Flatten
        for layer in self.dense_layers:
            x = layer(x)
        x = self.fc_layer_outputs(x)
        return x

def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)


if __name__ == '__main__':

    
    # Create our dataset splits
    print("Fetching data...") #webcrawlerDataFolder='dataset/web_crawled,'
    train_data = Dataset("dataset/seg_train", transform_type='train', data_amount=1, copy_amount=0, preload=False, img_resolution= resize_resolution, duplicate_smaller_samples=False, shrink_larger_samples=False)
    #valid_data = Dataset("dataset/seg_train", webcrawlerDataFolder='dataset/web_crawled', transform=eval_transforms, use_data='last', data_amount=val_size, img_resolution= resize_resolution, preload=True)
    test_data = Dataset("dataset/seg_test", transform_type='eval', img_resolution= resize_resolution, preload=True)


    # Check our dataset sizes
    print("Train: {} examples".format(len(train_data)))
    #print("Valid: {} examples".format(len(valid_data)))
    print("Test: {} examples".format(len(test_data)))


    # Wrap in DataLoader objects with batch size and shuffling preference
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    #valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    #test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False) # swapped to inspect...
    valid_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


    # Check number of batches
    print("Train: {} batches".format(len(train_loader)))
    print("Valid: {} batches".format(len(valid_loader))) 
    print("Test: {} batches".format(len(test_loader))) 

    # Create an object from the model
    model = Model(CNN_layers, Deep_layers, dropout=dropout)
    model.apply(init_weights)
    model.to(device)


    # Set loss function and optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Begining training...")
    model_trainer = trainer(model, train_loader, valid_loader, epochs, optimizer, device, loss_fn)
    train_losses, val_losses, train_accs, val_accs = model_trainer.train()

    

    test_accuracy, test_loss = 0, 0
    preds = []
    labels = []
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
            preds.append(test_output_i.argmax(dim=1))
            labels.append(label)

    print("Test loss: {:.4f}".format(test_loss))
    print("Test accuracy: {:.2f}%".format(test_accuracy*100))


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
    
    y_pred_test = np.argmax(model.predict(data.get_data('X_test')), axis=1)
    y_test = np.argmax(data.get_data('y_test'), axis=1)
    #print(y_pred_test, y_test)
    confusion_mat = confusion_matrix(y_test, y_pred_test)
    accuracy = accuracy_score(y_pred_test, y_test)
    print("Accuracy of {:.2f}% ".format(accuracy*100))

    # Confusion matrix
    classes = data.get_classes()
    #print(classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat/864,
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes,
                annot=True,
                fmt='.2%'
                )
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()