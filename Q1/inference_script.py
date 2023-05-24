

from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
from torchvision.models import densenet201, resnet50, ResNet50_Weights
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

import os
import torch
import numpy as np
import glob
import pandas as pd


# Check which device is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))
loss_fn = torch.nn.CrossEntropyLoss()


model = resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 6)
model.load_state_dict(torch.load('model'))
model = model.to(device)
model.eval()


print("Loading data...") 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_resolution=128):
        '''Dataset helper class for images containing buildings (0), forest (1), glacier (2), mountain (3), sea (4), street (5)'''

        # Establish number of inputs for each class to reduce NN bias
        max_count = 0 
        class_counts = {
            'buildings':0,
            'forest':0,
            'glacier':0,
            'mountain':0,
            'sea':0,
            'street':0
        } 

        self.labels = np.empty((0,1))
        self.image_paths = np.empty((0,1))
        class_folder = 'dataset'
        print("Found files for", os.listdir(class_folder))
        
        image_paths = glob.glob(os.path.join(class_folder, '*.jpg'))
        image_paths.sort()
            
        print(dir, "has", len(image_paths), "samples")
                
        self.image_paths = np.append(self.image_paths, image_paths) # ideally np.append/np.concat is not used as it will need to re-allocate memory each call, but this only happens 5 times?..
        
        temp_labels = np.full((len(image_paths), 1), -1)
        self.labels = np.append(self.labels, temp_labels) # similarly..


        for idx, path in enumerate(self.image_paths): # thought was cause for read_iamge bug
            if "\\" in path:
                self.image_paths[idx] = str(path).replace('\\', '/')
                

        self.eval_transforms = eval_transforms = torch.nn.Sequential(
            transforms.Resize(size=(img_resolution,img_resolution)),
        )
        self.pil_transform = transforms.ToPILImage() # This is a strangely efficient normalizer
        self.tensor_transform = transforms.ToTensor() # ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        self.n = len(self.image_paths)

        
    
    def eval_transform(self, img):
        img = self.eval_transforms(img)
        img = self.pil_transform(img)
        img = self.tensor_transform(img)
        return img

        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        """Get one example"""
        label = self.labels[idx]
        img_path = self.image_paths[idx]
        img = read_image(str(img_path), ImageReadMode.RGB).to(device)
        img_transformed = self.eval_transform(img)
        return img_transformed, img_path
            

dataset = Dataset()
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

accuracy, loss = 0, 0

# Switch model to evaluation (affects batch norm and dropout)
model.eval()
# Disable gradients
with torch.no_grad():
    # Iterate through batches
    preds = []
    paths = []
    for data, path in data_loader:
        # Move data to the used device
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        
        # Forward pass
        output = model(data)
        
        preds += output.argmax(dim=1).cpu().numpy().tolist()
        paths += (path[0].split("/")[-1],)
        
df = pd.DataFrame(paths)
df.columns = ['image filename']
df['predicted class'] = preds

df.to_csv("preds.csv", index=None)
        

