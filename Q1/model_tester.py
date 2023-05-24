
from torchvision.io import read_image
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
from torchvision.models import resnet50, ResNet50_Weights
import torch
import numpy as np

from Q1 import Dataset
from cm_analysis import cm_analysis


### Parameters ###########################
batch_size = 128
resize_resolution = 128

# Check which device is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))
loss_fn = torch.nn.CrossEntropyLoss()




model = resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 6)
model.load_state_dict(torch.load('best-model'))
model = model.to(device)
model.eval()



print("Loading data...") #webcrawlerDataFolder='dataset/web_crawled',
#train_data = Dataset("dataset/seg_train", transform_type='train', copy_amount=0, preload=False, img_resolution= resize_resolution, duplicate_smaller_samples=False, shrink_larger_samples=False)
#valid_data = Dataset("dataset/seg_train", webcrawlerDataFolder='dataset/web_crawled', transform=eval_transforms, use_data='last', data_amount=val_size, img_resolution= resize_resolution, preload=True, shrink_larger_samples=True)
test_data = Dataset("dataset/seg_test/seg_test", transform_type='eval', img_resolution= resize_resolution, preload=True)

# Wrap in DataLoader objects with batch size and shuffling preference
#train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
#valid_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False) # note this is set to test set!
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)



accuracy, loss = 0, 0

# Switch model to evaluation (affects batch norm and dropout)
#model.eval()
# Disable gradients
with torch.no_grad():
	# Iterate through batches
	preds = []
	y_test = []
	for data, label in test_loader:
		# Move data to the used device
		label = label.type(torch.LongTensor) # move to outside of training loop for efficiency..
		data = data.type(torch.FloatTensor)
		data = data.to(device)
		label = label.to(device)
		# Forward pass
		output = model(data)
		batch_loss = loss_fn(output, label)
		# Compute metrics
		acc = ((output.argmax(dim=1) == label).float().mean())
		accuracy += acc/len(test_loader)
		loss += batch_loss/len(test_loader) 
		
		preds += output.argmax(dim=1).cpu().numpy().tolist()
		y_test += label.cpu().numpy().tolist()
		print(label)

#print(y_test)
y_test = np.array(y_test).flatten()
#print(y_test)

#preds = data_loader.onehot(np.array(preds).reshape(-1,1)).toarray()
#preds = np.array(preds, dtype=np.int32).flatten()
#print(preds, y_test, type(preds[0]), type(y_test[0]))

# normalize
#cm_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

print("Accuracy of {:.2f}% ".format(accuracy*100))

# Confusion matrix
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

cm_analysis(y_test, preds, classes)