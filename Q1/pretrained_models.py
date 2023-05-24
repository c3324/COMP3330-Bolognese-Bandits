from torchvision.io import read_image
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights, densenet201, DenseNet201_Weights
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
batch_size = 64
resize_resolution = 128
epochs = 10
learning_rate = 1e-5
val_size = 0.1
weight_decay = 0 #0.01 #L2 regularization


# Initialize model with the best available weights
#weights = ResNet50_QuantizedWeights.DEFAULT
# weights = ResNet50_Weights.IMAGENET1K_V2
# model = resnet50(weights=weights)
# model.fc = torch.nn.Linear(model.fc.in_features, 6)
# model = model.to(device)

# weights = DenseNet201_Weights.DEFAULT
# model = densenet201(weights = weights)
# model.classifier = torch.nn.Linear(model.classifier.in_features, 6)
# model = model.to(device)

weights = VGG16_Weights.DEFAULT
model = vgg16(weights = weights)
model.classifier = torch.nn.Sequential(
	torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
	torch.nn.ReLU(inplace=True),
	torch.nn.Dropout(p=0.5, inplace=False),
	torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
	torch.nn.ReLU(inplace=True),
	torch.nn.Dropout(p=0.5, inplace=False),
	torch.nn.Linear(in_features=4096, out_features=6, bias=True)
)
model = model.to(device)

# Initialize the inference transforms
preprocess = weights.transforms()

# Set loss function and optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()

print("Loading data...") #webcrawlerDataFolder='dataset/web_crawled',
train_data = Dataset("Q1/dataset/seg_train/seg_train", transform_type='train', use_data="first", data_amount=0.85, copy_amount=0, preload=False, img_resolution= resize_resolution, duplicate_smaller_samples=False, shrink_larger_samples=False)
valid_data = Dataset("Q1/dataset/seg_train/seg_train", transform_type='eval', use_data="last", data_amount=0.15, img_resolution= resize_resolution, preload=False)
test_data = Dataset("Q1/dataset/seg_test/seg_test", transform_type='eval', img_resolution= resize_resolution, preload=True)

# Wrap in DataLoader objects with batch size and shuffling preference
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False) # note this is set to test set!
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

resnet_trainer = trainer(model, train_loader, valid_loader, test_loader, epochs, optimizer, device, loss_fn)
train_losses, val_losses, train_accs, val_accs = resnet_trainer.train()

test_losses, test_accs = resnet_trainer.test()

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



