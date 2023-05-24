
import torch
import time
import copy

class trainer():
	def __init__(self, model, train_loader, val_loader, test_loader, n_epochs, optimizer, device, loss_function):
		self.model = model
		self.best_model = model
		self.best_acc = 0
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.n_epochs = n_epochs
		self.optimizer = optimizer
		self.device = device
		self.loss_function = loss_function
		
	def getModel(self):
		return self.model
	
	def bestModel(self):
		return self.best_model
		
		
	def eval(self, model):
		### Evaluation
		# Track epoch loss and accuracy
		epoch_valid_accuracy, epoch_valid_loss = 0, 0
		# Switch model to evaluation (affects batch norm and dropout)
		model.eval()
		# Disable gradients
		with torch.no_grad():
			# Iterate through batches
			for data, label in self.val_loader:
				# Move data to the used device
				label = label.type(torch.LongTensor) # move to outside of training loop for efficiency..
				data = data.type(torch.FloatTensor)
				data = data.to(self.device)
				label = label.to(self.device)
				# Forward pass
				valid_output = model(data)
				valid_loss = self.loss_function(valid_output, label)
				# Compute metrics
				acc = ((valid_output.argmax(dim=1) == label).float().mean())
				epoch_valid_accuracy += acc/len(self.val_loader)
				epoch_valid_loss += valid_loss/len(self.val_loader) 
				
		if epoch_valid_accuracy > self.best_acc:
			self.best_model = copy.deepcopy(model)
			
		return epoch_valid_accuracy, epoch_valid_loss
		
	def train(self):
		train_losses, val_losses = [], []
		train_accs, val_accs = [], []
		best_model = ''
		best_val_acc = 0
		# Train for 10 epochs
		for epoch in range(self.n_epochs):
			### Training
			epoch_start_time = time.time()
			# Track epoch loss and accuracy
			epoch_loss, epoch_accuracy = 0, 0
			# Switch model to training (affects batch norm and dropout)
			self.model.train()
			# Iterate through batches
			for i, (data, label) in enumerate(self.train_loader):
				# Reset gradients
				self.optimizer.zero_grad()
				# Move data to the used device
				label = label.type(torch.LongTensor) # move to outside of training loop for efficiency..
				data = data.type(torch.FloatTensor)
				data = data.to(self.device) # data should already be on device
				label = label.to(self.device)
				# Forward pass
				output = self.model(data)
				loss = self.loss_function(output, label)
				# Backward pass
				loss.backward()
				# Adjust weights
				self.optimizer.step()
				# Compute metrics
				acc = ((output.argmax(dim=1) == label).float().mean())
				#print(label, output.argmax(dim=1))
				epoch_accuracy += acc/len(self.train_loader)
				epoch_loss += loss/len(self.train_loader)
			
			train_losses.append(epoch_loss.item())
			train_accs.append(epoch_accuracy.item())
			print('Epoch: {}, train accuracy: {:.2f}%, train loss: {:.4f}   |   {:.2f}s/epoch'.format(epoch+1, epoch_accuracy*100, epoch_loss,time.time()-epoch_start_time))
			
			val_start_time = time.time()
			epoch_valid_accuracy, epoch_valid_loss = self.eval(self.model)
			val_losses.append(epoch_valid_loss.item())
			val_accs.append(epoch_valid_accuracy.item())
			print('Epoch: {}, val accuracy:   {:.2f}%,   val loss: {:.4f}   |   {:.2f}s'.format(epoch+1, epoch_valid_accuracy*100, epoch_valid_loss,time.time()-val_start_time))
			
		#### Finished training
		#compare current model to best model
		best_acc, best_loss = self.eval(self.best_model)
		print("Best model:   accuracy = {:.2f}%,  loss = {:.5f}".format(best_acc, best_loss))
		
		filename = "model-{:.2f}%-acc".format(best_acc*100)
		torch.save(self.model.state_dict(), filename)
		
		
		best_acc, best_loss = self.eval(self.model)
		print("Final model:   accuracy = {:.2f}%,  loss = {:.5f}".format(best_acc, best_loss))
		
	  
		return train_losses, val_losses, train_accs, val_accs
	
	def test(self):
		 ### Evaluation
		# Track epoch loss and accuracy
		test_accuracy, test_loss = 0, 0
		# Switch model to evaluation (affects batch norm and dropout)
		self.model.eval()
		# Disable gradients
		with torch.no_grad():
			# Iterate through batches
			for data, label in self.test_loader:
				# Move data to the used device
				label = label.type(torch.LongTensor) # move to outside of training loop for efficiency..
				data = data.type(torch.FloatTensor)
				data = data.to(self.device)
				label = label.to(self.device)
				# Forward pass
				test_output = self.model(data)
				test_loss = self.loss_function(test_output, label)
				# Compute metrics
				acc = ((test_output.argmax(dim=1) == label).float().mean())
				test_accuracy += acc/len(self.test_loader)
				test_loss += test_loss/len(self.test_loader) 
		
		print('test accuracy:   {:.2f}%,   test loss: {:.4f}'.format(test_accuracy*100, test_loss))
		return test_loss, test_loss
	
