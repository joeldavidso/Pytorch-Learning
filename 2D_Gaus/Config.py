import puma
import numpy as np
import random
import h5py
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# Defines model hyperparameters
hyper_params ={
    "batch_size" : 256,
    "learning_rate" : 1e-3,
    "epochs" : 10,
}

#############################################################
#############################################################
######                                                 ######
######                 Definitions                     ######
######                                                 ######
#############################################################
#############################################################




# Creates dataset class
class Data(Dataset):

	def __init__(self, input_vecs, input_weights, input_labels):
		self.labels = input_labels
		self.vecs = input_vecs
		self.weights = input_weights
		
	def __len__(self):
		return len(self.labels)

	def __getitem__(self,index):
		vec = self.vecs[index]
		label = self.labels[index]
		weight = self.weights[index]
		return vec,weight,label


# Defines the network
class Network(nn.Module):

	def __init__(self):

		super(Network, self).__init__()
		self.operation = nn.Sequential(
                    nn.Linear(2,64),
                    nn.ReLU(),
                    nn.Linear(64,64),
                    nn.ReLU(),
					nn.Linear(64,64),
                    nn.ReLU(),
                    nn.Linear(64,1),
                    nn.Sigmoid()
                )

	def forward(self, input):
		out = self.operation(input)
		return out


# The training loop
def train_loop(dataloader, model, loss_fn, optimizer, scheduler, experiment,epoch):

	# Set modle to training mode (for batch normalization(not needed here)) and get size of data
	size = len(dataloader.dataset)
	num_batches=len(dataloader)
	model.train()
	loss,tot_loss = 0,0
	experiment.log_metric("learning_rate",scheduler.get_last_lr()[0],epoch=epoch)

	# Loop over batches (batch_size determined in dataloader)
	for batch, (vec,weight,label) in enumerate(dataloader):

		# Apply model to batch and calculate loss
		pred = model(vec)
		loss = loss_fn(pred,label)
		tot_loss+=loss.item()
		
		# Backpropagation
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		# Print current progress after every 20000 batches
		if batch%1000 == 0:
			loss, current = loss.item(), batch*hyper_params["batch_size"]+len(label)
			print("loss: "+str(loss)+"   ["+str(current)+"/"+str(size)+"]")
	scheduler.step()

	experiment.log_metric("epoch_train_loss",tot_loss.item()/num_batches,epoch=epoch)


# The validation loop
def val_loop(dataloader, model, loss_fn, experiment, epoch):

	# Set model to evaluation mode (same reasoning as train mode)
	model.eval()
	size=len(dataloader.dataset)
	num_batches=len(dataloader)
	test_loss, correct = 0,0

	# torch.no_grad() allows for evaluation with no gradient calculation (more efficient)
	with torch.no_grad():
		# Loop over all data in test/val dataloader
		for vec,weight,label in dataloader:
			# Calculates accuracy and avg loss for outputting
			pred = model(vec)
			test_loss+=loss_fn(pred,label).item()
			correct+=(torch.round(pred) == label).type(torch.float).sum().item()

	# Normalizes loss and accuracy then prints
	test_loss /= num_batches
	correct /= size
	print("Validation:")
	print("Accuracy: "+str(100*correct)+", Avg Loss: "+str(test_loss))
	experiment.log_metric("epoch_val_loss",test_loss,epoch=epoch)

# Testing/Plotting loop
def test_loop(dataloader, model, loss_fn):

	print("Running Test Loop")

	# Set model to evaluation mode (same reasoning as train mode)
	model.eval()
	test_loss, correct = 0,0

	# Creates arrays for outputting
	G1_out, G2_out = [],[] # G2 is the Target and has label = 1
	G1_in, G2_in = [],[]

	# torch.no_grad() allows for evaluation with no gradient calculation (more efficient)
	with torch.no_grad():
		# Loop over all data in test/val dataloader
		for vecs,weight,label in dataloader:
			# Calculates accuracy and avg loss for outputting
			pred=model(vecs)
			test_loss+=loss_fn(pred,label).item()
			correct+=(torch.round(pred) == label).type(torch.float).sum().item()
			# Turn pytorch tensors to numpy arrays for easier use later in plotting
			label_num = label.numpy()
			pred_num = pred.numpy()

			# Loop over entries in the batch
			for count, vec in enumerate(vecs):
				if label_num[count] == 0:
					G1_out.append(pred_num[count])
					G1_in.append(vec.detach().numpy())
				else:
					G2_out.append(pred_num[count])
					G2_in.append(vec.detach().numpy())

	return np.array(G1_out), np.array(G2_out), np.array(G1_in), np.array(G2_in)


def Guass_func(dimension,means,covariance_matrix,vector):
	
	cov_det = np.linalg.det(covariance_matrix)
	cov_inv = np.linalg.inv(covariance_matrix)
	norm = 1/(np.sqrt(cov_det*((2*np.pi)**dimension)))
	power = (-1/2)*np.matmul((vector-means).T,np.matmul(cov_inv,(vector-means)))
	return norm*np.exp(power)

def analytic_loop(dataloader):
	
	print("Running Analytic Loop")

	# Creates arrays for outputting
	G1_out, G2_out = [],[] # G2 is the Target and has label = 1
	G1_in, G2_in = [],[]

	# Covariance matrix for the 2-D gaussians 
    # (both distributions have the same cov matrix at the moment)
	cov_matrix = np.array([[1,0],
	                       [0,1]])

	# Mean vectors for the two distributions
	mean_1 = np.sqrt(2)/2
	mean_2 = -np.sqrt(2)/2
	means_1 = np.array([mean_1,mean_1])
	means_2 = np.array([mean_2,mean_2])

	# Loop over all data in test/val dataloader
	for vecs,weight,label in dataloader:
		for count,vec in enumerate(vecs):
			G1 = Guass_func(2,means_1,cov_matrix,vec.numpy())
			G2 = Guass_func(2,means_2,cov_matrix,vec.numpy())

			if label.numpy()[count] == 0:
				G1_out.append(G2/(G1+G2))
				G1_in.append(vec.numpy())
			else:
				G2_out.append(G2/(G1+G2))
				G2_in.append(vec.numpy())

	return np.array(G1_out), np.array(G2_out), np.array(G1_in), np.array(G2_in)





