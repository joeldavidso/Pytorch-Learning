import numpy as np
import h5py
import os
import glob
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from Config import Data, Network, hyper_params, train_loop, val_loop

#############################################################
#############################################################
######                                                 ######
######                   Main Body                     ######
######                                                 ######
#############################################################
#############################################################

# Defines te comet_ml experiment
experiment = Experiment(
  api_key="9uQNNIPVWsJ7kcG2nnma73Q91",
  project_name="gaus-disc-2d",
  workspace="salt-joeldavidso"
)

# Logs hyperparams
experiment.log_parameters(hyper_params)

# Imports training and testing samples
train_file = h5py.File("samples/train.h5","r")
test_file = h5py.File("samples/test.h5","r")

# Converst h5 file contents to datasets
train_data = Data(torch.stack((torch.from_numpy(train_file["Gaus"]["x"]),
							   torch.from_numpy(train_file["Gaus"]["y"])),-1),
				  torch.from_numpy(train_file["Gaus"]["weight"]).unsqueeze(1),
				  torch.from_numpy(train_file["Gaus"]["label"]).unsqueeze(1))

test_data = Data(torch.stack((torch.from_numpy(test_file["Gaus"]["x"]),
							   torch.from_numpy(test_file["Gaus"]["y"])),-1),
				  torch.from_numpy(test_file["Gaus"]["weight"]).unsqueeze(1),
				  torch.from_numpy(test_file["Gaus"]["label"]).unsqueeze(1))

# Closes h5 files
test_file.close()
train_file.close()

# Conversts datsets to datlaoaders for training
train_dataloader = DataLoader(train_data, batch_size=hyper_params["batch_size"], shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=hyper_params["batch_size"], shuffle=True)

# Outputs the structure of the data
train_vecs, train_weights, train_labels = next(iter(train_dataloader))
print(f"Vector batch shape: {train_vecs.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# Get training device
device = ("cuda" if torch.cuda.is_available() else "cpu")
print("Training device set to "+device)

# Create instance of Network and move to device
model = Network().to(device)
print("Model created with structure:")
print(model)

# Initialize loss function
loss_function = nn.BCELoss()

# Initialize the optimizer with Stockastic Gradient Descent function
optimizer = torch.optim.SGD(model.parameters(), lr=hyper_params["learning_rate"])

# Initialize the learning rate schduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)

# Loop over epochs to train and validate
for e in range(hyper_params["epochs"]):

	print("__________________________________")
	print("Epoch "+str(e+1)+" :")
	print("----------------------------------")
	print("Training:")
	train_loop(train_dataloader, model, loss_function, optimizer,scheduler,experiment,e+1)
	val_loop(test_dataloader, model, loss_function, experiment, e+1)

log_model(experiment,model=model,model_name="TheModel")

torch.save(model,"trained_Networks/model.pth")

print("Training Done!")
