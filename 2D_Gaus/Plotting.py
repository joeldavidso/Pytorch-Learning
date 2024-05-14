import puma
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from Config import hyper_params, test_loop, Network, Data, analytic_loop

#############################################################
#############################################################
######                                                 ######
######                   Main Body                     ######
######                                                 ######
#############################################################
#############################################################

# Downloads trained model
model = torch.load("trained_Networks/"+"model"+".pth")

# Initialize loss function
loss_function = nn.BCELoss()

# Imports training and testing samples
test_file = h5py.File("samples/test.h5","r")

# Converst h5 file contents to datasets
test_data = Data(torch.stack((torch.from_numpy(test_file["Gaus"]["x"]),
							   torch.from_numpy(test_file["Gaus"]["y"])),-1),
				  torch.from_numpy(test_file["Gaus"]["weight"]).unsqueeze(1),
				  torch.from_numpy(test_file["Gaus"]["label"]).unsqueeze(1))

# Closes h5 file
test_file.close()

# Conversts datsets to datlaoaders for training
test_dataloader = DataLoader(test_data, batch_size=hyper_params["batch_size"], shuffle=True)

# Runs over validation/test dataset to generate arrays of the network inputs and outputs
G1_output, G2_output, G1_input, G2_input = test_loop(test_dataloader,model,loss_function)
G1_output_an, G2_output_an, G1_input_an, G2_input_an = analytic_loop(test_dataloader)

print("Plotting")

# Plots the network output
output_plot = puma.HistogramPlot(xlabel = "Network Output", ylabel = "Normalised number of events")
output_plot.add(puma.Histogram(G1_output.flatten(),label = "Gauss_1"))
output_plot.add(puma.Histogram(G2_output.flatten(),label = "Gauss_2"))
output_plot.add(puma.Histogram(G1_output_an,label = "Gauss_1_analytic",linestyle="dashed"))
output_plot.add(puma.Histogram(G2_output_an,label = "Gauss_2_analytic",linestyle="dashed"))
output_plot.draw()
output_plot.savefig("plots/output.png")

# Creates reweighting array
G1_RW = np.ndarray(shape = G1_output.shape)
G2_RW = np.ndarray(shape = G2_output.shape)

for count,out in enumerate(G1_output):
    G1_RW[count] = out/(1-out)
for count,out in enumerate(G2_output):
    G2_RW[count] = out/(1-out)
    
G1_RW_an = np.ndarray(shape = G1_output_an.shape)
G2_RW_an = np.ndarray(shape = G2_output_an.shape)

for count,out in enumerate(G1_output_an):
    G1_RW_an[count] = out/(1-out)
for count,out in enumerate(G2_output_an):
    G2_RW_an[count] = out/(1-out)

# Bins for 2D plots
bins = [np.linspace(-5,5,50),np.linspace(-5,5,50)]


# RW plot for distribution 1
plt.hist2d(G1_input[:,0],
			G1_input[:,1],
            weights = G1_RW[:,0],
			bins = bins,
			cmap = plt.cm.jet,
            density = True)
cb = plt.colorbar()
cb.set_label("Normalised number of events")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Dist 1 Reweighted to Dist 2 Via the Network")
plt.savefig("plots/RW_1.png")
plt.clf()

plt.hist2d(G1_input_an[:,0],
			G1_input_an[:,1],
            weights = G1_RW_an,
			bins = bins,
			cmap = plt.cm.jet,
            density = True)
cb = plt.colorbar()
cb.set_label("Normalised number of events")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Dist 1 Reweighted to Dist 2 Via the Analytic Function")
plt.savefig("plots/RW_1_an.png")



print("Done")