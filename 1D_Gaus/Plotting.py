import puma
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from Config import hyper_params, test_loop, Network, Data, analytic_loop, CR

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

if not CR:
    # Converst h5 file contents to datasets
    test_data = Data(torch.from_numpy(test_file["Gaus"]["x"]).unsqueeze(-1),
                    torch.from_numpy(test_file["Gaus"]["weight"]).unsqueeze(1),
                    torch.from_numpy(test_file["Gaus"]["label"]).unsqueeze(1))

    # Pandas datframe for seaborn plotting
    dataframe = pd.DataFrame(
        {
            "x":test_file["Gaus"]["x"],
            "y":test_file["Gaus"]["y"],
            "label":np.where(test_file["Gaus"]["label"]==1,"Gauss_2","Gauss_1")
        }
    )

else:
    # Converted h5 for 2b->4b reweighting
    test_data = Data(torch.from_numpy(test_file["Gaus"]["x"][test_file["Gaus"]["y"]<0]).unsqueeze(-1),
    				 torch.from_numpy(test_file["Gaus"]["weight"][test_file["Gaus"]["y"]<0]).unsqueeze(1),
    				 torch.from_numpy(test_file["Gaus"]["label"][test_file["Gaus"]["y"]<0]).unsqueeze(1))

    # Pandas datframe for seaborn plotting
    dataframe = pd.DataFrame(
        {
            "x":test_file["Gaus"]["x"][test_file["Gaus"]["y"]<0],
            "y":test_file["Gaus"]["y"][test_file["Gaus"]["y"]<0],
            "label":np.where(test_file["Gaus"]["label"][test_file["Gaus"]["y"]<0]==1,"Gauss_2","Gauss_1"),
            "binary_label":test_file["Gaus"]["label"][test_file["Gaus"]["y"]<0]
        }
    )

# Gets total number of each distribution in the whle test sample
N1 = np.sum(test_file["Gaus"]["label"]==0)
N2 = np.sum(test_file["Gaus"]["label"]==1)

# Closes h5 file
test_file.close()

# Calculates the fractions in the Train sample assuming no change in class ratios
n1 = np.sum(dataframe["binary_label"]==0)
n2 = np.sum(dataframe["binary_label"]==1)

if N1==n1 and N2==n2:
    f_1 = 0.5
    f_2 = 0.5
else:
    f_1 = (N1-n1)/((N1-n1)+(N2-n2))
    f_2 = 1-f_1

# Calculates the fractions in the Test sample
F_1 = np.sum(dataframe["label"] == 0)/dataframe["label"].size
F_2 = np.sum(dataframe["label"] == 1)/dataframe["label"].size

# Conversts datsets to datlaoaders for training
test_dataloader = DataLoader(test_data, batch_size=hyper_params["batch_size"], shuffle=False)

# Runs over validation/test dataset to generate arrays of the network inputs and outputs
output = test_loop(test_dataloader,model,loss_function)
an_output = analytic_loop(test_dataloader,f_1,f_2)

# Duplicate dataframe to allow for analytic plotting in seaborn
temp_labels = dataframe["label"]
dataframe = pd.concat([dataframe,dataframe])

dataframe["output"] = np.append(output.flatten(),an_output.flatten())
dataframe["disc"] = np.append(np.log(output.flatten()/(np.ones_like(output.flatten())-output.flatten())),
                              np.log(an_output.flatten()/(np.ones_like(an_output.flatten())-an_output.flatten())))
dataframe["label"] = np.append(temp_labels,temp_labels+"_an")

print("Plotting")

sns.set_theme(style="ticks")
sns.color_palette("dark")

# X plot
sns.displot(dataframe[(dataframe["label"]=="Gauss_1") | (dataframe["label"]=="Gauss_2")],
            x="x",hue="label",element="step",bins=30)
plt.savefig("plots/X_plot.png")

# Y plot
sns.displot(dataframe[(dataframe["label"]=="Gauss_1") | (dataframe["label"]=="Gauss_2")],
            x="y",hue="label",element="step",bins=30)
plt.savefig("plots/Y_plot.png")
plt.clf()

# Output Plot
out_plot=sns.displot(dataframe,x="output",hue="label",element="step",bins=30,fill = False)
out_plot.set(xlim=(0,1))
plt.savefig("plots/output.png")
plt.clf()

# Disc Plot
disc_plot=sns.displot(dataframe,x="disc",hue="label",element="step",bins=30)
plt.savefig("plots/disc.png")
plt.clf()

# # Plots the network output
# output_plot = puma.HistogramPlot(xlabel = "Network Output", ylabel = "Number of events",norm=False)
# output_plot.add(puma.Histogram(G1_output.flatten(),label = "Gauss_1"))
# output_plot.add(puma.Histogram(G2_output.flatten(),label = "Gauss_2"))
# output_plot.add(puma.Histogram(G1_output_an.flatten(),label = "Gauss_1_analytic"
#                                ,linestyle="dashed"))
# output_plot.add(puma.Histogram(G2_output_an.flatten(),label = "Gauss_2_analytic"
#                                ,linestyle="dashed"))
# output_plot.draw()
# output_plot.savefig("plots/output.png")

# # Plots network discriminant(log(p2/p1))
# disc_plot = puma.HistogramPlot(xlabel="Discriminant",ylabel="Number of events",norm=False)
# disc_plot.add(puma.Histogram(np.log(np.divide(G1_output.flatten(),1-G1_output.flatten())),label="G1"))
# disc_plot.add(puma.Histogram(np.log(np.divide(G2_output.flatten(),1-G2_output.flatten())),label="G2"))
# disc_plot.add(puma.Histogram(np.log(np.divide(G1_output_an.flatten(),1-G1_output_an.flatten()))
#                              ,label="G1_an",linestyle="dashed"))
# disc_plot.add(puma.Histogram(np.log(np.divide(G2_output_an.flatten(),(1-G2_output_an.flatten())))
#                              ,label="G2_an",linestyle="dashed"))
# disc_plot.draw()
# disc_plot.savefig("plots/disc.png")


# # Creates reweighting array
# G1_RW = np.ndarray(shape = G1_output.shape)
# G2_RW = np.ndarray(shape = G2_output.shape)

# for count,out in enumerate(G1_output):
#     G1_RW[count] = F_2*f_1*out/(F_1*f_2*(1-out))
    
# G1_RW_an = np.ndarray(shape = G1_output_an.shape)
# G2_RW_an = np.ndarray(shape = G2_output_an.shape)

# for count,out in enumerate(G1_output_an):
#     G1_RW_an[count] = F_2*f_1*out/(F_1*f_2*(1-out))

# # RW plot for distribution 1
# RW_plot = puma.HistogramPlot(xlabel="X",ylabel="Number of Events",norm=False)
# RW_plot.add(puma.Histogram(arr_x[arr_label==1].flatten(),label="Gaussian_2"))
# RW_plot.add(puma.Histogram(arr_x[arr_label==0],weights=G1_RW.flatten(),label="Gaussian_1_reweighted"))
# RW_plot.add(puma.Histogram(arr_x[arr_label==0],weights=G1_RW_an.flatten(),label="Gaussian_1_reweighted_analytic"))
# RW_plot.draw()
# RW_plot.savefig("plots/RW.png")
# print("Done")