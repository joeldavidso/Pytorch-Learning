import puma
import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py


# Function for creating samples
def gen_sample(name,n_events):

	print("Generating "+str(n_events)+" "+name+"ing events")

	# Covariance matrix for the 2-D gaussians 
    # (both distributions have the same cov matrix at the moment)
	cov_matrix = np.array([[1,0],
	                      [0,1]])

	# Mean vectors for the two distributions
	mean_1 = 1
	mean_2 = -1
	means_1 = np.array([mean_1,mean_1])
	means_2 = np.array([mean_2,mean_2])

	# Generating distributions
	G_1 = np.random.multivariate_normal(means_1,cov_matrix,size=n_events)
	G_2 = np.random.multivariate_normal(means_2,cov_matrix,size=n_events)

	# Makes label tensors
	labels_1 = np.zeros(shape=n_events)
	labels_2 = np.ones(shape=n_events)

    # Create weight array
	weights_1 = np.ones(shape=n_events)
	weights_2 = np.ones(shape=n_events)

	# Concatinate inputs
	gauss_x = np.concatenate((G_1[:,0],G_2[:,0]),axis=0)
	gauss_y = np.concatenate((G_1[:,1],G_2[:,1]),axis=0)
	labelss = np.concatenate((labels_1,labels_2),axis=0)
	weights = np.concatenate((weights_1,weights_2),axis=0)

    # Plotting the x distributions
	x_plot = puma.HistogramPlot(xlabel = "X",ylabel = "Normalised number of events")
	x_plot.add(puma.Histogram(G_1[:,0],weights=weights_1,label="Gaussian 1"))
	x_plot.add(puma.Histogram(G_2[:,0],weights=weights_2,label="Gaussian 2"))

    # Plotting the y distributions
	y_plot = puma.HistogramPlot(xlabel = "Y",ylabel = "Normalised number of events")
	y_plot.add(puma.Histogram(G_1[:,1],weights=weights_1,label="Gaussian 1"))
	y_plot.add(puma.Histogram(G_2[:,1],weights=weights_2,label="Gaussian 2"))

    # Drawing/saving the x plot
	x_plot.draw()
	x_plot.savefig("plots/"+name+"_x.png")

    # Drawing/saving the y plot
	y_plot.draw()
	y_plot.savefig("plots/"+name+"_y.png")

	# Bins for 2D plots
	bins = [np.linspace(-4,4,50),np.linspace(-4,4,50)]

	# Create h5 file 
	file = h5py.File("samples/"+name+".h5","w")

	# Create datatype for the h5file
	datatype = \
	np.dtype \
	( list \
		( { "x" : np.float32
	    , "y" : np.float32
		, "weight" : np.float32
		, "label" : np.float32
		}.items()
		)
	)

	# Creates data dictionary to put into h5 file
	data_dict = np.recarray(gauss_x.shape, dtype=datatype)
	data_dict["x"] = gauss_x
	data_dict["y"] = gauss_y
	data_dict["weight"] = weights
	data_dict["label"] = labelss

	file.create_dataset("Gaus", data = data_dict)


	file.close()

	# Print Done
	print("Done")


#############################################################
#############################################################
######                                                 ######
######                   Main Body                     ######
######                                                 ######
#############################################################
#############################################################

# Define key parameters and network hyperparameters
n_train = 2000000
n_test = int(n_train/10)

gen_sample("train", n_train)
gen_sample("test", n_test)

print("Samples generated!!")