import puma
import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py


# Function for creating samples
def gen_sample(name,n_events):

	print("Generating "+str(n_events)+" "+name+"ing events")

	# Covariance matrix for the 3-D gaussians 
    # (both distributions have the same cov matrix at the moment)
	cov_matrix = np.array([[1,0,0],
	                       [0,1,0],
						   [0,0,1]])

	# Mean vectors for the two distributions
	mean_1 = np.sqrt(2)/2
	mean_2 = -np.sqrt(2)/2
	means_1 = np.array([mean_1,mean_1,mean_1])
	means_2 = np.array([mean_2,mean_2,mean_2])

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
	gauss_z = np.concatenate((G_1[:,2],G_2[:,2]),axis=0)
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

    # Plotting the x distributions
	z_plot = puma.HistogramPlot(xlabel = "Z",ylabel = "Normalised number of events")
	z_plot.add(puma.Histogram(G_1[:,2],weights=weights_1,label="Gaussian 1"))
	z_plot.add(puma.Histogram(G_2[:,2],weights=weights_2,label="Gaussian 2"))

    # Drawing/saving the x plot
	x_plot.draw()
	x_plot.savefig("plots/"+name+"_x.png")

    # Drawing/saving the y plot
	y_plot.draw()
	y_plot.savefig("plots/"+name+"_y.png")

    # Drawing/saving the y plot
	z_plot.draw()
	z_plot.savefig("plots/"+name+"_z.png")

	# Bins for 2D plots
	bins = [np.linspace(-4,4,50),np.linspace(-4,4,50)]

	# Rotation matrix function
	def Rotation(theta):
		theta = np.radians(theta)
		matrix = np.array([[np.cos(theta),-np.sin(theta)],
					      [np.sin(theta),np.cos(theta)]])
		return matrix

	# transform of input data via 2D matrix function
	def transform(matrix,xs,ys):
		arr = np.column_stack((xs,ys))
		arr2 = np.ndarray(shape=(len(xs),2))
		for count, vector in enumerate(arr):
			arr2[count]=np.matmul(matrix,vector)
		out_xs = arr2[:,0]
		out_ys = arr2[:,1]
		return out_xs, out_ys

	#rotation matricies
	rot_45_clockwise = Rotation(-45)
	rot_45_anticlockwise = Rotation(45)
	
	# Control and signal region corners for plotting
	half_region_width = 1
	
	CR1x = [-np.sqrt(2),-np.sqrt(2)/3,-np.sqrt(2)/3,-np.sqrt(2),-np.sqrt(2)]
	CRy = [half_region_width,half_region_width,-half_region_width,-half_region_width,half_region_width]
	CR1x,CR1y = transform(rot_45_anticlockwise,CR1x,CRy)

	CR2x = [np.sqrt(2),np.sqrt(2)/3,np.sqrt(2)/3,np.sqrt(2),np.sqrt(2)]
	CR2x,CR2y = transform(rot_45_anticlockwise,CR2x,CRy)

	SRx = [np.sqrt(2)/3,-np.sqrt(2)/3,-np.sqrt(2)/3,np.sqrt(2)/3,np.sqrt(2)/3]
	SRx,SRy = transform(rot_45_anticlockwise,SRx,CRy)

	# Plot for distribution 1
	plt.hist2d(G_1[:,0],
			   G_1[:,2],
			   bins = bins,
			   cmap = plt.cm.jet,
			   density=True)
	cb = plt.colorbar()
	cb.set_label("Normalised number of events")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title("Distribution 1 "+name+" Sample")
	plt.plot(CR1x,CR1y,color="black",linewidth=2)
	plt.plot(CR2x,CR2y,color="black",linewidth = 2)
	plt.plot(SRx,SRy,color="white",linewidth = 2,linestyle = "dashed")
	plt.text(np.mean(CR1x[:-1])-0.3,np.mean(CR1y[:-1])-0.1,"CR1",color="black")
	plt.text(np.mean(CR2x[:-1])-0.3,np.mean(CR2y[:-1])-0.1,"CR2",color="black")
	plt.text(np.mean(SRx[:-1])-0.2,np.mean(SRy[:-1])-0.1,"SR",color="black")
	plt.show()
	plt.savefig("plots/"+name+"_2D_1.png")
	plt.clf()

	# Plot for distribution 2
	plt.hist2d(G_2[:,0],
			   G_2[:,1],
			   bins = bins,
			   cmap = plt.cm.jet,
			   density=True)
	cb = plt.colorbar()
	cb.set_label("Normalised number of events")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title("Distribution 2 "+name+" Sample")
	plt.plot(CR1x,CR1y,color="black",linewidth=2)
	plt.plot(CR2x,CR2y,color="black",linewidth = 2)
	plt.plot(SRx,SRy,color="white",linewidth = 2,linestyle = "dashed")
	plt.text(np.mean(CR1x[:-1])-0.3,np.mean(CR1y[:-1])-0.1,"CR1",color="black")
	plt.text(np.mean(CR2x[:-1])-0.3,np.mean(CR2y[:-1])-0.1,"CR2",color="black")
	plt.text(np.mean(SRx[:-1])-0.2,np.mean(SRy[:-1])-0.1,"SR",color="black")
	plt.show()
	plt.savefig("plots/"+name+"_2D_2.png")
	plt.clf()

	# Create h5 file 
	file = h5py.File("samples/"+name+".h5","w")


	# determines the region labels for each datpoint
	region_labelss = np.ndarray(shape=gauss_x.shape) # Outside all = 0, CR1 = 1, CR2 = 2, SR = 3

	vectors = np.column_stack((gauss_x,gauss_y))
	for count,vec in enumerate(vectors):
		vec_prime = np.matmul(rot_45_clockwise,vec)
		vec_x = vec_prime[0]
		vec_y = vec_prime[1]

		if vec_y <= 1 and vec_y >= -1:
			if vec_x <= -np.sqrt(2)/3 and vec_x >= -np.sqrt(2):
				region_labelss[count]=1

			elif vec_x <= np.sqrt(2) and vec_x >= np.sqrt(2)/3:
				region_labelss[count]=2
			else:
				region_labelss[count]=3
		else:
			region_labelss[count]=0


	# Create datatype for the h5file
	datatype = \
	np.dtype \
	( list \
		( { "x" : np.float32
	    , "y" : np.float32
		, "z" : np.float32
		, "weight" : np.float32
		, "label" : np.float32
		, "region" : np.float32
		}.items()
		)
	)

	# Creates data dictionary to put into h5 file
	data_dict = np.recarray(gauss_x.shape, dtype=datatype)
	data_dict["x"] = gauss_x
	data_dict["y"] = gauss_y
	data_dict["z"] = gauss_z	
	data_dict["weight"] = weights
	data_dict["label"] = labelss
	data_dict["region"] = region_labelss

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