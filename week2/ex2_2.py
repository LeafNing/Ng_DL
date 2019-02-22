import numpy as np 
import matplotlib.pyplot as plt 
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# ===================== Overview of the Problem Set =====================
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
plt.show()
print("y = "+str(train_set_y[:,index])+", it's a '"+classes[np.squeeze(train_set_y[:,index])].decode("utf-8")+"' picture.")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig[0].shape[0]
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T 
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

# standardize dataset
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# ===================== General Building the parts of our algorithm =====================
# Helper functions
def sigmoid(x):
	s = 1/(1+np.exp(-x))
	return s
print("sigmoid([0,2]) = " + str(sigmoid(np.array([0,2]))))

# Initializing parameters
def initialize_with_zeros(dim):
	w = np.zeros((dim, 1))
	b = 0
	assert(w.shape == (dim,1))
	assert(isinstance(b,float) or isinstance(b,int))
	return w,b

dim = 2
w, b = initialize_with_zeros(dim)
print("w = " + str(w))
print("b = " + str(b))

# Forward and Backward propagation
def propagate(w, b, X, Y):
	m = X.shape[1]
	A = sigmoid(np.dot(w.T, X)+b)
	cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

	dw = 1/m*np.dot(X, (A-Y).T)
	db = 1/m*np.sum(A-Y)

	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())
	grads = {"dw":dw, "db":db}
	return grads, cost

w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

# Optimization
# update the parameters using gradient descent
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
	costs = []
	for i in range(num_iterations):
		grads,cost = propagate(w, b, X, Y)
		dw = grads["dw"]
		db = grads["db"]

		w = w-learning_rate*dw
		b = b-learning_rate*db

		if i%100==0:
			costs.append(cost)
		if print_cost and i%100==0:
			print("Cost after iteration %i: %f" %(i, cost))

	params = {"w":w, "b":b}
	grads = {"dw":dw, "db":db}
	return params, grads, costs

# Computing predictions
def predict(w, b, X):
	m = X.shape[1]
	Y_pre = np.zeros((1, m))
	w = w.reshape(X.shape[0], 1)

	A = sigmoid(np.dot(w.T, X)+b)

	Y_pre = np.where(A>=0.5, 1.0, 0)
	assert(Y_pre.shape == (1, m))

	return Y_pre

params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("predictions = " + str(predict(params["w"], params["b"], X)))

# ===================== Merge all functions into a model =====================
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
	w, b = initialize_with_zeros(X_train.shape[0])
	parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
	w = parameters["w"]
	b = parameters["b"]
	Y_pre_test = predict(w, b, X_test)
	Y_pre_train = predict(w, b, X_train)

	# Print train/test accuracy
	print("train accuracy : {} %".format(100-np.mean(np.abs(Y_pre_train-Y_train))*100))
	print("test accuracy: {} %".format(100-np.mean(np.abs(Y_pre_test-Y_test))*100))

	d = {"costs": costs,
		"Y_pre_train":Y_pre_train,
		"Y_pre_test":Y_pre_test,
		"w":w,
		"b":b,
		"learning_rate":learning_rate,
		"num_iterations":num_iterations}
	return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# Example of a picture that was wrongly classified
index = 1
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
plt.show()
# print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" 
# 	+ classes[d["Y_pre_test"][0,index]].decode("utf-8") +  "\" picture.")

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

print("ex2_2.py done...exit...")