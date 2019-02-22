import math
import numpy as np
import time

def sigmoid(x):
	s = 1/(1+np.exp(-x))
	return s

def sigmoid_derivative(x):
	s = 1/(1+np.exp(-x))
	ds = s*(1-s)
	return ds

def image2vector(image):
	v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2], 1))
	return v

def normalizeRows(x):
	x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
	x = x/x_norm
	return x

def softmax(x):
	x_exp = np.exp(x)
	x_sum = np.sum(x_exp, axis = 1, keepdims = True)
	s = x_exp / x_sum
	return s

# ===================== Building basic functions with numpy =====================
# ===================== 1.1: sigmoid function, np.exp() =====================
x = np.array([1, 2, 3])
# print(np.exp(x))
print("sigmoid(x) = " + str(sigmoid(x)))

# ===================== 1.2: Sigmoid gradient =====================
print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

# ===================== 1.3: Reshaping arrays =====================
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],
       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],
       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print("image2vector(image) = " + str(image2vector(image)))

# ===================== 1.4: Normalizing rows =====================
x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))

# ===================== 1.5: Broadcasting and the softmax function =====================
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))

# ===================== Vectorization =====================
def L1(yhat, y):
	loss = np.sum(abs(yhat-y))
	return loss

def L2(yhat, y):
	loss = np.sum((yhat-y)**2)
	return loss

# ===================== 2.1: Implement the L1 and L2 loss functions =====================
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat, y)))
print("L2 = " + str(L2(yhat, y)))

print("ex2_1.py done...exit...")