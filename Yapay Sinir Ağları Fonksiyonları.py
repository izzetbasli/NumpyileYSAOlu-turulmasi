# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 22:00:41 2019

@author: G
"""

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:100, :] 
y = iris.target[0:100]

#####   Aktivasyon Fonksiyonlarının tanıtılması

def hard_limiter(x):
    y = np.multiply(np.greater(x,0),1)
    return y

def signum(x):
    y = np.sign(x)
    return y

def piece_sigmoid(x):
    b = np.zeros(x.shape)
    b[np.nonzero(x>0.5)] = 1
    b[np.nonzero(x<-0.5)] = 0
    b[np.nonzero(np.logical_and((x<0.5), (x>-0.5)))] = a[np.nonzero(np.logical_and((x<0.5), (x>-0.5)))] + 0.5

    return b

def bip_sigmoid(x):  #bipolar piece linear sigmoid
    b = np.zeros(x.shape)
    b[np.nonzero(x>=1)] = 1
    b[np.nonzero(x<=-1)] = -1
    b[np.nonzero(np.logical_and((x<1), (x>-1)))] = a[np.nonzero(np.logical_and((x<1), (x>-1)))]

    return b



def sigmoid(x, alpha = 1.0):
    return np.divide(1, (1 + np.exp(- alpha * x)))

def b_sigmoid(x, alpha = 1.0):
    return np.divide(2, (1 + np.exp(-alpha * x))) -1

def tanh(x, alpha = 1.0):
    return np.divide((1 - np.exp(-alpha * x)), (1 + np.exp(-alpha * x)))

import matplotlib.pyplot as plt

a = np.linspace(-7,7,1000)
plt.plot(a,hard_limiter(a))
plt.grid()

plt.plot(a,signum(a))
plt.grid()


plt.plot(a,piece_sigmoid(a))
plt.grid()

plt.plot(a,bip_sigmoid(a))
plt.grid()

plt.plot(a,sigmoid(a))
plt.grid()

plt.plot(a,b_sigmoid(a))
plt.grid()

plt.plot(a,tanh(a))
plt.grid()


######-------------------EXOR  Örneği---------------

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

W0 = np.array([[-1,1,0.5],[1,-1,0.5]])

## Giriş matrisine treshold değerinin eklenmesi
X = np.c_[X, -1 * np.ones(X.shape[0])]

o0 = X.dot(W0.T[:,0])
o1 = X.dot(W0.T[:,1])

output0 = hard_limiter(o0)
output1 = hard_limiter(o1)

W1 = np.array([1,1,0.5])

Out = np.c_[output0, output1]
Out = np.c_[Out, -1 * np.ones(Out.shape[0])]

y_pred = Out.dot(W1.T)
y_pred = hard_limiter(y_pred)


####---------- MultiLayer Perceptron Örneği-------------

W1_0 = np.array([[1,0,1],[-1,0,-2],[0,1,0],[0,-1,-3]])
W1_1 = np.array([1,1,1,1,3.5])

x1 = 5*(np.random.rand(30000)-0.2)
x2 = 5*(np.random.rand(30000)-0.2)

X1 = np.c_[x1,x2]
X1 = np.c_[X1, -1 * np.ones(X1.shape[0])]

output2 = signum(X1.dot(W1_0.T))
output2 = np.c_[output2, -1 * np.ones(output2.shape[0])]

y_pred1 = signum(output2.dot(W1_1.T))

plt.scatter(x1,x2, c=y_pred1)

####------------Multilayer Perceptron ödevi-----------

W2_0 = np.array([[2,-1,-2],[-1,-1,-2],[-1.5,1,-3],[3,1,-3]])
W2_1 = np.array([1,1,1,1,3.5])

x1 = 5*(np.random.rand(30000)-0.5)
x2 = 5*(np.random.rand(30000)-0.5)

X2 = np.c_[x1,x2]
X2 = np.c_[X2, -1 * np.ones(X2.shape[0])]

output3 = signum(X2.dot(W2_0.T))
output3 = np.c_[output3, -1 * np.ones(output3.shape[0])]

y_pred2 = signum(output3.dot(W2_1.T))

plt.scatter(x1,x2, c=y_pred2)







### İki vector veya matris arasındaki euclidean norm
def euclidean(x):
    return np.sum(np.square(x)) / len(x)

def mean_squared_error(x,y):
    if np.isscalar(x):
        return np.square(x - y)
    else:    
        return np.sum(np.square(x - y)) / len(x)


a1 = np.array([2,1,5,2,7,3])
b1 = np.array([3,6,2,9,4,1])

print(euclidean(a1))

print(mean_squared_error(a1,b1))


### Batch Learning
    #####    w = Argmin (1/L) * sum(H(x,w) - d)
    
### Gradient descent algoritm(Batch Mode)--- Stochastic Gradient Descent(SGD)(Pattern Mode)

def activation(w, input_signal, func = 'signum'):
    if func == 'signum':
        output = signum(input_signal.dot(w.T))
    if func == 'sigmoid':
        output = sigmoid(input_signal.dot(w.T))
    if func == 'hard_limiter':
        output = hard_limiter(input_signal.dot(w.T))
    if func == 'tanh':
        output = tanh(input_signal.dot(w.T))
    if func == 'b_sigmoid':
        output = b_sigmoid(input_signal.dot(w.T))
    if func == 'piece_sigmoid':
        output = piece_sigmoid(input_signal.dot(w.T))
    return output

def activation_diff(w, input_signal, func = 'sigmoid'):
    if func == 'sigmoid':
        output = sigmoid(input_signal.dot(w.T)) * (1 - sigmoid(input_signal.dot(w.T)))
    if func == 'tanh':
        output = 0.5 * (1 - tanh(input_signal.dot(w.T))**2)
    if func == 'signum' or func == 'hard_limiter':
        output = 0
    if func == 'piece_sigmoid' or func == 'b_sigmoid':
        output = np.zeros(input_signal.dot(w.T).shape)
        output[np.nonzero(np.logical_and((output<1), (output>-1)))] = 1
    return output
    
### Learning Rules
    ### w(k+1) = w(k) + lr * learning_signal * input_signal
    
        # Hebbian Learning Rule --> learning_signal = output_signal
        
def hebbian(w, lr, input_signal, fonk = 'signum'):
    w = w + lr * activation(w,input_signal, fonk) * input_signal
    return w

        # Perceptron Learning Rule --> learning_signal = desired - output
    
def perceptron(w, lr, desired, input_signal, func = 'signum'):
    w = w + lr * (desired - activation(w, input_signal, func)) * input_signal
    return w

        # Delta Learning Rule --> learning_signal = (desired - output) * (türev(output))

def delta(w, lr, desired, input_signal, func = 'sigmoid'):
    w = w + lr * (desired - activation(w,input_signal,func)) * activation_diff(w,input_signal,func) * input_signal
    return w

        # Widrow-Hoff Learning Rule --> learning_signal = desired - output
                                    ### Output = f(wTx)  ---> linear activation function
        # ADALINE --> Adaptive Linear Element if activation function is f(v) = v
        # called LMS(Least Mean Squared ) learning rule
def widrow_hoff(w, lr, desired, input_signal):
    w = w + lr * (desired - input_signal.dot(w.T)) * input_signal
    return w

        # Correlation Learning Rule --> learning_signal = desired
        # Usually Applied to discrete perceptron
def correl_learn(w, lr, desired, input_signal):
    w = w + lr * desired * input_signal
        
### Hebbian learning rule example

w = np.array([1,-1,0,0.5])
x = np.array([[1,-2,1.5,0],[1,-0.5,-2,-1.5],[0,1,-1,1.5]])

w1 = hebbian(w, 1, x[0,:])
w2 = hebbian(w1, 1, x[1,:])
w3 = hebbian(w2, 1, x[2, :]) 

w1_1 = hebbian(w, 1, x[0,:], 'b_sigmoid')
w1_2 = hebbian(w1_1, 1, x[1,:], 'b_sigmoid')
w1_3 = hebbian(w1_2, 1, x[2,:], 'b_sigmoid')

### Perceptron Leraning rule example
lr = 0.1
w = np.array([1,-1,0,0.5])
x = np.array([[1,-2,0,-1],[0,1.5,-0.5,-1],[-1,1,0.5,-1]])
d = np.array([[-1],[-1],[1]])

w1 = perceptron(w, lr, d[0], x[0,:])
w2 = perceptron(w1, lr, d[1], x[1,:])
w3 = perceptron(w2, lr, d[2], x[2,:])


### Delta Learning Rule example
lr = 0.1
w = np.array([1,-1,0,0.5])
x = np.array([[1,-2,0,-1],[0,1.5,-0.5,-1],[-1,1,0.5,-1]])
d = np.array([[-1],[-1],[1]])

w1_d1 = delta(w, lr, d[0], x[0,:], 'tanh')
w1_d2 = delta(w1_d1, lr, d[1], x[1,:], 'tanh')
w1_d3 = delta(w1_d2, lr, d[2], x[2,:], 'tanh')



####---------Multilayer Perceptron İris flower dataset--------
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:100, :] 
y = iris.target[0:100]
func = 'sigmoid'
if func == 'tanh':
    y = 2 * y - 1 


X1 = np.c_[X, -np.ones(len(X))]
x_train, x_test, y_train, y_test = train_test_split(X1,y,test_size = 0.3)


W = np.random.rand(1,x_train.shape[1])
lr = 0.1
loss_his = []
w_his = []
w_his.append(W)
for i in range(0, x_train.shape[0]):
    mse = (y_train[i] - activation(W, x_train[i,:], func))**2
    loss_his.append(mse)
    W = delta(W, lr, y_train[i], x_train[i,:], func)
    w_his.append(W)
    
    
plt.plot(loss_his)
plt.xlabel('Iteration number')
plt.ylabel('MSE')
plt.grid()

w_his = np.array(w_his)
w1 = w_his[:,0,0]
w2 = w_his[:,0,1]
w3 = w_his[:,0,2]
w4 = w_his[:,0,3]
w5 = w_his[:,0,4]

plt.plot(w1)
plt.plot(w2)
plt.plot(w3)
plt.plot(w4)
plt.plot(w5)
plt.legend(['w1','w2','w3','w4','w5'])


####
digit = datasets.load_digits()
#digit.data.shape
images = digit['images']

plt.gray()
plt.matshow(images[18])
plt.show()

X = digit['data']
y = digit['target']   #  desired signal

### ANN yapısı 64-100-40-10

X1 = np.c_[X, -np.ones(len(X))]
y1 = np.zeros((len(y),10))
for i in range(0, len(y)):
    for k in range(0, 10):
        if k == y[i]:
            y1[i,k] = 1
            

x_train, x_test, y_train, y_test = train_test_split(X1 ,y1 ,test_size=0.3)

W1 = (2*np.random.rand(25,x_train.shape[1])-1)/1
W2 = (2*np.random.rand(10,26)-1)/1
W3 = (2*np.random.rand(1,11)-1)/1


#t1 = x_train[0:1,:].dot(W1.T)
#z1 = sigmoid(t1)
#z1_1 = np.c_[z1,-1]
#
#t2 = z1_1.dot(W2.T)
#z2 = sigmoid(t2)
#z2_2 = np.c_[z2,-1]
#
#t3 = z2_2.dot(W3.T)
#z3 = sigmoid(t3)
#
#error = np.sum((y_train[0:1,:] - z3)**2)/z3.shape[1]
#
#delta3 = (y_train[0:1,:] - z3)*(z3*(1-z3))
#W3 = W3 + lr * delta3.T.dot(z2_2)
#
#delta2 = delta3.dot(W3[:,:-1]) * (z2*(1-z2))
#W2 = W2 + lr * delta2.T.dot(z1_1)
#
#delta1 = delta2.dot(W2[:,:-1]) * (z1*(1-z1))
#W1 = W1 + lr * delta1.T.dot(x_train[0:1,:])
error_his = []
lr = 0.1
for k in range(0,100):
        
    for i in range(0, x_train.shape[0]):
        t1 = x_train[i:i+1,:].dot(W1.T)
        z1 = sigmoid(t1)
        z1_1 = np.c_[z1,-1]  # add bias
        
        t2 = z1_1.dot(W2.T)
        z2 = sigmoid(t2)
        z2_2 = np.c_[z2,-1]
        
        t3 = z2_2.dot(W3.T)
        z3 = sigmoid(t3)
        
        error = np.sum((y_train[i:i+1,:] - z3)**2)/z3.shape[1]
        if i%100 ==1:
            error_his.append(error)
        
        ##backprop
        delta3 = (y_train[i:i+1,:] - z3)*(z3*(1-z3))
        W3 = W3 + lr * delta3.T.dot(z2_2)
        
        delta2 = delta3.dot(W3[:,:-1]) * (z2*(1-z2))
        W2 = W2 + lr * delta2.T.dot(z1_1)
        
        delta1 = delta2.dot(W2[:,:-1]) * (z1*(1-z1))
        W1 = W1 + lr * delta1.T.dot(x_train[0:1,:])
        
        
plt.plot(error_his)
plt.show()


### https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets

t1 = x_train.dot(W1.T)
z1 = sigmoid(t1)
z1_1 = np.c_[z1, -np.ones(len(z1))]

t2 = z1_1.dot(W2.T)
z2 = sigmoid(t2)
z2_2 = np.c_[z2, -np.ones(len(z1))]

t3 = z2_2.dot(W3.T)
z3 = sigmoid(t3)

y_pred = []
for i in range(0, len(z3)):
    y_pred.append(np.argmax(z3[i,:]))

temp = 0
for i in range(0,len(z3)):
    if np.argmax(y_train[i,:]) == y_pred[i]:
        temp += 1

accuracy = temp / len(z3)

print(accuracy)