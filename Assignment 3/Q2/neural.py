import numpy as np
import sys
import time
start_time = time.time()
train_data = np.loadtxt("../../../part2/fmnist_train.csv",dtype=int,delimiter=',')
(m,features) = np.shape(train_data)
features = features-1
x_train = np.zeros((m,features))
y_train = np.zeros((m,1))
x_train = train_data[:,:features]
y_train = train_data[:,features]

#data normalized by dividing by highest pixel unit
x_train_normalize = x_train/(256.0)
CLASSES= 10
y_train_encoding = np.zeros((m,CLASSES),dtype=np.double)
def encode(y_val):
    answer = np.zeros((1,CLASSES),dtype=np.double)
    answer[0,y_val-1] = 1.0
    return answer
for i in range(m):
    y_train_encoding[i,:] = encode(y_train[i])
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
#function to find o(l) and net(l) given a theta dictionary and x_data
#we will always have to add a neuron that outputs 1
#assume theta for each neuron arranged in every row
def output(x_data,theta,layers):
    input = np.vstack((np.array([[1.0]]),x_data))
    layer_output = {0:x_data}
    for l in range(1,layers+1):
        input = theta[l] @ input
        input = sigmoid(input)
        layer_output[l] = input
        input = np.vstack((np.array([[1.0]]),input))
    return layer_output #this is the column vector o(l)

def all_outputs(x_data,theta,layers):
    
#y_data is matrix of one-hot encoding each row represents a output
def gradients(x_data,theta,layers,y_data):
    layer_output = output(x_data,theta,layers)
    theta_gradients = {}
    net_gradients = {}
    net_gradients[layers] =(y_data.T - layer_output[layers]) @ (layer_output[layers] * (layer_output[layers]-1))
    theta_gradients[layers] = net_gradients[layers] @ (np.vstack((np.array([[1.0]])),layer_output[layers-1]).T)
    for l in range(layers-1,0,-1):
        J_l = net_gradients[l+1]
        net_gradients[l] = ((J_l.T @ theta[l+1]).T) * (layer_output[l] * (1-layer_output[l]))
        output_new = np.vstack((np.array([[1.0]]),layer_output[l-1]))
        theta_gradients[l] = net_gradients[l] @ output_new.T
    return theta_gradients

def gradient_descent(x_train_norm,y_train_bit,eta,layer_list):
    train_whole = np.hstack((x_train,y_train_bit))
    np.random.shuffle(train_whole)
    theta = {}
    layers = len(layer_list)
    previous_input = features
    for layer in range(1,layers+1):
        neurons = layer_list[layer-1]
        theta_val = np.zeros((neurons,1+previous_input),dtype=np.double)
        previous_input = neurons
        theta[layer] = theta_val
    #theta constructed now only batch taking and gradient change left
    gradients(x_train,theta,layers,y_train_bit)
gradient_descent(x_train_normalize,y_train_encoding,0.1,[100,50])
end_time = time.time()
print(f"Time Taken is : {end_time - start_time}")