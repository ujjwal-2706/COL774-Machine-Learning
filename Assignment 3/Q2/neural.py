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
x_train_normalize = x_train/(255.0)
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
#x_data arranged in each column
def output(x_data,theta,layers):
    feat,points = np.shape(x_data)
    input = np.vstack((np.ones((1,points),dtype=np.double),x_data))
    layer_output = {0:x_data}
    for l in range(1,layers+1):
        input = theta[l] @ input
        input = sigmoid(input)
        layer_output[l] = input
        feat,points = np.shape(input)
        input = np.vstack((np.ones((1,points),dtype=np.double),input))
    return layer_output #this dictoinary of matrix o(l) in each column for m points
    
#y_data is matrix of one-hot encoding each row represents a output
#x_data arranged in column form
def gradients(x_data,theta,layers,y_data):
    layer_output = output(x_data,theta,layers)
    theta_gradients = {}
    net_gradients = {}
    #its storing a matrix with ith column representing ith data point
    net_gradients[layers] =(y_data.T - layer_output[layers]) * (layer_output[layers] * (layer_output[layers]-1))
    row,col = np.shape(layer_output[layers-1])
    theta_gradients[layers] = net_gradients[layers] @ (np.vstack(((np.ones((1,col),dtype=np.double)),layer_output[layers-1])).T)
    for l in range(layers-1,0,-1):
        J_l = net_gradients[l+1]
        theta_temp = theta[l+1][:,1:] #remove the constants
        net_gradients[l] = (theta_temp.T @ J_l) * (layer_output[l] * (1-layer_output[l]))
        row,col = np.shape(layer_output[l-1])
        theta_gradients[l] = net_gradients[l] @ (np.vstack(((np.ones((1,col),dtype=np.double)),layer_output[l-1])).T)
    return theta_gradients

#kth batch of size batch_size and r is the output one-hot encodes
def batch_slice(train_whole,batch_size,r,k):
    total_points,values = np.shape(train_whole)
    left_end = (batch_size*k) % total_points
    right_end = (batch_size*(k+1)) % total_points
    if left_end < right_end:
        output = train_whole[left_end:right_end,:]
        x_value = output[:,:values-r]
        y_value = output[:,values-r:]
        return (x_value,y_value)
    else:
        output1 = train_whole[left_end:,:]
        output2 = train_whole[:right_end,:]
        output = np.vstack((output1,output2))
        x_value = output[:,:values-r]
        y_value = output[:,values-r:]
        return (x_value,y_value)
def gradient_descent(x_train_norm,y_train_bit,eta,layer_list,batch_size):
    train_whole = np.hstack((x_train_norm,y_train_bit))
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
    start_val = time.time()
    gradients(x_train.T,theta,layers,y_train_bit)
    end_val = time.time()
    print(f"gradient time is : {end_val-start_val}")
gradient_descent(x_train_normalize,y_train_encoding,0.1,[100,50,10])
end_time = time.time()
print(f"Time Taken is : {end_time - start_time}")