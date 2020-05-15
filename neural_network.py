#Yugesha Sapte
#1001669305

import sys
import pandas as pd
import numpy as np
import math
import random

weight_matrix = []
each_layer_error = []
each_layer_output = []
each_layer_output_with_B = []
classes = []
no_of_classes = 0
predicted_label = []

def initialize_weight_matrix(units_per_layer, layers, no_of_features, no_of_classes): 
    np.random.seed(42)
    input_weights = initialize_weights(units_per_layer, no_of_features + 1)
    weight_matrix.append(input_weights)

    for i in range(1, layers): 
        hidden_layer_weights = initialize_weights(units_per_layer, units_per_layer + 1)
        weight_matrix.append(hidden_layer_weights)
    if(layers != 0): 
        output_weights = initialize_weights( no_of_classes, units_per_layer + 1)
        weight_matrix.append(output_weights)

def normalize_data(data):
  max = np.amax(data)
  data = data / max
  return data

def initialize_weights(n, m):
   return np.array(np.random.uniform(low= -0.05, high= 0.05, size=(n, m)))

def load_data(data): 
    training_labels = data[:, -1]
    data = normalize_data(data[:, :-1])
    return data, training_labels

def add_bias(data): 
    x0  = np.ones((1, data.shape[0]))  
    return np.concatenate((x0, data), axis = 1)

def get_unique_classes(training_labels):
    classes = np.unique(training_labels)
    no_of_classes = len(classes)
    return classes, no_of_classes

def train_model(training_data, layers): 
    training_data = add_bias(training_data)
    each_layer_output.clear()
    each_layer_output_with_B.clear()
    each_layer_output.append(training_data)
    each_layer_output_with_B.append(training_data)

    for i in range(layers + 1): 
        training_data = np.dot(training_data, weight_matrix[i].T)
        training_data = sigmoid(training_data)
        each_layer_output.append(training_data)

        if(i != layers):
            training_data = add_bias(training_data)

        each_layer_output_with_B.append(training_data)
        
    return training_data
              

def sigmoid(data): 
    return (1 / (1 + np.exp(-data)))

def one_hot_encoding(training_labels): 
    one_hot_encoding_label = np.zeros((1, no_of_classes))
    index = np.where(classes == training_labels)
    one_hot_encoding_label[0][index] = 1
    return one_hot_encoding_label

def calculate_accuracy(predicted_label, testing_labels): 
    test_labels = []  
    test_accuracy = []  
    acc_value = 0

    for objId in range(len(predicted_label)):
        max_value = np.amax(predicted_label[objId])
        max_count = np.count_nonzero(predicted_label[objId] == max_value)        

        if(max_count == 1): 
            label_index = np.argmax(predicted_label[objId]) 
            label =  classes[label_index]
          
            if(label == testing_labels[objId]): 
                acc_value += 1
        else: 
            if(is_valid_class(predicted_label[objId], testing_labels[objId],max_value)): 
                label_index = np.argmax(predicted_label[objId]) 
                label = classes[random.choice(label_index)]
                acc_value += 1/max_count
    
        test_labels.append(label)  
        
        test_accuracy.append(acc_value/len(testing_labels)*100)
        print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n'% (objId+1, label, testing_labels[objId], test_accuracy[objId]))   
    print("classification accuracy=%6.4f\n" %(acc_value/len(testing_labels)*100))


def is_valid_class(predicted_label, testing_labels, max_value):
    for i in range(len(predicted_label)): 
        if(max_value == predicted_label[i] and predicted_label[i] == testing_labels): 
            return True
    return False 


def calculate_error_and_update_weights(data, training_labels, layers, learning_rate):
    each_layer_error.clear()
    one_hot_encoding_label = one_hot_encoding(training_labels)
    term1 = (data - one_hot_encoding_label)
    term2 = (1-data) * data
    each_layer_error.append(term1 * term2)
    
    #calculate for hidden layers 
    for i in range(layers, 0, -1): 
        weight_without_bias = np.delete(weight_matrix[i],0 ,axis=1)
        delta_u = (np.dot(each_layer_error[0], weight_without_bias)) 
        delta_i = delta_u * (each_layer_output[i] * (1 - each_layer_output[i]))
        each_layer_error.insert(0, delta_i)
       

    for i in range(len(weight_matrix)):       
        weight_matrix[i] = weight_matrix[i] - (learning_rate * (np.dot(np.array(each_layer_error[i]).T, each_layer_output_with_B[i])))

def read_datasets(): 
 
    if(len(sys.argv) == 6 or len(sys.argv) == 5): 

        if(len(sys.argv) == 6): 
            training_file_name = sys.argv[1]
            testing_file_name = sys.argv[2]
            layers = int(sys.argv[3]) - 2 
            units_per_layer = int(sys.argv[4])
            rounds = int(sys.argv[5])
        else: 
            training_file_name = sys.argv[1]
            testing_file_name = sys.argv[2]
            layers = int(sys.argv[3]) - 2 
            units_per_layer = 0
            rounds = int(sys.argv[4])

        training_data = np.loadtxt(training_file_name)
        testing_data = np.loadtxt(testing_file_name)

        training_data , training_labels = load_data(training_data)
        testing_data, testing_labels = load_data(testing_data)        

        global classes
        global no_of_classes
        classes, no_of_classes =  get_unique_classes(training_labels)

        if(len(sys.argv) == 5):
             units_per_layer = no_of_classes
               
        initialize_weight_matrix(units_per_layer, layers, training_data.shape[1], no_of_classes)    
       
        learning_rate = 1
        for i in range(rounds) : 
                    
            for index in range(len(training_data)): 
                data = training_data[index, np.newaxis]                 
                data = train_model(data, layers)

                calculate_error_and_update_weights(data, training_labels[index], layers, learning_rate)
                
                
            learning_rate = learning_rate * 0.98

        for index in range(len(testing_data)): 
            data = testing_data[index, np.newaxis]                 
            data = train_model(data, layers)
            predicted_label.append(data)

        calculate_accuracy(predicted_label, testing_labels)

    else:
        print("Please provide valid command line inputs")


read_datasets()