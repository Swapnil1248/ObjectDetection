__author__ = 'Dhananjay Mehta/Swapnil Kumar'
from glob import glob
import numpy as np
from PIL import Image
import pandas as pd
import os
import cPickle
import gzip

def dir_to_dataset(new_path):
    #function extracts the data from the folder and returns resized array of grayscale image [28:28]
    dataset = []
    glob_files_updated_path = new_path + "*.png"
    for file_count, file_name in enumerate(sorted(glob(glob_files_updated_path),key=len)):
        image = Image.open(file_name)
        img = Image.open(file_name).convert('LA')
        img = img.resize((28,28))
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
    return dataset

def generate_csv_file(valid_dir_path, iteration_val):

    dir_path_valid_nw=valid_dir_path
    png = os.listdir(dir_path_valid_nw)
    l = len(png)
    np_valid_y_training = []
    #print "path", dir_path_valid_nw
    #c = int(dir_path_valid_nw[81:83])
    c = iteration_val
    if c <= 10:
        for j in range(l):
            #print ("\n"+str(47+c))
            np_valid_y_training.append(47+c)
    if c >10 and c<=36:
        for j in range(l):
            #print ("\n"+str(c+54))

            np_valid_y_training.append(54+c)
    if c >36:
        for j in range(l):
            #("\n"+str(c+60))

            np_valid_y_training.append(60+c)
    return np_valid_y_training

def create_dataset(valid_dir_path,csv_path):
    #valid_dir_path = "/Users/dhananjaymehta/CIS730/Vision/Training_Dataset/valid_dataset_1/"
    #csv_path = "/Users/dhananjaymehta/CIS730/Vision/EnglishFnt/CSV2/"

    final_train_set_x = np.zeros((1,784))  #final training dataset of images
    final_train_set_y = np.array([])       #final training dataset with the labels
    final_val_set_x = np.zeros((1,784))    #final validation dataset
    final_val_set_y = np.array([])         #finla validation dataset  with the labels
    final_test_set_x = np.zeros((1,784))   #final data to test
    final_test_set_y = np.array([])        #final data to test with labels


    #generating the data for the VALID INPUT  in scale - train:validate:test - 70:15:15
    Sample_list = os.listdir(valid_dir_path)
    i = 1
    for dir_in_sample in Sample_list:
        new_path=valid_dir_path + dir_in_sample + "/"
        csv_file_name = csv_path + dir_in_sample + ".csv"
        #print csv_file_name
        if(dir_in_sample != ".DS_Store"):
            np_valid_y_training= generate_csv_file(new_path, i)
            i += 1
            valid_data_training = dir_to_dataset(new_path)
            leng = len(valid_data_training)
            np_valid_data_training = np.array(valid_data_training)
            #np_valid_y_training = np.ones(leng)

            # : train_set_x -  calculate the value of training data - for images
            # : train_set_y -  calculate the value of training data - for labels
            train_set_x = np_valid_data_training[0:.7 * (leng)]
            train_set_y = np_valid_y_training[0:int(.7 * leng)]

            # : val_set_x -  calculate the value of validation data - for images
            # : val_set_y -  calculate the value of validation data - for labels
            val_set_x = np_valid_data_training[.7 * (leng): 0.85 * (leng)]
            val_set_y = np_valid_y_training[int(.7 * leng): int(0.85 * leng)]

            # : test_set_x -  calculate the value of validation data - for images
            # : test_set_y -  calculate the value of validation data - for labels
            test_set_x = np_valid_data_training[.85 * (leng): 1 * (leng)]
            test_set_y = np_valid_y_training[int(.85 * leng): 1 * (leng)]

            #print "shape",train_set_x.shape
            #print "shape",train_set_y.shape
            final_train_set_x = np.concatenate((final_train_set_x,train_set_x),axis=0)
            final_train_set_y = np.concatenate((final_train_set_y,train_set_y),axis=0)
            final_val_set_x = np.concatenate((final_val_set_x,val_set_x),axis=0)
            final_val_set_y = np.concatenate((final_val_set_y,val_set_y),axis=0)
            final_test_set_x = np.concatenate((final_test_set_x,val_set_x),axis=0)
            final_test_set_y = np.concatenate((final_test_set_y,val_set_y),axis=0)

    train_set = final_train_set_x[1:len(final_train_set_x)], final_train_set_y
    val_set = final_val_set_x[1:len(final_val_set_x)],final_val_set_y
    test_set = final_test_set_x[1:len(final_test_set_x)],final_test_set_y

    data_set = [train_set, val_set, test_set]
    print ("writing to file")

    f = gzip.open('char_num_classify.pkl.gz','wb')
    cPickle.dump(data_set, f, protocol=2)
    print ("file written")
    f.close()