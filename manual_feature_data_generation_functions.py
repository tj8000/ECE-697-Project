# -*- coding: utf-8 -*-
"""Manual Feature Data Generation Functions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rI7QcdeQSsERNrTlEXd-DF_GhT9dkNRp

Note: this python file contains the manual feature extraction function MF() and the batched manual feature extraction function MF_batching(). The functions contained in this file should be called from the MLP Testing Notebook.ipynb file. Examples of using these functions are provided in the MLP Testing Notebook.ipynb file.

Note: any code used from a source has been reference via a comment directly next to that specific line of code.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

#this function is used to extract manual features from the Info field of raw network packet data
#inputs include just the dataframe containing the Info field

def MF(data):

  a,b = np.shape(data)
  info_data = data['Info']
  new_features = np.zeros((a,16)) # matrix to house the newly extracted features
  

  #define some keywords that we are going to search for
  PCAP_keywords = ['SYN','FIN','PSH','ACK', 'reassembled PDU', 'unseen segment', 'Previus segment not captured', 'HTTP', 'Fragmented', 'Bad', 'UDP', 'Malformed', 'Null', 'TCP']
  PCAP_integers = ['Seq', 'Ack']

  for i in range(0,a): # iterate through all the data samples
    keyword_count = 0 # for new_features index
    for j in PCAP_keywords: # iterate through all of the keywords we want to search for
      found_keyword = info_data[i].find(j)
      if found_keyword >= 0:
        new_features[i,keyword_count] = 1 # if keyword is found
      if found_keyword == -1: 
        new_features[i,keyword_count] = 0 # if keyword is not found
      keyword_count = keyword_count + 1

    for k in PCAP_integers: # now going to iterate though all the integers we are looking for
      start_location = info_data[i].find(k) 
      end_location = info_data[i].find(' ',start_location) # if we find the keyword, we want the location of the space afterwards to locate the integer
      found_integer = info_data[i][start_location+4:end_location] # then keep just the integer, not the location

      if start_location >= 0:
        new_features[i,keyword_count] = found_integer # if keyword is found, save integer
      if start_location == -1:
        new_features[i,keyword_count] = 0 # if keyword is not found, save a zero

      keyword_count = keyword_count + 1
  return new_features

#this function takes the manual features created in the previous function and batches then in groups, based on sorted IP addresses
#inputs include the manual features, desired batch size, and number of samples

def MF_batching(data,batch_size,a):

  #next we will batch these features based on IP address:
  batching_samples = data.head(a) #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html
  batching_samples = batching_samples.to_numpy() 

  start_index = 0 #start at the first IP address
  stop_index = 0
  #new_data = np.zeros((a,17)) # new matrix to contain our newly batched features
  new_data = np.empty((0,23), int)
  done = 0

  #start a loop to move through all of the samples given
  for j in range (0,a):

    if done == 1: #done flag
      break

    next_feature = np.zeros((1,23)) #this is an empty array that will hold the next set of features from the next 10 packets

    ip_start = batching_samples[start_index,2] # locate the IP address for the starting index

  #look at the next 10 samples, decide where to stop(either at 10, or when the IP address changes). If we reach the end of all samples, then stop there
    if start_index >= a-batch_size:
      for i in range (start_index+1,a):
        if ip_start != batching_samples[i,2]:
          stop_index = i
          done = 1
          break
        stop_index = i
        done = 1
    else:
      for i in range (start_index+1,start_index+batch_size):
        if ip_start != batching_samples[i,2]:
          stop_index = i
          break
        stop_index = i

    #matrix that contains only the rows we care about for this specific batch
    batch_matrix = batching_samples[start_index:stop_index,:]

    #first 16 features are summing the original 16 manual features
    for k in range(0,16):
      next_feature[0,k] = sum(batch_matrix[:,k+8])

    next_feature[0,16] = stop_index - start_index + 1

    #next feature generated is how many unique sequence numbers are found
    unique_seq = np.unique(batch_matrix[:,22])
    unique_seq_count = np.size(unique_seq)
    next_feature[0,17] = unique_seq_count

    #next feature generated is how many unique ack numbers are found
    unique_ack = np.unique(batch_matrix[:,23])
    unique_ack_count = np.size(unique_ack)
    next_feature[0,18] = unique_ack_count

    #next three features are checking to see which protocol flags are present when the other flags are also present
    if sum(batch_matrix[:,8]) == sum(batch_matrix[:,9]):
      next_feature[0,19] = 1
    
    if sum(batch_matrix[:,8]) == sum(batch_matrix[:,10]):
      next_feature[0,20] = 1

    if sum(batch_matrix[:,8]) == sum(batch_matrix[:,11]):
      next_feature[0,21] = 1

    #calculate final attack label
    attack_sum = sum(batch_matrix[:,7])
    if attack_sum > 0 :
      attack = 1
    else:
      attack = 0
    
    next_feature[0,22] = attack

    #add this newly generated row of features to a list that will be returned at the end
    new_data = np.append(new_data,next_feature,axis=0)

    start_index = stop_index

  return new_data
