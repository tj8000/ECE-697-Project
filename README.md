### Zero-day DDoS Attack Detection

### Cameron Boeder, Troy Januchowski

### ECE 697 Codebase

### 8/19/22

Project Abstract: The ability to detect zero-day (novel) attacks has become essential in the network security industry. Due to ever evolving attack signatures, existing network intrusion detection systems often fail to detect these threats. This project aims to solve the task of detecting zero-day DDoS (distributed denial-of-service) attacks by utilizing network traffic that is captured before entering a private network. Modern feature extraction techniques are used in conjunction with neural networks in order to determine if a network packet is either benign or malicious.

# Table of Contents

1. Purpose of Codebase

1. Project Description

1. File Descriptions

> 3a. SQL_Sorting.py

> 3b. PLACEHOLDER - hexdump file

> 3c. Manual_Features.py

> 3d. NLP_Features.py

> 3e. PLACEHOLDER - AE Feature generation

> 3f. Feedforward_NN_Testing.ipynb

> 3g. PLACEHOLDER - LSTM testing notebook

> 3g. PLACEHOLDER - any other files?

4. Sample dataset

5. How to load python files and dataset

6. How to execute and use files

# 1. Purpose of Codebase

# 2. Project Description

# 3. File Descriptions

## 3a. SQL_Sorting.py

The SQL_Sorting.py file is used to sort the packet data in the .csv file format. This python script contains a function that should be called from one of the testing python notebook files. The function contained in this file utilizes SQL to sort the data packets based on IP address and time.

This function is ran specifically for generating the batched manual or NLP features. If you want to generate the manual or NLP features without batching, then this file is not needed. If generating batched manual or NLP features, this function should be ran first before running the respective manual or NLP feature generation functions.

This function outputs a numpy matrix that is the same dimensions as the pcap .csv input table, but it is now sorted.

## 3b.


## 3c. Manual_Features.py

The Manual_Features.py file is used to generate manual features and batched manual features from the .csv pcap file format. This python script contains two functions that should be called from one of the testing python notebook files. The functions contained in this file generate manual features to be used for classification testing.

add more about the output...


## 3d. NLP_Features.py

The NLP_Features.py file is used to generate NLP features and batched NLP features from the .csv pcap file format. This python script contains two functions that should be called from one of the testing python notebook files. The functions contained in this file generate NLP features to be used for classification testing.

add more about the output...


## 3e.


## 3f. Feedforward_NN_Testing.ipynb

The Feedforward_NN_Testing.ipynb file is to be used for classification testing using a feed foward neural network. This file is a python notebook intended to be opened in a colab environment. This file can be used to call any of the functions from the .py files above.

This file contains 5 different Feedforward neural network models that were testing during stage 2 of our project. Before running any of the models, the user will need to first generate a feature set from the .py files noted above.


## 3g.


## 3h.

# 4. Sample Dataset

A small sample dataset was created for the purpose of testing this codebase. The sample includes 500 network data packets from the test set used in scenarios 1-4 of our project. The sample data set includes both benign and attack packets, labeled 0 and 1 respectively. 

The sample dataset comes in two different file formats: one format is the .csv output from a packet analyzer, and the second format is .pcap. The Manual and NLP feature extractors use the .csv as input, whereas the autoencoder uses the .pcap as input. The third file included for the dataset is a separate .csv file that contains the 500 labels.

The sample dataset in .csv format, .pcap format, and the labels can be found in the "Sample Dataset" folder of the project github.

# 5. How to load files

# 6. How to execute files
