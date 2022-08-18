### Zero-day DDoS Attack Detection

### Cameron Boeder, Troy Januchowski

### ECE 697 Project Codebase

### 8/19/22

Project Abstract: The ability to detect zero-day (novel) attacks has become essential in the network security industry. Due to ever evolving attack signatures, existing network intrusion detection systems often fail to detect these threats. This project aims to solve the task of detecting zero-day DDoS (distributed denial-of-service) attacks by utilizing network traffic that is captured before entering a private network. Modern feature extraction techniques are used in conjunction with neural networks in order to determine if a network packet is either benign or malicious.

# Table of Contents

1. Purpose of Codebase

1. Project Abstract

1. File Descriptions

1. Sample dataset

1. How to load and run python files and dataset

# 1. Purpose of Codebase

The purpose of this codebase is to provide a copy of the code used throughout this capstone project. This codebase can also be used as a guide for anyone who wants to run this code on their local machine. A sample dataset of 500 samples has been included in this codebase to run with the code. All train/test datasets used throughout this project have not been included in this codebase repository. 

# 2. Project Abstract

The ability to detect zero-day (novel) attacks has become essential in the network security industry. Due to ever evolving attack signatures, existing network intrusion detection systems often fail to detect these threats. This project aims to solve the task of detecting zero-day DDoS (distributed denial-of-service) attacks by utilizing network traffic that is captured before entering a private network. Modern feature extraction techniques are used in conjunction with neural networks in order to determine if a network packet is either benign or malicious.

# 3. File Descriptions

## 3a. manual_feature_data_generation_functions.py

This file is used to generate manual features and batched manual features from the .csv pcap file format. This python script contains two functions that should be called from the MLP Testing Notebook.ipynb. The functions contained in this file generate manual features to be used for classification testing. There are instructions within the MLP Testing Notebook.ipynb that walk through the use of the functions contained in this file.

## 3b. nlp_feature_data_generation_functions.py

This file is used to generate NLP features and batched NLP features from the .csv pcap file format. This file also contains a function to fine tune the Hugging Face Distilbert model. This python script contains three functions that should be called from the MLP Testing Notebook.ipynb. The functions contained in this file generate NLP features to be used for classification testing. There are instructions within the MLP Testing Notebook.ipynb that walk through the use of the functions contained in this file.


## 3c. MLP Testing Notebook.ipynb

This file contains instructions on how to extract manual and NLP features from the sample dataset. This file also contains the 5 MLP testing variations used in our project. All of the feature extraction techniques used in this notebook are ran through each of the 5 MLP models. This file is intended to be opened in a Google colab environment. There are instructions within this file on how to run each cell. Note that there are file locations that will need to be set in order to import the sample dataset.




## 3d. 


## 3e.


## 3f.


## 3g.


## 3h.

# 4. Sample Dataset

A small sample dataset was created for the purpose of testing this codebase. The sample includes 500 network data packets from the test set used in scenarios 1-4 of our project. The sample data set includes both benign and attack packets, labeled 0 and 1 respectively. There is a roughly 50/50 split between 0 and 1 labels in this sample dataset.

The sample dataset comes in two different file formats: one format is the .csv output from a packet analyzer, and the second format is .pcap. The Manual and NLP feature extractors use the .csv as input, whereas the autoencoder uses the .pcap as input. The third file included for the dataset is a separate .csv file that contains the 500 labels.

The sample dataset in .csv format, .pcap format, and the labels can be found in the "codebase sample dataset" folder of the project github.

# 5. How to load and run python files and dataset

First, download the code repository, including the sample dataset folder, to your local machine. Place files in on a Google Drive.

**MLP Testing Notebook.ipynb**: open this notebook in a Google colab environment. There are instructions within the notebook that walk the user through the running of each cell. This notebook calls on the manual_feature_data_generation_functions.py and nlp_feature_data_generation_functions.py files, as well as imports the sample dataset files. When running the file, make sure to replace the drive location of the manual_feature_data_generation_functions.py file, nlp_feature_data_generation_functions.py file, and the sample data files to wherever you have copied them to your Google drive. The remaining instructions are within the notebook file. The notebook walks the user through each step.

