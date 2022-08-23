### Zero-day DDoS Attack Detection

### Cameron Boeder, Troy Januchowski

### ECE 697 Project Codebase

### 8/23/22

# Table of Contents

1. Purpose of Codebase

1. Project Abstract

1. References

1. File Descriptions

1. Sample dataset

1. How to load and run python files and dataset

# 1. Purpose of Codebase

The purpose of this codebase is to provide a copy of the code used throughout this capstone project. This codebase can also be used as a guide for anyone who wants to run this code on their local machine. A sample dataset of 500 samples has been included in this codebase to run with the code. All train/test datasets used throughout this project have not been included in this codebase repository but can be found on the supplemental material google drive (https://drive.google.com/drive/folders/1hfz-N2XZDlBGXl0WxZ7MMmnPKSLsDvyb?usp=sharing). 

# 2. Project Abstract

The ability to detect zero-day (novel) attacks has become essential in the network security industry. Due to ever evolving attack signatures, existing network intrusion detection systems often fail to detect these threats. This project aims to solve the task of detecting zero-day DDoS (distributed denial-of-service) attacks by utilizing network traffic that is captured before entering a private network. Modern feature extraction techniques are used in conjunction with neural networks in order to determine if a network packet is either benign or malicious.

# 3. References

Any code used from a source has been reference via a comment directly next to that specific line of code. Note that the MLP and LSTM testing .ipynb notebooks call functions from separate .py files - these .py function files also contain code references.

# 4. File Descriptions

## 4a. manual_feature_data_generation_functions.py

This file is used to generate manual features and batched manual features from the .csv pcap file format. This python script contains two functions that should be called from the MLP Testing Notebook.ipynb. The functions contained in this file generate manual features to be used for classification testing. There are instructions within the MLP Testing Notebook.ipynb that walk through the use of the functions contained in this file.

## 4b. nlp_feature_data_generation_functions.py

This file is used to generate NLP features and batched NLP features from the .csv pcap file format. This file also contains a function to fine tune the Hugging Face Distilbert model. This python script contains three functions that should be called from the MLP Testing Notebook.ipynb. The functions contained in this file generate NLP features to be used for classification testing. There are instructions within the MLP Testing Notebook.ipynb that walk through the use of the functions contained in this file.

## 4c. MLP Testing Notebook.ipynb

This file contains instructions on how to extract manual and NLP features from the sample dataset. This file also contains the 5 MLP testing variations used in our project. All of the feature extraction techniques used in this notebook are ran through each of the 5 MLP models. This file is intended to be opened in a Google colab environment. There are instructions within this file on how to run each cell. Note that there are file locations that will need to be set in order to import the sample dataset.

## 4d. train_autoencoder.py

Given input data, this python script is used to train the autoencoder model.

## 4e. fine_tune_nlp.py

Given input data and labels, this python script is used to fine tune the Distilbert NLP model. This script was adapted from nlp_feature_data_generation_functions.py for use by the LSTM model.

## 4f. autoencoder_feature_generator.py

Given input data, this python script will generate autoencoder features using an already trained encoder model.

## 4g. manual_feature_generator.py

Given input data, this python script will generate manual features.

## 4h. nlp_feature_generator.py

Given input data, this python script will generate NLP features using a fine tuned Distilbert NLP model. This script was adapted from nlp_feature_data_generation_functions.py for use by the LSTM model.

## 4i. create_lstm_inputs.py

Given input features and a specified input timestep, this python script will create input timestep data for the LSTM model.

## 4j. lstm_model.py

Given input timestep features, this python script will create, train, and test the LSTM model.

## 4k. lstm_classification_testing.py

This script uses the autoencoder_feature_generator.py, manual_feature_generator.py, nlp_feature_generator.py, create_lstm_inputs.py, and lstm_model.py scripts to perform end to end testing of the LSTM model. This script lets the user decide what feature extraction technique to use, what the LSTM input timestep should be, how the LSTM input timesteps should be created, what the LSTM hyperparameters are, and how to split up the training and testing data.

## 4l. LSTM_Sample_Code_Notebook.ipynb

This google colab python notebook, walks the user through how to use the train_autoencoder.py, fine_tune_nlp.py, and lstm_classification_testing.py scripts using the codebase sample dataset stored on the supplementary material google drive (https://drive.google.com/drive/folders/1hfz-N2XZDlBGXl0WxZ7MMmnPKSLsDvyb?usp=sharing). 

## 4m. Scenario 1 Dataset Creation.ipynb

This file is included in the codebase to show the code structure used for creating the scenario 1-3 datasets. This is meant to be a standalone file just to show how the pcap files were generated. The data files called in this file have not been included in this codebase. Scenario 1-3 datasets can be found on the supplementary material google drive (https://drive.google.com/drive/folders/1hfz-N2XZDlBGXl0WxZ7MMmnPKSLsDvyb?usp=sharing).

## 4n. Scenario_4_Dataset_Creation.ipynb

This file is included in the codebase to show the code structure used for creating the scenario 4 dataset. This is meant to be a standalone file just to show how the pcap files were generated. The data files called in this file have not been included in this codebase. Scenario 1-3 datasets can be found on the supplementary material google drive (https://drive.google.com/drive/folders/1hfz-N2XZDlBGXl0WxZ7MMmnPKSLsDvyb?usp=sharing).

# 5. Sample Dataset

A small sample dataset was created for the purpose of testing this codebase. The sample includes 500 network data packets from the test set used in scenarios 1-4 of our project. The sample data set includes both benign and attack packets, labeled 0 and 1 respectively. There is a roughly 50/50 split between 0 and 1 labels in this sample dataset.

The sample dataset comes in two different file formats: one format is the .csv output from a packet analyzer, and the second format is .pcap. The Manual and NLP feature extractors use the .csv as input, whereas the autoencoder uses the .pcap as input. The third file included for the dataset is a separate .csv file that contains the 500 labels.

The sample dataset in .csv format, .pcap format, and the labels can be found in the "codebase sample dataset" folder of the project github.

The sample dataset is also stored on the supplementary material google drive (https://drive.google.com/drive/folders/1hfz-N2XZDlBGXl0WxZ7MMmnPKSLsDvyb?usp=sharing).

Note: the sample dataset uses 500 packets from the SUEE 2017 dataset, reference: https://github.com/vs-uulm/2017-SUEE-data-set

# 6. How to load and run python files and dataset

First, download the code repository, including the sample dataset folder, to your local machine. Place files in on a Google Drive.

**MLP Testing Notebook.ipynb**: open this notebook in a Google colab environment. There are instructions within the notebook that walk the user through the running of each cell. This notebook calls on the manual_feature_data_generation_functions.py and nlp_feature_data_generation_functions.py files, as well as imports the sample dataset files. When running the file, make sure to replace the drive location of the manual_feature_data_generation_functions.py file, nlp_feature_data_generation_functions.py file, and the sample data files to wherever you have copied them to your Google drive. The remaining instructions are within the notebook file. The notebook walks the user through each step.

**LSTM Testing Notebook.ipynb**: open this notebook in a Google colab environment. There are instructions within the notebook that walk the user through the running of each cell.

