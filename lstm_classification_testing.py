# Function to prompt user for choice of classification train/test methodology
def train_test_method_prompt(): 
  valid = 0
  while valid == 0:
    train_test_method = input('Enter classification train/test methodology (80/20 split or separate train/test sets): ')
    if train_test_method == '80/20 split' or train_test_method == 'separate train/test sets':
      valid = 1
      break
    else:
      valid = 0
      print('Invalid input.')
      
  return train_test_method


# Function to prompt user for pcap, pcap csv, and pcap label csv file paths for 80/20 train/tets split methodology
def train_test_split_file_paths_prompt():
  import pathlib

  valid = 0
  while valid == 0:
    pcap_file_path = input('Enter pcap file path for feature generation: ') # directory path to where the pcap file is located
    if pathlib.Path(pcap_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0
      print('Invalid file path.')

  valid = 0
  while valid == 0:
    pcap_csv_file_path = input('Enter pcap csv file path feature generation: ') # directory path to where the pcap csv file is located
    if pathlib.Path(pcap_csv_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0
      print('Invalid file path.')

  valid = 0
  while valid == 0:
    pcap_label_file_path = input('Enter pcap label file path for classification: ') # directory path to where the pcap label file is located
    if pathlib.Path(pcap_label_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0
      print('Invalid file path.')

  return pcap_file_path, pcap_csv_file_path, pcap_label_file_path


# Function to prompt user for training and testing pcap, pcap csv, and pcap label csv file paths separte train/test sets methodology
def train_test_separate_file_paths_prompt():
  import pathlib

  valid = 0
  while valid == 0:
    train_pcap_file_path = input('Enter training pcap file path for feature generation: ') # directory path to where the training pcap file is located
    if pathlib.Path(train_pcap_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0
      print('Invalid file path.')

  valid = 0
  while valid == 0:
    train_pcap_csv_file_path = input('Enter training pcap csv file path feature generation: ') # directory path to where the training pcap csv file is located
    if pathlib.Path(train_pcap_csv_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0
      print('Invalid file path.')

  valid = 0
  while valid == 0:
    train_pcap_label_file_path = input('Enter training pcap label file path for classification: ') # directory path to where the training pcap label file is located
    if pathlib.Path(train_pcap_label_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0

  valid = 0
  while valid == 0:
    test_pcap_file_path = input('Enter testing pcap file path for feature generation: ') # directory path to where the testing pcap file is located
    if pathlib.Path(test_pcap_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0
      print('Invalid file path.')

  valid = 0
  while valid == 0:
    test_pcap_csv_file_path = input('Enter testing pcap csv file path feature generation: ') # directory path to where the testing pcap csv file is located
    if pathlib.Path(test_pcap_csv_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0
      print('Invalid file path.')

  valid = 0
  while valid == 0:
    test_pcap_label_file_path = input('Enter testing pcap label file path for classification: ') # directory path to where the testing pcap label file is located
    if pathlib.Path(test_pcap_label_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0

  return train_pcap_file_path, train_pcap_csv_file_path, train_pcap_label_file_path, test_pcap_file_path, test_pcap_csv_file_path, test_pcap_label_file_path


# Function to prompt user for feature extraction method to be used
def feature_extraction_method_prompt():
  valid = 0
  while valid == 0:  
    feature_extraction_method = input('Enter feature extraction method to use (AE, Manual, or NLP): ')
    if feature_extraction_method == 'AE' or feature_extraction_method == 'Manual' or feature_extraction_method == 'NLP':
      valid = 1
      break
    else:
      valid = 0
      print('Invalid input.')

  return feature_extraction_method


# Function to prompt user for encoder model file path
def encoder_file_path_prompt():
  import pathlib

  valid = 0
  while valid == 0:
    encoder_file_path = input('Enter encoder file path for autoencoder feature generation: ') # directory path to where the feature encoder is located
    if pathlib.Path(encoder_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0

  return encoder_file_path


# Function to prompt user for fine tuned NLP model file path
def fine_tuned_nlp_file_path_prompt():
  import pathlib

  valid = 0
  while valid == 0:
    fine_tuned_nlp_file_path = input('Enter fine tuned nlp file path for nlp feature generation: ') # directory path to where the fine tuned nlp is located
    if pathlib.Path(fine_tuned_nlp_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0

  return fine_tuned_nlp_file_path


# Function to prompt user for LSTM model input timestep value
def lstm_input_timestep_prompt():
  import numpy as np

  valid = 0
  while valid == 0:
    timestep = input('Enter input timestep for LSTM model (1-64): ')
    timestep = int(timestep)
    if timestep in np.arange(1,65):
      valid = 1
      break
    else:
      valid = 0
      print('Invalid input.')

  return timestep


# Function to prompt user for how LSTM model input timesteps are created
def lstm_input_timestep_method_prompt():
  valid = 0
  while valid == 0:
    timestep_method = input('Enter timestep method (Normal or Flow): ')
    if timestep_method == 'Normal' or timestep_method == 'Flow':
      valid = 1
      break
    else:
      valid = 0
      print('Invalid input.')

  return timestep_method


# Function to prompt user for LSTM model hyperparameters
def lstm_hyperparameter_prompt():
  # enter value for model training learning rate, check if input is a valid float greater than zero
  while True:
    lr = input('Enter value for model training learning rate: ')
    try:
      lr = float(lr)
    except ValueError:
      print('Invalid input.')
      continue
    if lr > 0:
      break
    else: 
      print('Invalid input.')
      continue

  # enter value for model training batch size, check if input is a valid int greater than zero
  while True:
    batch_size = input('Enter batch size for model training: ')
    try:
      batch_size = int(batch_size)
    except ValueError:
      print('Invalid input.')
      continue
    if batch_size > 0:
      break
    else: 
      print('Invalid input.')
      continue

  # enter value for model training epochs, check if input is a valid int greater than zero
  while True:
    epochs = input('Enter number of epochs to train model for: ')
    try:
      epochs = int(epochs)
    except ValueError:
      print('Invalid input.')
      continue
    if epochs > 0:
      break
    else: 
      print('Invalid input.')
      continue

  return lr, batch_size, epochs


if __name__ == '__main__':
  train_test_method = train_test_method_prompt() # get classification train/test methodology
  
  if train_test_method == '80/20 split':
    pcap_file_path, pcap_csv_file_path, pcap_label_file_path = train_test_split_file_paths_prompt() # get necessary files for the 80/20 train/test split methodology
  elif train_test_method == 'separate train/test sets': 
    train_pcap_file_path, train_pcap_csv_file_path, train_pcap_label_file_path, test_pcap_file_path, test_pcap_csv_file_path, test_pcap_label_file_path = train_test_separate_file_paths_prompt() # get necessary files for the separate train/test methodology

  feature_extraction_method = feature_extraction_method_prompt() # get feature extraction method

  if feature_extraction_method == 'AE':
    encoder_file_path = encoder_file_path_prompt() # get encoder file path

    import autoencoder_feature_generator # import script to generate autoencoder features

    if train_test_method == '80/20 split':
      pcap_features = autoencoder_feature_generator.create_ae_features(pcap_file_path, encoder_file_path, 'both') # create autoencoder features using specified pcap file and encoder model
    elif train_test_method == 'separate train/test sets':
      train_pcap_features = autoencoder_feature_generator.create_ae_features(train_pcap_file_path, encoder_file_path, 'train') # create training autoencoder features using specified pcap file and encoder model
      test_pcap_features = autoencoder_feature_generator.create_ae_features(test_pcap_file_path, encoder_file_path, 'test') # create testing autoencoder features using specified pcap file and encoder model

  elif feature_extraction_method == 'Manual':
    import manual_feature_generator # import script to generate manual features

    if train_test_method == '80/20 split':
      pcap_features = manual_feature_generator.create_manual_features(pcap_csv_file_path, 'both') # create manual features using specified pcap csv file
    elif train_test_method == 'separate train/test sets':
      train_pcap_features = manual_feature_generator.create_manual_features(train_pcap_csv_file_path, 'train') # create training manual features using specified pcap csv file
      test_pcap_features = manual_feature_generator.create_manual_features(test_pcap_csv_file_path, 'test') # create testing manual features using specified pcap csv file

  elif feature_extraction_method == 'NLP':
    fine_tuned_nlp_file_path = fine_tuned_nlp_file_path_prompt() # get fine tuned NLP file path

    import nlp_feature_generator # import script to generate NLP features

    if train_test_method == '80/20 split':
      pcap_features = nlp_feature_generator.create_nlp_features(pcap_csv_file_path, fine_tuned_nlp_file_path, 'both') # create Distilbert NLP features using specificed pcap file and fine tuned model
    elif train_test_method == 'separate train/test sets':
      train_pcap_features = nlp_feature_generator.create_nlp_features(train_pcap_csv_file_path, fine_tuned_nlp_file_path, 'train') # create Distilbert training NLP features using specificed pcap file and fine tuned model
      test_pcap_features = nlp_feature_generator.create_nlp_features(train_pcap_csv_file_path, fine_tuned_nlp_file_path, 'test') # create Distilbert testing NLP features using specificed pcap file and fine tuned model

  timestep = lstm_input_timestep_prompt() # get timestep value for lstm input

  timestep_method = lstm_input_timestep_method_prompt() # get timestep build method for lstm input

  import create_lstm_inputs # import script to generate LSTM input timesteps

  if timestep_method == 'Normal':
    if train_test_method == '80/20 split':
      lstm_inputs = create_lstm_inputs.create_timesteps(timestep, feature_extraction_method, 'both') # create lstm input timesteps
    elif train_test_method == 'separate train/test sets':
      lstm_train_inputs = create_lstm_inputs.create_timesteps(timestep, feature_extraction_method, 'train') # create lstm training input timesteps
      lstm_test_inputs = create_lstm_inputs.create_timesteps(timestep, feature_extraction_method, 'test') # create lstm testing input timesteps

  elif timestep_method == 'Flow':
    if train_test_method == '80/20 split':
      lstm_inputs = create_lstm_inputs.create_flow_timesteps(pcap_file_path, timestep, feature_extraction_method, 'both') # create lstm input flow timesteps
    elif train_test_method == 'separate train/test sets':
      lstm_train_inputs = create_lstm_inputs.create_flow_timesteps(train_pcap_file_path, timestep, feature_extraction_method, 'train') # create lstm training input flow timesteps
      lstm_test_inputs = create_lstm_inputs.create_flow_timesteps(test_pcap_file_path, timestep, feature_extraction_method, 'test') # create lstm testing input flow timesteps

  lr, batch_size, epochs = lstm_hyperparameter_prompt() # get lstm model hyperparameters

  import lstm_model # import script to train and test lstm model

  if train_test_method == '80/20 split':
    lstm_model.train_test_split_model(timestep, timestep_method, feature_extraction_method, pcap_label_file_path, lr, batch_size, epochs) # train model and predict using a single dataset with an 80/20 train/test split

  elif train_test_method == 'separate train/test sets':
    lstm_model.train_test_separate_model(timestep, timestep_method, feature_extraction_method, train_pcap_label_file_path, test_pcap_label_file_path, lr, batch_size, epochs) # train model and predict using a single dataset with an 80/20 train/test split
    