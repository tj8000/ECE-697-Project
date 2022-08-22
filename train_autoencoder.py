# Function to pip install any needed packages
def install(package):
  import subprocess
  import sys
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])


# Function to convert hexadecimal to binary bytes
def convert_hex_to_bin(_byte):
  scale = 16
  num_of_bits = 8
  return bin(int(_byte, scale))[2:].zfill(num_of_bits)


# Function to convert binary bytes to binary bits
def convert_bytes_to_bits(_bytes):
  temp_bits = ''.join(_bytes)
  bits = list(temp_bits)
  return bits


# Function to prompt user for autoencoder training pcap file
def ae_train_pcap_file_paths_prompt():
  import pathlib

  valid = 0
  while valid == 0:
    pcap_file_path = input('Enter pcap file path to be used for autoencoder training: ') # directory path to where the pcap file is located
    if pathlib.Path(pcap_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0
      print('Invalid file path.')

  return pcap_file_path


# Function to prepare pcap file for autoencoder handling
def prepare_ae_inputs(_pcap_file_path):
  install('scapy') # install scapy package

  # import necessary packages
  import numpy as np
  import re
  from scapy.all import PcapReader
  from scapy.all import bytes_hex

  pcap = PcapReader(_pcap_file_path) # read in pcap file
  raw_pcap_data = [] # initialize list to store raw pcap data
  for pkt in pcap: # loop through each packet in the pcap file
    temp_pkt = str(bytes_hex(pkt)) # scapy convert raw packet to a string of hexadecimal bytes
    temp_pkt = re.findall('..', temp_pkt)[1:] # parse string into bytes
    temp_pkt = np.pad(temp_pkt, (0, 1518-len(temp_pkt)), 'constant', constant_values=('00')) # pad packet to 1,518 bytes
    pkt_bytes = [] # initialize list to store packet bytes
    for i in range(len(temp_pkt)):
      pkt_bytes.append(convert_hex_to_bin(temp_pkt[i])) # loop to convert each byte in packet from hexadecimal to binary 
    pkt_bits = convert_bytes_to_bits(pkt_bytes) # parse each binary byte into 8 bits
    raw_pcap_data.append(pkt_bits) # append each packets raw data to the overall list
  raw_pcap_data = np.array(raw_pcap_data, dtype=int)
  return raw_pcap_data


# Function to prompt user for autoencoder model hyperparameters
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


def custom_loss(_y_true, _y_pred):
  # import necessary packages
  from tensorflow import keras
  mse = keras.losses.mean_squared_error(_y_true, _y_pred)
  kl = keras.losses.kl_divergence(_y_true, _y_pred)
  return mse+kl


# Function to train autoencoder model
def train_ae_model(_input_data, _lr, _batch_size, _epochs):
  # import necessary packages
  import tensorflow as tf
  from tensorflow import keras
  from keras import layers
  import matplotlib.pyplot as plt

  # build autoencoder model
  input_layer = keras.Input(shape=(12144,), name='input_layer')
  encoder_h1 = layers.Dense(1518, name='encoder_hidden_layer')(input_layer)
  bottleneck_layer = layers.Dense(128, activation='sigmoid', activity_regularizer=keras.regularizers.L1(0.0001), name='sparse_bottleneck_layer')(encoder_h1)
  decoder_h1 = layers.Dense(1518, name='decoder_hidden_layer')(bottleneck_layer)
  output_layer = layers.Dense(12144, activation='sigmoid', name='ouput_layer')(decoder_h1)

  encoder = keras.Model(input_layer, bottleneck_layer) # define encoder model
  autoencoder = keras.Model(input_layer, output_layer) # define autoencoder model

  autoencoder.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=_lr), metrics=['mean_squared_error', 'kullback_leibler_divergence']) # compile autoencoder model

  history = autoencoder.fit(_input_data, _input_data, batch_size=_batch_size, epochs=_epochs , shuffle=True) # train model

  # plot the training loss and save
  loss = history.history['loss']
  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, 'r', label='Training loss')
  plt.title('Training loss')
  plt.legend()
  plt.savefig('AE_Training_Loss_Curve.png')
  
  print('\n')

  encoder.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=_lr)) # compile encoder model for saving

  autoencoder.save('autoencoder') # save auotencoder model
  encoder.save('encoder') # save encoder


if __name__ == '__main__':
  pcap_file_path = ae_train_pcap_file_paths_prompt() # get pcap file path for training

  ae_inputs = prepare_ae_inputs(pcap_file_path) # prepare pcap data for input to autoencoder

  lr, batch_size, epochs = lstm_hyperparameter_prompt() # get autoencoder model parameters

  train_ae_model(ae_inputs, lr, batch_size, epochs) # train autoencoder model
  