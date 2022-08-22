#Function to pip install any needed packages
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


# Function to convert raw pcap data into autoencoder features
def create_ae_features(_pcap_file_path, _encoder_file_path, _set):
  install('scapy') # install scapy package

  # import necessary packages
  import re
  import numpy as np
  import pandas as pd
  from tensorflow import keras
  from scapy.all import PcapReader
  from scapy.all import bytes_hex
  
  pcap = PcapReader(_pcap_file_path) # read in pcap file
  encoder = keras.models.load_model(_encoder_file_path, compile=False) # load tensorflow feature encoder model
  
  pcap_ae_features = []
  for pkt in pcap: # loop through each packet in the pcap file
    temp_pkt = str(bytes_hex(pkt)) # scapy convert raw packet to a string of hexadecimal bytes
    temp_pkt = re.findall('..', temp_pkt)[1:] # parse string into bytes
    temp_pkt = np.pad(temp_pkt, (0, 1518-len(temp_pkt)), 'constant', constant_values=('00')) # pad packet to 1,518 bytes
    pkt_bytes = [] # initialize list to store packet bytes
    for i in range(len(temp_pkt)): 
        pkt_bytes.append(convert_hex_to_bin(temp_pkt[i])) # loop to convert each byte in packet from hexadecimal to binary
    pkt_bits = convert_bytes_to_bits(pkt_bytes) # parse each binary byte into 8 bits
    pkt_features = encoder.predict(np.array(pkt_bits, dtype=int).reshape(1, -1)) # use specified encoder model to compress packet from 12,144 bits to 128 bits
    pcap_ae_features.append(np.squeeze(pkt_features))
  pcap_ae_features = np.array(pcap_ae_features)
  if _set == 'both':
    pd.DataFrame(pcap_ae_features).to_csv('Generated_AE_Features.csv', header=None, index=False) # save autoencoder features to a csv file
  elif _set == 'train':
    pd.DataFrame(pcap_ae_features).to_csv('Generated_AE_Training_Features.csv', header=None, index=False) # save autoencoder features to a csv file
  elif _set == 'test':
    pd.DataFrame(pcap_ae_features).to_csv('Generated_AE_Testing_Features.csv', header=None, index=False) # save autoencoder features to a csv file
  return pcap_ae_features
  