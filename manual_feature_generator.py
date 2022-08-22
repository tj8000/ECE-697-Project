# Function to convert pcap wireshark info data into manual features
def create_manual_features(_pcap_csv_file_path, _set):
  # import necessary packages
  import numpy as np
  import pandas as pd

  pcap_csv = pd.read_csv(_pcap_csv_file_path) # read in pcap csv file

  input_data = pcap_csv['Info'] # prepare pcap csv for handling by nlp
  input_data_len = input_data.size # record length of input data

  pcap_manual_features = np.zeros((input_data_len, 16)) # matrix to house the newly extracted manual features
  
  # define keywords that to search for
  pcap_keywords = ['SYN', 'FIN', 'PSH', 'ACK', 'reassembled PDU', 'unseen segment', 'Previus segment not captured', 'HTTP', 'Fragmented', 'Bad', 'UDP', 'Malformed', 'Null', 'TCP']
  pcap_integers = ['Seq', 'Ack']

  for i in range(0, input_data_len): # iterate through all the data samples
    keyword_count = 0 # for manual features index
    for j in pcap_keywords: # iterate through all of the keywords to search for
      found_keyword = input_data[i].find(j)
      if found_keyword >= 0:
        pcap_manual_features[i, keyword_count] = 1 # if keyword is found
      if found_keyword == -1: 
        pcap_manual_features[i, keyword_count] = 0 # if keyword is not found
      keyword_count = keyword_count + 1

    for k in pcap_integers: # iterate though all the integers to search for
      start_location = input_data[i].find(k) 
      end_location = input_data[i].find(' ', start_location) # if keyword is found, store the location of the space afterwards to locate the integer
      found_integer = input_data[i][start_location+4:end_location] # keep just the integer, not the location

      if start_location >= 0:
        pcap_manual_features[i, keyword_count] = found_integer # if keyword is found, save integer
      if start_location == -1:
        pcap_manual_features[i, keyword_count] = 0 # if keyword is not found, save a zero

      keyword_count = keyword_count + 1

  if _set == 'both':
    pd.DataFrame(pcap_manual_features).to_csv('Generated_Manual_Features.csv', header=None, index=False) # save manual features to a csv file
  elif _set == 'train':
    pd.DataFrame(pcap_manual_features).to_csv('Generated_Manual_Training_Features.csv', header=None, index=False) # save manual features to a csv file
  elif _set == 'test':
    pd.DataFrame(pcap_manual_features).to_csv('Generated_Manual_Testing_Features.csv', header=None, index=False) # save manual features to a csv file
  return pcap_manual_features
