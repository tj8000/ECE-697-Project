#Function to pip install any needed packages
def install(package):
  import subprocess
  import sys
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])


# Function that reads in features and creates timesteps for the LSTM model
def create_timesteps(_timestep, _feature_extraction_method, _set):
  # import necessary packages
  import numpy as np
  import pandas as pd

  if _set == 'both':
    pcap_features = np.array(pd.read_csv('Generated_'+_feature_extraction_method+'_Features.csv', header=None)) # read in feature csv file and convert to numpy array
  elif _set == 'train':
    pcap_features = np.array(pd.read_csv('Generated_'+_feature_extraction_method+'_Training_Features.csv', header=None)) # read in feature csv file and convert to numpy array
  elif _set == 'test':
    pcap_features = np.array(pd.read_csv('Generated_'+_feature_extraction_method+'_Testing_Features.csv', header=None)) # read in feature csv file and convert to numpy array
  
  num_features = np.shape(pcap_features)[1] # number of features in dataset
  
  pcap_timestep_features = np.zeros((len(pcap_features), (num_features*_timestep))) # create empty array to store timestepped features

  for i in range(len(pcap_features)): # loop through all packets in dataset
    if i < _timestep: # if current packet number is less then timestep pad with zeros
      pkt_timestep = pcap_features[:i+1, :] # load packets 1 to i + 1 into timestep array
      pkt_timestep = pkt_timestep.flatten()
      pkt_timestep = np.pad(pkt_timestep, (num_features*_timestep - len(pkt_timestep), 0)) # pad to fill full timestep
    else: # load last timestep number of packets into timestep array
      pkt_timestep = pcap_features[i-(_timestep-1):i+1, :] 
      pkt_timestep = pkt_timestep.flatten()
    pcap_timestep_features[i, :] = pkt_timestep # store flattened timestep array
  
  if _set == 'both':
    pd.DataFrame(pcap_timestep_features).to_csv('Generated_'+_feature_extraction_method+'_Timestep_Features.csv', header=None, index=False) # save timestepped features to a csv file
  elif _set == 'train':
    pd.DataFrame(pcap_timestep_features).to_csv('Generated_'+_feature_extraction_method+'_Timestep_Training_Features.csv', header=None, index=False) # save timestepped features to a csv file
  elif _set == 'test':
    pd.DataFrame(pcap_timestep_features).to_csv('Generated_'+_feature_extraction_method+'_Timestep_Testing_Features.csv', header=None, index=False) # save timestepped features to a csv file
  return pcap_timestep_features


# Function that reads in features and creates flow timesteps for the LSTM model
def create_flow_timesteps(_pcap_file_path, _timestep, _feature_extraction_method, _set):
  install('scapy') # install scapy package

  # import necessary packages
  import numpy as np
  import pandas as pd
  from scapy.all import PcapReader
  from scapy.layers.inet import IP

  pcap = PcapReader(_pcap_file_path) # read in pcap file

  i = 0 # initialize packet index tracking variable
  flows = [] # initialize packet flow list
  flows_pkt_idx = [] # initialize packet flow index list
  pkt1 = [pcap.next()] # load first packet in pcap file
  flows.append(pkt1) # store first packet flow 
  flows_pkt_idx.append([i]) # store first packet flow index
  i += 1 # increment packet index tracking variable

  # loop through the all the remaining packets in the pcap file to create flows
  for pkt in pcap:
      k = -1 # initialize variable to track index in reversed flow index array
      rev_flow_idx = np.flip(np.arange(len(flows))) # create an sequential array the length of the current flow array and reversed the order

      # loop through each flow already created in a reversed order
      for flow in reversed(flows):
          k += 1 # icrement reversed flow index tracking variable
          temp_flow = [] # initialize list to store current flow build
          temp_flow_idx = [] # initialize list to store current flow index
          if IP in pkt and IP in flow[-1][0]: # check if there in a valid IP layer in the current packet and current flow
              if pkt[IP].src == flow[-1][0][IP].src and pkt[IP].dst == flow[-1][0][IP].dst \
                      or pkt[IP].dst == flow[-1][0][IP].src and pkt[IP].src == flow[-1][0][IP].dst: # check if current packet has same sourc/destination IP addresss asa current flow
                  if len(flow) < _timestep: # if the current flow length is less than the timestep
                      for j in range(len(flow)):
                          temp_flow.append(flow[j]) # append each packet in the the current flow to the new flow
                          temp_flow_idx.append(flows_pkt_idx[rev_flow_idx[k]][j]) # append each packet index in the current flow to the new flow index
                      temp_flow.append(pkt) # append the current packet to the new flow
                      temp_flow_idx.append(i) # append the current packet index to the new flow index 
                      break # new flow is ready to be stored, break flow loop
                  else: # if the current flow length is greater than or equal to the timestep
                      for j in range((len(flow) - _timestep + 1), len(flow)):
                          temp_flow.append(flow[j]) # append all packets except the first one in the current flow to the new flow
                          temp_flow_idx.append(flows_pkt_idx[rev_flow_idx[k]][j]) # append all packet indexes except the first on ein the current flow to the new flow
                      temp_flow.append(pkt) # append the current packet to the new flow
                      temp_flow_idx.append(i) # append the current packet index to the new flow index 
                      break
      if len(temp_flow) < 1: # if the current packet source/destination IP didn't match any previous flows
          temp_flow.append([pkt]) # create a new flow only containing the current packet
          temp_flow_idx.append(i) # store the index of the current packet for the new flow
      flows.append(temp_flow) # append the new flow to the packet flow list
      flows_pkt_idx.append(temp_flow_idx) # append the new flow index to the packet flow index list
      i += 1 # increment the packet index tracking variable

  if _set == 'both':
    pcap_features = np.array(pd.read_csv('Generated_'+_feature_extraction_method+'_Features.csv', header=None)) # read in feature csv file and convert to numpy array
  elif _set == 'train':
    pcap_features = np.array(pd.read_csv('Generated_'+_feature_extraction_method+'_Training_Features.csv', header=None)) # read in feature csv file and convert to numpy array
  elif _set == 'test':
    pcap_features = np.array(pd.read_csv('Generated_'+_feature_extraction_method+'_Testing_Features.csv', header=None)) # read in feature csv file and convert to numpy array
 
  num_features = np.shape(pcap_features)[1] # number of features in dataset

  i = 0 # initialize index tracking variable
  pcap_flow_timestep_features = np.zeros((len(flows_pkt_idx), num_features*_timestep)) # create empty array to store flow timestepped features
  for flow_pkt_idx in flows_pkt_idx: # loop through all of the created flows
      temp_flow = [] # initialize list to store the features of each flow
      flow_features = [] # initialize list to store each flow
      for pkt_idx in flow_pkt_idx: # loop through all indexes in each flow
          temp_flow.append(pcap_features[pkt_idx]) # append the current flow packet features to the current flow
      flow_features.append(np.squeeze(temp_flow)) # reduce dimensions of each flow packet features to 1D
      flow_features = np.array(flow_features).flatten() # convert to numpy array and reduce dimensions to 1D
      flow_features = np.pad(flow_features, (num_features*_timestep - len(flow_features), 0), 'constant') # pad each flow to specificed timestep length
      pcap_flow_timestep_features[i] = flow_features # store each flow in the final flow timestepped feature array
      i += 1 # increment index tracking variable 

  if _set == 'both':
    pd.DataFrame(pcap_flow_timestep_features).to_csv('Generated_'+_feature_extraction_method+'_Flow_Timestep_Features.csv', header=None, index=False) # save flow timestepped features to a csv file
  elif _set == 'train':
    pd.DataFrame(pcap_flow_timestep_features).to_csv('Generated_'+_feature_extraction_method+'_Flow_Timestep_Training_Features.csv', header=None, index=False) # save flow timestepped features to a csv file
  elif _set == 'test':
    pd.DataFrame(pcap_flow_timestep_features).to_csv('Generated_'+_feature_extraction_method+'_Flow_Timestep_Testing_Features.csv', header=None, index=False) # save flow timestepped features to a csv file
  return pcap_flow_timestep_features
  