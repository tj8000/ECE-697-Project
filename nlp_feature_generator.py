#Function to pip install any needed packages
def install(package):
  import subprocess
  import sys
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])


# This function references code developed at 
# https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb
# Function to convert pcap wireshark info data into NLP features
def create_nlp_features(_pcap_csv_file_path, _fine_tuned_nlp_file_path, _set):
  install('transformers') # install the transformers package
  
  # import necessary packages
  import numpy as np
  import pandas as pd
  import torch
  import transformers as ppb
  from transformers import AutoModelForSequenceClassification
  
  pcap_csv = pd.read_csv(_pcap_csv_file_path) # read in pcap csv file
  
  input_data = pcap_csv['Info'] # prepare pcap csv for handling by nlp
  input_data_len = input_data.size # record length of input data

  model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased') # load Distilbert tokenizer
  tokenizer = tokenizer_class.from_pretrained(pretrained_weights) # load pretrained weights for Distilbert tokenizer

  model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2) # load Distilbert model
  model = model.from_pretrained(_fine_tuned_nlp_file_path) # load fine tuned weights for Distilbert

  done = 0 # done flag
  pcap_nlp_features = np.empty((0, 768), int) # create an empty array to put the newly generated sentence embedding for each sample

  # this for loop is needed, as the sample sizes used in this project are often too large (i.e. 100k samples) to run all at once
  # do not have enough ram with colab to run.
  # instead, the sentence embeddings are generated in batches - for example, groups of 150.
  for i in range(0, 1000):

    if done == 1:
      break
    
    # checking how many samples are left to generate. If more than 150, run full 150. If less, run that amount
    if i*150 < input_data_len-150:
      start_index = i*150
      end_index = (i+1)*150

    if i*150 >= input_data_len-150:
      done = 1
      start_index = i*150
      end_index = input_data_len

    batch = input_data[start_index:end_index] # isolate batch of samples that are going to have sentence embeddings generated this batch

    tokenized = batch.apply((lambda x: tokenizer.encode(x, add_special_tokens=True))) # tokenize sample strings

    # loop to pad tokenized values
    max_len = 0 
    for j in tokenized.values: 
        if len(j) > max_len: 
            max_len = len(j) 
    padded = np.array([k + [0]*(max_len-len(k)) for k in tokenized.values])

    # have model ignore the masks
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape

    # create torch tensors from both input ids and masks
    # both are required inputs for Distilbert
    input_ids = torch.tensor(padded)   
    attention_mask = torch.tensor(attention_mask)

    # limit tokens to 512, the Distilbert limit
    input_ids = input_ids[:, 0:512]
    attention_mask = attention_mask[:, 0:512]

    # predict using fine tuned model
    with torch.no_grad():
      last_hidden_states = model.distilbert(input_ids, attention_mask=attention_mask)
    
    # extract features from first position in each sentence embedding provided by Distilbert
    features = last_hidden_states[0][:, 0, :].numpy()

    pcap_nlp_features = np.append(pcap_nlp_features, features, axis=0)
    
  if _set == 'both':  
    pd.DataFrame(pcap_nlp_features).to_csv('Generated_NLP_Features.csv', header=None, index=False) # save NLP features to a csv file
  elif _set == 'train':
    pd.DataFrame(pcap_nlp_features).to_csv('Generated_NLP_Training_Features.csv', header=None, index=False) # save NLP features to a csv file
  elif _set == 'test':
    pd.DataFrame(pcap_nlp_features).to_csv('Generated_NLP_Testing_Features.csv', header=None, index=False) # save NLP features to a csv file
  return pcap_nlp_features
