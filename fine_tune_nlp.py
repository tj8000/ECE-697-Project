# Function to pip install any needed packages
def install(package):
  import subprocess
  import sys
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])


# Function to prompt user for pcap csv and pcap label csv file paths for nlp fine tunning
def fine_tune_nlp_file_paths_prompt():
  import pathlib

  valid = 0
  while valid == 0:
    pcap_csv_file_path = input('Enter pcap csv file path NLP fine tuning: ') # directory path to where the pcap csv file is located
    if pathlib.Path(pcap_csv_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0
      print('Invalid file path.')

  valid = 0
  while valid == 0:
    pcap_label_file_path = input('Enter pcap label file path for NLP fine tuning: ') # directory path to where the pcap label file is located
    if pathlib.Path(pcap_label_file_path).exists() == True:
      valid = 1
      break
    else:
      valid = 0
      print('Invalid file path.')

  return pcap_csv_file_path, pcap_label_file_path


# This function references code developed at
# https://huggingface.co/docs/transformers/tasks/sequence_classification
# This function was apapted from nlp_feature_data_generation_functions.py
# Function to fine tune the Hugging Face Distilbert model 
def fine_tune_nlp(_pcap_csv_file_path, _pcap_label_file_path):
  install('torch-summary') # install torch-summary package
  install('transformers') # install transformers package
  install('datasets') # install datasets package

  # import necessary packages
  import numpy as np
  import pandas as pd
  import torch
  import transformers as ppb
  import warnings
  warnings.filterwarnings('ignore')
  import datasets 
  from datasets import Dataset
  from sklearn.model_selection import train_test_split
  from transformers import AutoTokenizer
  from transformers import DataCollatorWithPadding
  from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
  
  data = pd.read_csv(_pcap_csv_file_path) # read in input data
  labels = pd.read_csv(_pcap_label_file_path, header=None) # read in input labels

  input_data = data['Info'] # extract the Info field from the input data
  input_labels = np.squeeze(labels) # flatten label array to 1D
  

  X_train, X_test, y_train, y_test = train_test_split(input_data, input_labels, test_size=0.3, random_state=42) # split the dataset into train/test for the purpose of creating a data dictionary to fine tune the NLP model

  # train portion of the data dictionary 
  train_data = pd.DataFrame({ #https://stackoverflow.com/questions/71618974/convert-pandas-dataframe-to-datasetdict
      "label" : y_train, #https://stackoverflow.com/questions/71618974/convert-pandas-dataframe-to-datasetdict
      "text" : X_train #https://stackoverflow.com/questions/71618974/convert-pandas-dataframe-to-datasetdict
  }) 

  # test portion of the data dictionary
  test_data = pd.DataFrame({ #https://stackoverflow.com/questions/71618974/convert-pandas-dataframe-to-datasetdict
      "label" : y_test, #https://stackoverflow.com/questions/71618974/convert-pandas-dataframe-to-datasetdict
      "text" : X_test #https://stackoverflow.com/questions/71618974/convert-pandas-dataframe-to-datasetdict
  })

  # create the data dictionary using both the train and test portion
  train_dataset = Dataset.from_dict(train_data) #https://stackoverflow.com/questions/71618974/convert-pandas-dataframe-to-datasetdict
  test_dataset = Dataset.from_dict(test_data)  #https://stackoverflow.com/questions/71618974/convert-pandas-dataframe-to-datasetdict
  my_dataset_dict = datasets.DatasetDict({"train":train_dataset, "test":test_dataset}) #https://stackoverflow.com/questions/71618974/convert-pandas-dataframe-to-datasetdict

  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # load Distilbert tokenizer from Hugging Face #https://huggingface.co/docs/transformers/tasks/sequence_classification

  def preprocess_function(examples): #https://huggingface.co/docs/transformers/tasks/sequence_classification
    return tokenizer(examples["text"], truncation=True)  #https://huggingface.co/docs/transformers/tasks/sequence_classification


  tokenized = my_dataset_dict.map(preprocess_function, batched=True) # tokenize the Info strings from the newly created data dictionary  #https://huggingface.co/docs/transformers/tasks/sequence_classification

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # load model for fine tuning #https://huggingface.co/docs/transformers/tasks/sequence_classification

  model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2) #https://huggingface.co/docs/transformers/tasks/sequence_classification

  training_args = TrainingArguments( #https://huggingface.co/docs/transformers/tasks/sequence_classification
    output_dir="./results", #https://huggingface.co/docs/transformers/tasks/sequence_classification
    learning_rate=2e-5, #https://huggingface.co/docs/transformers/tasks/sequence_classification
    per_device_train_batch_size=15, #https://huggingface.co/docs/transformers/tasks/sequence_classification
    per_device_eval_batch_size=15, #https://huggingface.co/docs/transformers/tasks/sequence_classification
    num_train_epochs=1,  #https://huggingface.co/docs/transformers/tasks/sequence_classification
    weight_decay=0.01, #https://huggingface.co/docs/transformers/tasks/sequence_classification
  )

  trainer = Trainer( #https://huggingface.co/docs/transformers/tasks/sequence_classification
    model=model, #https://huggingface.co/docs/transformers/tasks/sequence_classification
    args=training_args, #https://huggingface.co/docs/transformers/tasks/sequence_classification
    train_dataset=tokenized["train"], #https://huggingface.co/docs/transformers/tasks/sequence_classification
    eval_dataset=tokenized["test"], #https://huggingface.co/docs/transformers/tasks/sequence_classification
    tokenizer=tokenizer, #https://huggingface.co/docs/transformers/tasks/sequence_classification
    data_collator=data_collator, #https://huggingface.co/docs/transformers/tasks/sequence_classification
  )

  trainer.train() # fine tune Hugging Face Distilbert model #https://huggingface.co/docs/transformers/tasks/sequence_classification

  model.save_pretrained('fine_tuned_nlp')


if __name__ == '__main__':
  pcap_csv_file_path, pcap_label_file_path = fine_tune_nlp_file_paths_prompt() # get input and label data for fine tunning

  fine_tune_nlp(pcap_csv_file_path, pcap_label_file_path) # fine tune Distilbert NLP model
  
