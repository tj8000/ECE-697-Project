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
  train_data = pd.DataFrame({
      "label" : y_train,
      "text" : X_train
  }) 

  # test portion of the data dictionary
  test_data = pd.DataFrame({
      "label" : y_test,
      "text" : X_test
  })

  # create the data dictionary using both the train and test portion
  train_dataset = Dataset.from_dict(train_data)
  test_dataset = Dataset.from_dict(test_data) 
  my_dataset_dict = datasets.DatasetDict({"train":train_dataset, "test":test_dataset})

  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # load Distilbert tokenizer from Hugging Face

  def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True) 


  tokenized = my_dataset_dict.map(preprocess_function, batched=True) # tokenize the Info strings from the newly created data dictionary 

  data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # load model for fine tuning

  model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

  training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=15,
    per_device_eval_batch_size=15,
    num_train_epochs=1, 
    weight_decay=0.01,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
  )

  trainer.train() # fine tune Hugging Face Distilbert model

  model.save_pretrained('fine_tuned_nlp')


if __name__ == '__main__':
  pcap_csv_file_path, pcap_label_file_path = fine_tune_nlp_file_paths_prompt() # get input and label data for fine tunning

  fine_tune_nlp(pcap_csv_file_path, pcap_label_file_path) # fine tune Distilbert NLP model
  