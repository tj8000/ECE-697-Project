# Function that reads in timestep data, trains an lstm model, and predicts using a single dataset with an 80/20 train/test split 
def train_test_split_model(_timestep,  _timestep_method, _feature_extraction_method, _pcap_label_file_path, _lr, _batch_size, _epochs):
  # import necessary packages
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from tensorflow import keras
  from keras.models import Sequential
  from keras.layers import Dense
  from keras.layers import LSTM
  from keras.layers import Dropout
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import classification_report

  if _timestep_method == 'Normal':
    input_data = np.array(pd.read_csv('Generated_'+_feature_extraction_method+'_Timestep_Features.csv', header=None)) # read in lstm input data if timestep method is normal
  elif _timestep_method == 'Flow':
    input_data = np.array(pd.read_csv('Generated_'+_feature_extraction_method+'_Flow_Timestep_Features.csv', header=None)) # read in lstm input data if timestep method is flow

  num_features = int(np.shape(input_data)[1]/_timestep) # determine the number of features for each input

  input_data = input_data.reshape((len(input_data), _timestep, num_features)) # reshape input data into 3D array (num samples, timestep, num features)

  labels = np.array(pd.read_csv(_pcap_label_file_path, header=None)).reshape(-1, 1) # read in dataset labels

  X_train, X_test, y_train, y_test = train_test_split(input_data, labels, test_size=0.2, random_state=13) # perform an 80/20 train test split on input data and labels

  # build LSTM model
  model = Sequential()
  model.add(LSTM(128, input_shape=(None, num_features), return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(128, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(128, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(128))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=_lr)) # compile LSTM model

  history = model.fit(X_train, y_train, batch_size=_batch_size, epochs=_epochs, shuffle=False) # train model

  # plot the training loss and save
  loss = history.history['loss']
  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, 'r', label='Training loss')
  plt.title('Training loss')
  plt.legend()
  plt.savefig('LSTM_Training_Loss_Curve.png')
  
  print('\n')

  model.save('LSTM_Model') # save the LSTM model
  
  y_hat_test = model.predict(X_test) # use lstm model to predict labels for input data

  # threshold predicted labels
  y_hat_test[y_hat_test < 0.5] = 0
  y_hat_test[y_hat_test >= 0.5] = 1

  if _timestep_method == 'Normal':
    pd.DataFrame(y_hat_test).to_csv('Generated_'+_feature_extraction_method+'_Timestep_Features_Predicted_Labels.csv', header=None, index=False) # save predicted labels to a csv file
  elif _timestep_method == 'Flow':
    pd.DataFrame(y_hat_test).to_csv('Generated_'+_feature_extraction_method+'_Flow_Timestep_Features_Predicted_Labels.csv', header=None, index=False) # save predicted labels to a csv file

  # create confusion matrix and print out results
  cf = confusion_matrix(y_test, y_hat_test)
  print('\n')
  print('Confusion Matrix: ')
  print(cf)
  print('\n')

  # create classification report and print out results
  report = classification_report(y_test, y_hat_test)
  print('Classification Report: ')
  print(report)


# Function that reads in timestep data, trains an lstm model, and predicts using a separate dataset for training and testing 
def train_test_separate_model(_timestep,  _timestep_method, _feature_extraction_method, _train_pcap_label_file_path, _test_pcap_label_file_path, _lr, _batch_size, _epochs):
  # import necessary packages
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  import matplotlib.pyplot as plt
  from tensorflow import keras
  from keras.models import Sequential
  from keras.layers import Dense
  from keras.layers import LSTM
  from keras.layers import Dropout
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import classification_report

  if _timestep_method == 'Normal':
    input_train_data = np.array(pd.read_csv('Generated_'+_feature_extraction_method+'_Timestep_Training_Features.csv', header=None)) # read in lstm input training data if timestep method is normal
    input_test_data = np.array(pd.read_csv('Generated_'+_feature_extraction_method+'_Timestep_Testing_Features.csv', header=None)) # read in lstm input testing data if timestep method is normal
  elif _timestep_method == 'Flow':
    input_train_data = np.array(pd.read_csv('Generated_'+_feature_extraction_method+'_Flow_Timestep_Training_Features.csv', header=None)) # read in lstm input training data if timestep method is flow
    input_test_data = np.array(pd.read_csv('Generated_'+_feature_extraction_method+'_Flow_Timestep_Testing_Features.csv', header=None)) # read in lstm input testing data if timestep method is flow

  num_features = int(np.shape(input_train_data)[1]/_timestep) # determine the number of features for each input

  input_train_data = input_train_data.reshape((len(input_train_data), _timestep, num_features)) # reshape input training data into 3D array (num samples, timestep, num features)
  input_test_data = input_test_data.reshape((len(input_test_data), _timestep, num_features)) # reshape input testing data into 3D array (num samples, timestep, num features)

  train_labels = np.array(pd.read_csv(_train_pcap_label_file_path, header=None)).reshape(-1, 1) # read in training dataset labels
  test_labels = np.array(pd.read_csv(_train_pcap_label_file_path, header=None)).reshape(-1, 1) # read in testing dataset labels

  # build LSTM model
  model = Sequential()
  model.add(LSTM(128, input_shape=(None, num_features), return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(128, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(128, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(128))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=_lr)) # compile LSTM model

  history = model.fit(input_train_data, train_labels, batch_size=_batch_size, epochs=_epochs, shuffle=False) # train model
  
  # plot the training loss and save
  loss = history.history['loss']
  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, 'r', label='Training loss')
  plt.title('Training loss')
  plt.legend()
  plt.savefig('LSTM_Training_Loss_Curve.png')

  print('\n')

  model.save('LSTM_Model') # save the LSTM model
  
  pred_test_labels = model.predict(input_test_data) # use lstm model to predict labels for input data

  # threshold predicted labels
  pred_test_labels[pred_test_labels < 0.5] = 0
  pred_test_labels[pred_test_labels >= 0.5] = 1

  if _timestep_method == 'Normal':
    pd.DataFrame(pred_test_labels).to_csv('Generated_'+_feature_extraction_method+'_Timestep_Features_Predicted_Labels.csv', header=None, index=False) # save predicted labels to a csv file
  elif _timestep_method == 'Flow':
    pd.DataFrame(pred_test_labels).to_csv('Generated_'+_feature_extraction_method+'_Flow_Timestep_Features_Predicted_Labels.csv', header=None, index=False) # save predicted labels to a csv file

  # create confusion matrix and print out results
  cf = confusion_matrix(test_labels, pred_test_labels)
  print('\n')
  print('Confusion Matrix: ')
  print(cf)
  print('\n')

  # create classification report and print out results
  report = classification_report(test_labels, pred_test_labels)
  print('Classification Report: ')
  print(report)
  