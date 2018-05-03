
import numpy as np
import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
import pandas as pd
from sklearn.metrics import roc_auc_score

def neuralNetwork():
    embedded_vect = 100 #vector embedding of the word which is of size 100
    num_hidden_layers =100
    constant_size=30
    X=np.load('X.npy')
    y=np.load('y.npy')
    print('Building LSTM model starts')
    model = Sequential()
    model.add(LSTM(input_dim=embedded_vect, output_dim=num_hidden_layers, \
        input_length=constant_size, return_sequences=True))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim=num_hidden_layers, output_dim=num_hidden_layers, \
                   input_length=constant_size, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    print('Training model starts' )        
    
    model.fit(X[0:44400], y[0:44400], batch_size=128, nb_epoch=10)
    model.save_weights('model_weight.hdf5',overwrite=True)

    probability_predictions = model.predict(X[44400:], verbose=0) 
    print('Successfull')
    
    real = np.argmax(y[44400:], axis =1)+1
    predicted = np.argmax(probability_predictions, axis=1)+1
    df = pd.DataFrame(real, columns =['real'])
    df['predicted'] = predicted
    df[df['real'] == df['predicted']]
    
    #Accuracy Calculation using AUC score
    
    print('Calculating AUC Score / Accuracy Check....')
    df['correct_prediction'] = np.where(df['real'] == df['predicted'], 1, 0)
    df['score'] = np.amax(probability_predictions, axis=1)
    print('Accuracy is')
    print(roc_auc_score(df['correct_prediction'], df['score']))