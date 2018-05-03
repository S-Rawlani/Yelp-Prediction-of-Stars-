
import pandas as pd

def data_prep():
    df = pd.read_pickle('business_data')
    df= df.sample(frac=1)  #Shuffling the dataset
    print('file_read')
    
    #Preparing the input in numpy array
    import numpy as np
    X_mod = df['words_vector'].tolist()
    y_mod=  np.array(df['stars'].tolist())
    del df
    #print('X_mod ', X_mod, 'y_mod ', y_mod )
    
    #Preparing the input in numpy array
    import numpy as np
    import sys
    max_words=30# This number can be increased but limited due to memory constraints
    vector_size=100
    X = np.empty((len(X_mod),max_words,vector_size))
    for index1 in range(0,len(X_mod)):
        if len(X_mod[index1])< max_words:
            for index2 in range(0,max_words - len(X_mod[index1])):
                 for index3 in range(0,vector_size):
                        X[index1][index2][index3]=0
            for index2 in range(0,len(X_mod[index1])):
                 for index3 in range(0,vector_size):
                        X[index1][max_words - len(X_mod[index1]) + index2][index3]=X_mod[index1][index2][index3]
        else:
            for index2 in range(0,max_words):
                 for index3 in range(0,vector_size):
                        X[index1][index2][index3]=X_mod[index1][index2][index3]
#    print('done')
    
    
    #Preparing the output vector
    y = np.zeros((len(y_mod),5))
    for item in range(0,len(y_mod)):
        y[item][y_mod[item]-1] = 1
    #print('X', X, 'y',y)
    
    #Freeing memory
    del X_mod
    del y_mod
    np.save('X',X)
    np.save('y',y)
    print('done')