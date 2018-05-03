
import gensim

def textToVector():
    model = gensim.models.Word2Vec.load("100features_10minwords_10context")
    print('done')
    
    # create the pandas dataframe - working with 1 set of files i.e review due to limitation of memory
    import pandas as pd
    import re
    print('start')
    df= pd.read_pickle('review46')
    #Steps 1. Preprocess text 
    text_list = df['text'].tolist()
    sent_list=[]
    for item in range(0,len(text_list)):
        sent_tmp=[]
        try:
            sent_tmp= re.split('\W', text_list[item])
        except:
            sent_tmp.append('NA')
        sent_list.append([i.strip().lower() for i in sent_tmp if str(i)!='' ])
    df['words']=sent_list
    print('done')
    
    #Create the vector representation of the words and store it in pandas
    import numpy as np
    words_list= df['words'].tolist()
    words_list_vector=[]
    for item1 in words_list:
        word_tmp=[]
        for item2 in item1:
            try:
                word_tmp.append(model[item2])
            except:
                word_tmp.append(np.array([0 for i in range(0,101)]))
        words_list_vector.append(word_tmp)
    df['words_vector']=words_list_vector
    df.to_pickle('business_data')
    
    print('done')