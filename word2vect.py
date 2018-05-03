
import pandas as pd

def word2vector():
    df1= pd.read_pickle('review1')
    df2= pd.read_pickle('review2')
    df3=pd.read_pickle('review3')
    df4=pd.read_pickle('review4')
    df5=pd.read_pickle('review5')
    df = pd.concat([df1,df2, df3,df4,df5])
#    print('done')
#    df = df1
    import re
    print('start')
    #Steps 2. Preprocess text 
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
    
    
    #Steps 3. vector representation of words 
    sentences = df['words'].tolist()
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)
    
    # Set values for various parameters
    num_features = 100    # Word vector dimensionality                      
    min_word_count = 10   # Minimum word count                        
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words
    
    # Initialize and train the model
    from gensim.models import word2vec
    print ("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)
    
    model.init_sims(replace=True)
    
    model_name = "100features_10minwords_10context"
    model.save(model_name)
    return df