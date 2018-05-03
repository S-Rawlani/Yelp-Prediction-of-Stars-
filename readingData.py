
#Read the review data and store it in pandas
import json
import pandas as pd
import sys

def readData():
    counter =0
    with open('review.json', 'r', encoding='UTF-8') as infile:
        review =[]
        for line in infile:
            review.append(json.loads(line))
            if sys.getsizeof(review)> 1000000:
                counter = counter +1
                df=pd.DataFrame.from_dict(review)
                df.drop(['review_id','user_id','business_id','date','useful', 'funny', 'cool'], axis=1).to_pickle('review'+ str(counter))
                review =[]
                del df
    counter = counter +1
    df=pd.DataFrame.from_dict(review)
    df.drop(['review_id','user_id','business_id','date','useful', 'funny', 'cool'], axis=1).to_pickle('review'+ str(counter))
#    print(df)
#    df1= pd.read_pickle('review1')
#    print(df1)
    return df