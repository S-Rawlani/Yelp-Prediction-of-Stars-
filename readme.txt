/**
*Group Project by: 
* Name: Apurva Mithal (axm174531)
* Name: Sampoorna Karinje (sxk174630)
* Name: Simran Rawlani (sxr174130)
* Topic: "Predicting Yelp stars from Reviews"
*/

I. Files List
 readme.txt --- A description of the file in the unzipped folders.
 report.pdf --- A detailed report on the topic.
 Python files: readingData.py, word2vect.py, text2vector.py, dataPreparation.py, neuralNetwork.py, main.py
 dataset:  
   Please download the dataset from the link mentioned in the report.
  We are not able to upload it on elearning due to huge dataset file. 
   So just the python files are included.
 
 
II. How to Run the Program
Open a terminal(command prompt- cmd), go to the current path.  
        1. Run readingData.py file by changing in main.py and then type:
             - python main.py
        Output will be- 46 review files
       2. Run word2vect.py file by changing in main.py and then type:
           - python main.py
        Output will be-  a file named'100features_10minwords_10context'
      3. Run text2vector.py file by changing in main.py and then type:
             - python main.py
        Output will be- a file named 'business data'
      4. Run dataPreparation.py file by changing in main.py and then type:
             - python main.py
        Output will be- two files named 'X.npy', 'y.npy'
      5. Run neuralNetwork.py file by changing in main.py and then type:
             - python main.py
The program will take around 15 minutes to run. 
The output displays the loss functions for all epochs and the accuracy.