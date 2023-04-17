#!/usr/bin/env python
# coding: utf-8

# # TWITTER SENTIMENT ANALYSIS 

# ### IMPORTING THE REQUIRED PACKAGES AND MODULES

# In[68]:


import pandas as pd              #for data analysis and basic operations
import numpy as np               #for data analysis and basic operations
import re                        #for regex
import seaborn as sns            #for data visualisation
import matplotlib.pyplot as plt  #for data visualisation
from matplotlib import style     #style for the plot
style.use('ggplot')
from textblob import TextBlob                    #process the textual data
from nltk.tokenize import word_tokenize          #for tokenization
from nltk.stem import PorterStemmer              #for stemming
from nltk.corpus import stopwords                #to remove stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer                          #to vectorize the text document
from sklearn.model_selection import train_test_split                                 #to split the data into training and testing data
from sklearn.linear_model import LogisticRegression                                  #to perform logistic regression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay # for evaluating accuracy and displaying matrix for evaluating the model


# ### READING  AND DISPLAYING THE DATASET

# In[72]:


df = pd.read_csv('LGBT_Tweets.csv')


# In[73]:


df.head(20)


# In[74]:


df.info()          #to describe the dataset


# In[75]:


df.columns        #to obtain the column names


# ### CREATING A NEW DATAFRAME FOR TWEETS

# In[76]:


text_df = df.drop(['Unnamed: 0', 'date', 'time', 'id', 'language', 'replies_count',
       'retweets_count', 'likes_count'], axis=1) #drop all columns except the "text" column
text_df.head(20)                                   #new dataframe


# In[77]:


print(text_df['text'].iloc[0],"\n")   #analyse data in the "text" dataframe
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")


# In[78]:


text_df.info() #to describe the new dataframe


# ### CONVERSION OF RAW DATA TO USEFUL DATA

# In[79]:


def data_processing(text):            #to convert the raw data into usable format
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)  #remove URLs
    text = re.sub(r'\@w+|\#','',text) #remove hashtags 
    text = re.sub(r'[^\w\s]','',text) #remove punctuation marks
    text_tokens = word_tokenize(text) #remove stopwords
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


# In[80]:


text_df.text = text_df['text'].apply(data_processing) 


# In[81]:


text_df = text_df.drop_duplicates('text')  #remove duplicate data


# ### STEMMING 

# In[82]:


stemmer = PorterStemmer()  #stemming for reducing tokenized words to their root form
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


# In[83]:


text_df['text'] = text_df['text'].apply(lambda x: stemming(x)) #apply stemming to the processed data


# In[84]:


text_df.head(20)


# In[85]:


print(text_df['text'].iloc[0],"\n") 
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")


# In[86]:


text_df.info() #updated dataframe


# ### CALCULATING THE POLARITY

# In[87]:


def polarity(text):                          #to calculate polarity using TextBlob
    return TextBlob(text).sentiment.polarity


# In[88]:


text_df['polarity'] = text_df['text'].apply(polarity)


# In[89]:


text_df.head(20)


# ### OBTAINING THE SENTIMENT LABEL FOR EACH TWEET

# In[90]:


def sentiment(label):     #to define the sentiment of a particular tweet
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"


# In[91]:


text_df['sentiment'] = text_df['polarity'].apply(sentiment)


# In[135]:


text_df.head(20)


# ### VISUALIZATION OF DATA USING COUNTPLOT AND PIE CHART

# In[136]:


fig = plt.figure(figsize=(5,5))                #data visualization using countplot
sns.countplot(x='sentiment', data = text_df)


# In[137]:


fig = plt.figure(figsize=(7,7))                #data visualization using pie chart
colors = ("pink", "turquoise", "gold")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = text_df['sentiment'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,startangle=90, wedgeprops = wp, explode = explode, label='')
plt.title('*** VISUALIZATION OF SENTIMENTS ***')


# ### BUILDING THE MODEL

# In[138]:


vect = CountVectorizer(ngram_range=(1,2)).fit(text_df['text'])  #count vectorization for the model


# In[148]:


feature_names = vect.get_feature_names()                        #get and print the first 30 features
print("Total number of features are: {}\n".format(len(feature_names)))
print("The first 30 features are:\n {}".format(feature_names[:30]))


# In[149]:


X = text_df['text']                 #separation of data into x and y for transformation
Y = text_df['sentiment']
X = vect.transform(X)


# In[150]:


#split the data into training and testing data

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[151]:


#print the size of training and testing data

print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))

print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))


# In[159]:


#to get rid of warnings

import warnings
warnings.filterwarnings('ignore')


# ### TRAINING THE MODEL

# In[160]:


#train the data on logicticregression model

logreg = LogisticRegression()

logreg.fit(x_train, y_train) #fit the data
logreg_pred = logreg.predict(x_test) #predict the value for test data

logreg_acc = accuracy_score(logreg_pred, y_test) #calculate the accuracy for the model
print("Accuracy of the model is: {:.2f}%".format(logreg_acc*100)) 


# ### OBTAINING THE CLASSIFICATION REPORT AND PRINTING THE RELEVANT CONFUSION MATRIX

# In[164]:


#display the confusion matrix

style.use('classic')
cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=logreg.classes_)
disp.plot()


# In[165]:


#print the confusion matrix and classification report


print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))


# In[ ]:




