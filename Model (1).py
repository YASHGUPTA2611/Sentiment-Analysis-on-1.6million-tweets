#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all important libraries which will be in use
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras


# In[2]:


df = pd.read_csv('training.1600000.processed.noemoticon.csv')


# In[3]:


df.head()


# In[4]:


df.columns = ['Target','Id', 'Time', 'Query', 'Name', 'Tweet']


# In[5]:


df


# In[6]:


df['Target'].value_counts()


# In[7]:


df.isnull().sum()


# In[8]:


tweets = df['Tweet']


# In[9]:


tweets[100]


# ## Normalizing the text data first

# In[10]:


import regex as re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 


# In[11]:


lemm = WordNetLemmatizer()


# In[ ]:


corpus = []
for i in range(len(df)):
    tweet = re.sub('[^a-zA-Z]', ' ' , tweets[i])
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [lemm.lemmatize(word) for word in tweet if word not in set(stopwords.words('english')) ]
    tweet = ' '.join(tweet)
    corpus.append(tweet)


# In[ ]:


corpus


# In[2]:


from keras.preprocessing.text import one_hot


# ## Doing one hot representation

# In[13]:


vocab_size = 10000


# In[ ]:


one_hot_rep = [one_hot(word,vocab_size) for word in corpus]


# ## Padding

# In[15]:


sent_len = 20


# In[3]:


from keras.preprocessing.sequence import pad_sequences


# In[ ]:


embedded_docs = pad_sequences(one_hot_rep, padding='pre', max_len=sent_len )


# In[9]:


# Importing libraries which helps in making Bidirectional model
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import Dense


# ## Applying Embedding and Bidirectional layer

# In[ ]:


dim = 100
model = Sequential()
model.add(Embedding(vocab_size,dim,input_length = sent_len))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[ ]:


X = np.array(embedded_docs)
y = np.array(df['Target'])


# ## Train-test split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.43, random_state=42)


# ## Fitting the model

# In[ ]:


model.fit(X_trin,y_train, validation_data=(X_test,y_test),epochs=100, batch_size=64)


# In[ ]:


# Predicting y_test values
y_pred = model.predict(X_test)


# # Performance Matrix

# In[17]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report 


# In[ ]:


confusion_matrix(y_pred,y_test)


# In[ ]:


accuracy_score(y_pred,y_test)


# In[ ]:


classification_report(y_pred,y_test)

