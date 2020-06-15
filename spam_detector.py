# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:50:05 2020

@author: kingslayer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

spam_df=pd.read_csv(r"emails.csv")

#Visualising
sns.countplot("spam",data=spam_df)
spam_df.info()

#Text Cleaning
"""import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,5728):
    review=re.sub("[^a-zA-Z]"," ",spam_df["text"][i])
    review=review.lower()
    ps=PorterStemmer()
    review=review.split()
    #review=[ps.stem(word) for word in review not in set(stopwords.words("english"))]
    review=" ".join(review)
    corpus.append(review)
 """   
 
 
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(spam_df["text"]).toarray()
y=spam_df["spam"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


#model


#Get No Spam as Ham
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))

#Good
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))

#Get no ham as spam
from sklearn.ensemble import RandomForestClassifier as RFR
classifier=RFR(n_estimators=100)

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))


sns.heatmap(cm,annot=True)