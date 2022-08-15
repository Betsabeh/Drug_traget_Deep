from IPython.core.display import TextDisplayObject
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf
import matplotlib.pyplot as plt
import re
nltk.download('punkt')
#-------------------------------------------
#1-read dataset
Data=pd.read_csv('movie.csv')
#print(Data.info)
Label=Data['sentiment']
#print(Label)
Label=(Label=='positive')
Label=np.int16(Label)
Text=Data['review']
#print(Label)
#----------------------------------------------
#2-Tokenization
tokens=[]
for sent in Text:
  #print(sent)
  temp=nltk.word_tokenize(sent,language='english')
  tokens.append(temp)
print(tokens)
#-------------------------------------------------
#3-Bag of word
count=CountVectorizer()
Text=np.array(Text)
bag=count.fit_transform(Text)
print("vocabulary")
Voc=count.vocabulary_
print(Voc)
#create vector
print('Sentence BOW')
BOW=bag.toarray()
print(BOW)
#print(BOW[0][474]) # numer of is in sentence 1
#---------------------------------------------
def preprocessing(txt):
  #print(txt)
  #remove html tag
  txt=re.sub('<[^>]*>','',txt)
  emotion=re.findall('(?::|;|=)(?:=)?(?:\)|\(D|P)',txt)
  txt=re.sub('[\W]+',' ',txt)
  #print(txt)
  txt=txt.lower()
  emotion=' '.join(emotion)
  emotion=emotion.replace('-','')
  txt=txt+emotion
  return txt

t=preprocessing(Text[0]) 
print(t)
#----------------------------------------------
# 4-calculate TF-IDF
tfidf=TfidfTransformer(use_idf=True,smooth_idf=True,norm="l2")
tfidf_value=tfidf.fit_transform(count.fit_transform(Text))
tfidf_value=tfidf_value.toarray()
print("TFIDFvalues:")
print(tfidf_value)
